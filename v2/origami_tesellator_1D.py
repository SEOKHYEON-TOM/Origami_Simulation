"""1D strip assembly with explicit crease maps and shared path selection.

User sectors are theta0..theta3 between c0/c1,..,c3/c0. Public fold
getters use c0..c3; rhos retains the historical solver-order storage.
Contact checks individual zero-thickness vertices, not global strip collision.
"""

from dataclasses import dataclass, field
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from origami_simulator import (
    OrigamiDegree4Simulator,
    PathError,
    wrap_angle,
    lift_angles,
)
from origami_kinematics import classify_geometry, closure_error


def crease_frame(axis, normal):
    """Proper orthonormal frame with X on a crease and Z on its panel normal."""

    def unit(vector):
        vector = np.asarray(vector, float)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError("finite 3D frame vectors required")
        scale = float(np.max(abs(vector)))
        if scale == 0:
            raise ValueError("degenerate frame vector")
        value = vector / scale
        return value / np.linalg.norm(value)

    x = unit(axis)
    normal = unit(normal)
    projected = normal - x * np.dot(x, normal)
    if np.linalg.norm(projected) < 1e-12:
        raise ValueError("degenerate panel normal")
    z = unit(projected)
    y = unit(np.cross(z, x))
    return np.column_stack((x, y, z))


@dataclass
class UnitState:
    unit_index: int
    cell_index: int
    solver_rhos: np.ndarray
    folds: np.ndarray
    mode_id: str
    branch_id: str
    tangent: np.ndarray | None
    residual: float
    interval: tuple | None
    family_free_crease: int | None = None
    status: str = "OK"
    events: tuple = ()
    output_crease: int = 2

    @property
    def input_angle(self):
        return float(self.folds[0])

    @property
    def output_angle(self):
        return float(self.folds[self.output_crease])


@dataclass
class StripResult:
    input_angle: float
    total_units: int
    states: list = field(default_factory=list)
    vertices: list = field(default_factory=list)
    faces: list = field(default_factory=list)
    termination: dict | None = None
    joints: list = field(default_factory=list)
    global_contact_checked: bool = False
    contact_scope: str = "off"
    requested_input: float | None = None
    continuation_steps: int = 0

    @property
    def valid_units(self):
        return len(self.states)

    @property
    def complete(self):
        return self.termination is None and self.valid_units == self.total_units

    @property
    def status(self):
        return "OK" if self.termination is None else self.termination["reason"]


class OrigamiTesellator1D:
    def __init__(
        self,
        cell_configs,
        num_periods,
        lengths=(1.0, 1.0, 1.0),
        scale_factor=1.0,
        contact=True,
        verbose=True,
        *,
        sigma_role="opposite",
        joint_policy="axis",
        atlas_resolution=129,
    ):
        """Build cell definitions; select each unit's path when queried.

        sigma_role='opposite' means sign(rho0)*sign(rho2)=sigma, including
        iout=1/3. 'output' explicitly requests the legacy output-sign hint.
        A hint that leaves multiple modes produces AMBIGUOUS. Configs may
        specify mode_id, branch_id, initial_state (four c-order radians),
        or free_angle (the explicitly reported family's free-fold radians).
        joint_policy='axis' preserves legacy finite-edge lengths; 'shared_edge'
        additionally requires matching finite endpoints. No length is changed.
        """
        if not isinstance(cell_configs, (list, tuple)) or not cell_configs:
            raise ValueError("cell_configs must be a nonempty list")
        if (
            isinstance(num_periods, (bool, np.bool_))
            or not isinstance(num_periods, (int, np.integer))
            or num_periods < 1
        ):
            raise ValueError("num_periods must be a positive integer")
        if not np.isfinite(scale_factor) or scale_factor <= 0:
            raise ValueError("scale_factor must be finite and positive")
        if sigma_role not in ("opposite", "output"):
            raise ValueError("sigma_role must be 'opposite' or 'output'")
        if joint_policy not in ("axis", "shared_edge"):
            raise ValueError("joint_policy must be 'axis' or 'shared_edge'")
        if (
            isinstance(atlas_resolution, bool)
            or not isinstance(atlas_resolution, (int, np.integer))
            or atlas_resolution < 8
        ):
            raise ValueError("atlas_resolution must be an integer >=8")
        self.N = len(cell_configs)
        self.cell_configs = deepcopy(list(cell_configs))
        self.num_periods = int(num_periods)
        self.total_units = self.num_units = self.N * self.num_periods
        self.scale_factor = float(scale_factor)
        self.contact = bool(contact)
        self.verbose = bool(verbose)
        self.lengths = lengths
        self.sigma_role = sigma_role
        self.joint_policy = joint_policy
        self.atlas_resolution = int(atlas_resolution)
        self._models = {}
        self._curve_cache = {}
        self.last_result = None
        self.last_attempt = None
        self.last_error = None
        self.unit_states = []
        self.global_vertices = []
        self.global_faces = []
        self.unit_iouts = []
        self.rhos = np.full((self.total_units, 4), np.nan)
        self.cells = []
        global_lengths = np.asarray(lengths, float)
        if global_lengths.shape == (3,):
            global_lengths = np.tile(global_lengths, (self.N, 1))
        if global_lengths.shape != (self.N, 3):
            raise ValueError("lengths must have shape (3,) or (N,3)")
        for i, cfg in enumerate(self.cell_configs):
            if not isinstance(cfg, dict) or "alphas" not in cfg:
                raise ValueError("each cell needs an alphas entry")
            self.cells.append(
                self._init_cell(cfg, cfg.get("lengths", global_lengths[i]))
            )
        self._print_init_summary()

    def _init_cell(self, cfg, lengths):
        a = np.array(cfg["alphas"], float, copy=True)
        geometry = classify_geometry(a)
        if geometry["degenerate_sectors"]:
            raise ValueError("UNSUPPORTED_DEGENERATE_SECTOR: 180 degree sectors")
        sigma = cfg.get("sigma", 1)
        iout = cfg.get("iout", 2)
        if sigma not in (-1, 1) or isinstance(sigma, bool):
            raise ValueError("sigma must be +1 or -1")
        if iout not in (1, 2, 3) or isinstance(iout, bool):
            raise ValueError("iout must be 1,2 or 3")
        role = cfg.get("sigma_role", self.sigma_role)
        if role not in ("opposite", "output"):
            raise ValueError("invalid sigma_role")
        ls = np.array(lengths, float)
        if ls.shape != (3,) or not np.isfinite(ls).all() or np.any(ls <= 0):
            raise ValueError("three finite positive crease lengths required")
        geom = "euclidean" if geometry["case"] == "developable" else geometry["case"]
        roll = 0 if geom == "elliptic" else -1
        drive = 1 if geom == "elliptic" else 4
        solver_angles = np.roll(a, roll)
        key = (tuple(solver_angles), drive)
        if key not in self._models:
            self._models[key] = OrigamiDegree4Simulator(
                solver_angles, driving_crease=drive, contact=False, verbose=False
            )
        solver = self._models[key]
        clv = {k: (k + 1 if geom == "elliptic" else (k - 1) % 4 + 1) for k in range(4)}
        indices = tuple(clv[k] - 1 for k in range(4))
        if (
            indices[0] != solver.drive_index
            or solver.geometry["case"] != geometry["case"]
        ):
            raise ValueError("inconsistent geometry/drive mapping")
        angles = np.deg2rad(a)
        A = B = None
        if iout == 2:
            t0, t1, t2, t3 = angles
            denominator = np.sin(t1) * np.sin(t2)
            if abs(denominator) > 1e-13:
                A = float(np.sin(t3) * np.sin(t0) / denominator)
                B = float(
                    (np.cos(t1) * np.cos(t2) - np.cos(t3) * np.cos(t0)) / denominator
                )
        initial = cfg.get("initial_state")
        if initial is not None:
            initial = np.array(initial, float)
            if initial.shape != (4,) or not np.isfinite(initial).all():
                raise ValueError("initial_state needs four finite c0..c3 radians")
        if cfg.get("free_angle") is not None and not np.isfinite(
            float(cfg["free_angle"])
        ):
            raise ValueError("free_angle must be finite")
        for key in ("mode_id", "branch_id"):
            if cfg.get(key) is not None and not isinstance(cfg[key], str):
                raise ValueError(key + " must be a string")
        for array in (a, angles, ls):
            array.setflags(write=False)
        return dict(
            alphas_deg=a,
            alphas=angles,
            geometry=geometry,
            geom=geom,
            roll=roll,
            solver=solver,
            CLV=clv,
            fold_indices=indices,
            output_solver_idx=indices[iout],
            sigma=int(sigma),
            iout=int(iout),
            lengths=tuple(ls),
            A=A,
            B=B,
            is_flat_foldable=geometry["euclidean_flat_foldable"],
            sigma_role=role,
            mode_id=cfg.get("mode_id"),
            branch_id=cfg.get("branch_id"),
            initial_state=initial,
            free_angle=cfg.get("free_angle"),
        )

    @staticmethod
    def _canonical(cell, q):
        return np.asarray(q, float)[list(cell["fold_indices"])].copy()

    @staticmethod
    def _solver_order(cell, folds):
        q = np.empty(4)
        q[list(cell["fold_indices"])] = folds
        return q

    @staticmethod
    def _mode_name(cell, name):
        if name and name.startswith("E_unique_e"):
            name = name[2:]
        if name and name.startswith("unique_e"):
            solver_index = int(name[len("unique_e") :]) - 1
            return "unique_c" + str(cell["fold_indices"].index(solver_index))
        if name and name.startswith(("ff_", "collinear_")):
            head, _, tail = name.rpartition("_c")
            if tail.isdigit():
                name = head
        return name

    def _ensure_atlas(self, cell):
        model = cell["solver"]
        if not model.branches:
            model.run_simulation(self.atlas_resolution)
        return model.branches

    def _branch_state(self, cell, branch, drive):
        model = cell["solver"]
        branches = self._ensure_atlas(cell)
        if branch.startswith("unique_c"):
            try:
                crease = int(branch[len("unique_c") :])
            except ValueError as error:
                raise PathError(
                    "MODE_UNAVAILABLE", "invalid unique-crease alias"
                ) from error
            if crease not in range(4):
                raise PathError("MODE_UNAVAILABLE", "unique crease must be c0..c3")
            branch = "E_unique_e" + str(cell["fold_indices"][crease] + 1)
        matches = (
            [branches[branch]]
            if branch in branches
            else [
                b
                for b in branches.values()
                if b.mode_id == branch or b.id.startswith(branch + "_c")
            ]
        )
        if not matches:
            raise PathError("MODE_UNAVAILABLE", "unknown branch " + str(branch))
        target = float(wrap_angle(drive))
        if drive > 0 and abs(target + np.pi) < 2e-14:
            target = float(np.pi)
        reached = []
        failures = []
        for b in matches:
            if b.family is not None:
                continue
            if abs(drive) <= np.pi + 1e-12 and not any(
                lo - 1e-8 <= target <= hi + 1e-8 for lo, hi in b.reachable_intervals
            ):
                continue
            reference = (
                target
                if abs(drive) <= np.pi + 1e-12
                else float(np.clip(drive, *b.reachable_intervals[0]))
            )
            ids = np.argsort(abs(b.states[:, model.drive_index] - reference))
            if (
                abs(np.sin(target)) <= 32 * np.finfo(float).eps
                and model.configuration_solutions(target)["families"]
            ):
                # The legacy sampled map can contain an arbitrary continuum
                # value at this exact drive. Approach from regular rows instead.
                ids = np.array(
                    [
                        i
                        for i in ids
                        if abs(b.states[i, model.drive_index] - target) > 1e-8
                    ]
                )
            ids = ids[:2]
            if len(ids) < 2:
                continue
            q, z = b.states[ids]
            dt = z[model.drive_index] - q[model.drive_index]
            if abs(dt) < 1e-13:
                continue
            try:
                state, tangent, _ = model.continue_configuration(q, drive, (z - q) / dt)
            except PathError as error:
                failures.append(error)
                continue
            state[model.drive_index] = drive
            reached.append((state, tangent, b))
        if len(reached) != 1:
            if not reached and failures:
                raise failures[0]
            raise PathError(
                "AMBIGUOUS" if reached else "DRIVE_EXTREMUM",
                "branch at requested input",
            )
        return reached[0]

    def _initial_label(self, cell, q, candidate):
        tag = self._mode_name(cell, candidate.mode_id)
        if tag:
            return tag, tag
        model = cell["solver"]
        matches = []
        for bid, b in self._ensure_atlas(cell).items():
            if b.family is not None:
                continue
            try:
                state, _, _ = self._branch_state(
                    cell, bid, float(wrap_angle(q[model.drive_index]))
                )
            except PathError:
                continue
            if np.linalg.norm(wrap_angle(state - q)) < 2e-6:
                matches.append(b)
        if len(matches) != 1:
            raise PathError(
                "AMBIGUOUS" if matches else "NUMERICAL_FAILURE",
                "cannot label the initial path uniquely",
            )
        b = matches[0]
        return self._mode_name(cell, b.mode_id), b.id

    @staticmethod
    def _oracle(cell, drive):
        if cell["A"] is None:
            return None
        value = cell["A"] * np.cos(drive) + cell["B"]
        tolerance = 64 * np.finfo(float).eps * (1 + abs(cell["A"]) + abs(cell["B"]))
        if abs(value) > 1 + tolerance:
            raise PathError("DRIVE_EXTREMUM", "opposite-crease cosine outside [-1,1]")
        if abs(value) > 1 + 1e-7:
            raise PathError(
                "NUMERICAL_FAILURE",
                "ill-conditioned cosine coefficients do not resolve the boundary",
            )
        return float(np.clip(value, -1, 1))

    def _seed(self, cell, drive, free_angle=None):
        model = cell["solver"]
        packet = model.configuration_solutions(drive)
        oracle = self._oracle(cell, drive)
        if cell["initial_state"] is not None:
            anchor = self._solver_order(cell, cell["initial_state"])
            if closure_error(model.original_alphas, anchor) > 2e-8:
                raise PathError("NUMERICAL_FAILURE", "initial rotation closure")
            anchor_drive = float(anchor[model.drive_index])
            if cell["branch_id"] or cell["mode_id"]:
                state, tangent, b = self._branch_state(
                    cell, cell["branch_id"] or cell["mode_id"], anchor_drive
                )
                if np.linalg.norm(wrap_angle(state - anchor)) > 2e-6:
                    raise PathError(
                        "MODE_UNAVAILABLE",
                        "initial_state does not belong to the named path",
                    )
                q, tangent, _ = model.continue_configuration(anchor, drive, tangent)
                return q, tangent, self._mode_name(cell, b.mode_id), b.id, None
            origin = model.configuration_solutions(anchor_drive)
            if origin["families"]:
                if abs(anchor_drive - drive) > 1e-12:
                    raise PathError(
                        "UNDERDETERMINED",
                        "a family point alone cannot select an outgoing regular path",
                    )
                for f in origin["families"]:
                    c = f["evaluate"](float(anchor[f["free_crease"] - 1]))
                    if (
                        c is not None
                        and np.linalg.norm(wrap_angle(c.rhos - anchor)) < 2e-6
                    ):
                        return (
                            anchor,
                            None,
                            f["mode_id"],
                            f["mode_id"],
                            cell["fold_indices"].index(f["free_crease"] - 1),
                        )
                raise PathError(
                    "NUMERICAL_FAILURE", "initial family state does not close"
                )
            matching = [
                c
                for c in origin["candidates"]
                if np.linalg.norm(wrap_angle(c.rhos - anchor)) < 2e-6
            ]
            if len(matching) != 1 or matching[0].status == "KINEMATIC_SINGULARITY":
                raise PathError(
                    "AMBIGUOUS",
                    "initial singular/coincident state needs a mode or branch",
                )
            mode, bid = self._initial_label(cell, anchor, matching[0])
            q, tangent, _ = model.continue_configuration(anchor, drive)
            return q, tangent, mode, bid, None
        if cell["branch_id"] and not packet["families"]:
            q, tangent, b = self._branch_state(cell, cell["branch_id"], drive)
            return q, tangent, self._mode_name(cell, b.mode_id), b.id, None
        if packet["families"]:
            named = cell["branch_id"] or ""
            if (
                named.endswith("_minus_pi") and packet["families"][0]["fixed_drive"] > 0
            ) or (
                named.endswith("_plus_pi") and packet["families"][0]["fixed_drive"] < 0
            ):
                raise PathError(
                    "MODE_UNAVAILABLE",
                    "the named family uses the opposite signed drive boundary",
                )
            free = cell["free_angle"] if free_angle is None else free_angle
            if free is None:
                if cell["branch_id"]:
                    q, tangent, b = self._branch_state(cell, cell["branch_id"], drive)
                    return q, tangent, self._mode_name(cell, b.mode_id), b.id, None
                raise PathError(
                    "UNDERDETERMINED",
                    "fixed-drive continuum needs free_angle, initial_state or an incoming path",
                )
            families = [
                f
                for f in packet["families"]
                if not cell["branch_id"] or cell["branch_id"].startswith(f["mode_id"])
            ]
            if len(families) != 1:
                raise PathError("AMBIGUOUS", "choose one fixed-drive family")
            f = families[0]
            c = f["evaluate"](float(free))
            if c is None:
                raise PathError("NUMERICAL_FAILURE", "family recovery")
            q = c.rhos.copy()
            q[model.drive_index] = drive
            q[f["free_crease"] - 1] = float(free)
            return (
                q,
                None,
                f["mode_id"],
                f["mode_id"],
                cell["fold_indices"].index(f["free_crease"] - 1),
            )
        candidates = list(packet["candidates"])
        if not candidates:
            reason = (
                "DRIVE_EXTREMUM"
                if packet["status"] == "NO_SOLUTION"
                else "NUMERICAL_FAILURE"
            )
            raise PathError(reason, packet["status"])
        if oracle is not None:
            for c in candidates:
                if abs(np.cos(c.rhos[cell["output_solver_idx"]]) - oracle) > 2e-7:
                    raise PathError(
                        "NUMERICAL_FAILURE", "opposite-crease equation disagreement"
                    )
        if cell["mode_id"]:
            hint = cell["mode_id"]
            filtered = [
                c
                for c in candidates
                if c.mode_id == hint or self._mode_name(cell, c.mode_id) == hint
            ]
            if not filtered:
                q, tangent, b = self._branch_state(cell, hint, drive)
                return q, tangent, self._mode_name(cell, b.mode_id), b.id, None
        else:
            crease = 2 if cell["sigma_role"] == "opposite" else cell["iout"]
            filtered = []
            for c in candidates:
                folds = wrap_angle(self._canonical(cell, c.rhos))
                if (
                    abs(folds[0]) < 1e-9
                    or abs(folds[crease]) < 1e-9
                    or np.sign(folds[0]) * np.sign(folds[crease]) == cell["sigma"]
                ):
                    filtered.append(c)
        # Only exact algebraic duplicates with the SAME mode may collapse.
        unique = []
        for c in filtered:
            if not any(
                c.mode_id == z.mode_id
                and np.linalg.norm(wrap_angle(c.rhos - z.rhos)) < 1e-10
                for z in unique
            ):
                unique.append(c)
        if len(unique) != 1:
            raise PathError(
                "AMBIGUOUS" if unique else "MODE_UNAVAILABLE",
                "sigma/mode hint does not select one path",
            )
        candidate = unique[0]
        q = candidate.rhos.copy()
        q[model.drive_index] = drive
        mode, bid = self._initial_label(cell, q, candidate)
        return q, None, mode, bid, None

    def _check_contact(self, cell, q):
        if np.any(cell["alphas"] >= np.pi):
            raise PathError(
                "UNSUPPORTED_CONTACT_GEOMETRY",
                "reflex-sector panels are outside the local triangle contact scope",
            )
        report = cell["solver"].contact_report(q)
        if not report["angle_bounds"]:
            raise PathError("CONTACT_LIMIT", "continuous fold bound")
        if report["penetrations"]:
            raise PathError("CONTACT_LIMIT", "panel penetration")

    def available_configurations(self, input_angle, cell_index=0):
        """All candidates and independent family parameters in c0..c3 radians."""
        if not 0 <= cell_index < self.N:
            raise IndexError("invalid cell index")
        cell = self.cells[cell_index]
        packet = cell["solver"].configuration_solutions(input_angle)
        rows = []
        for c in packet["candidates"]:
            try:
                mode, bid = self._initial_label(cell, c.rhos, c)
            except PathError:
                mode, bid = self._mode_name(cell, c.mode_id), None
            rows.append(
                dict(
                    folds=self._canonical(cell, c.rhos),
                    mode_id=mode,
                    branch_id=bid,
                    residual=float(c.residual),
                    contact=(
                        None
                        if np.any(cell["alphas"] >= np.pi)
                        else cell["solver"].contact_report(c.rhos)["admissible"]
                    ),
                    contact_status=(
                        "UNSUPPORTED_CONTACT_GEOMETRY"
                        if np.any(cell["alphas"] >= np.pi)
                        else "CHECKED"
                    ),
                )
            )
        families = [
            dict(
                mode_id=f["mode_id"],
                fixed_drive=f["fixed_drive"],
                free_crease=cell["fold_indices"].index(f["free_crease"] - 1),
                valid_intervals=tuple(f["valid_intervals"]),
            )
            for f in packet["families"]
        ]
        return dict(status=packet["status"], candidates=rows, families=families)

    def _make_state(self, index, cell, q, tangent, mode, bid, free_crease=None):
        model = cell["solver"]
        folds = self._canonical(cell, q)
        residual = float(closure_error(model.original_alphas, q))
        if not np.isfinite(q).all() or residual > 2e-8:
            raise PathError("NUMERICAL_FAILURE", "rotation closure")
        if self.contact:
            self._check_contact(cell, q)
        target = float(wrap_angle(folds[0]))
        winding = round((folds[0] - target) / (2 * np.pi))
        interval = next(
            (
                (lo + winding * 2 * np.pi, hi + winding * 2 * np.pi)
                for lo, hi in model.kinematic_intervals()
                if lo - 1e-8 <= target <= hi + 1e-8
            ),
            None,
        )
        return UnitState(
            index,
            index % self.N,
            q.copy(),
            folds,
            mode,
            bid,
            None if tangent is None else tangent.copy(),
            residual,
            interval,
            free_crease,
            output_crease=cell["iout"],
        )

    def _evaluate(self, initial_rho0, previous=None, free_angles=None):
        drive = float(initial_rho0)
        if not np.isfinite(drive):
            raise ValueError("finite input angle required")
        result = StripResult(
            drive,
            self.total_units,
            contact_scope="unit_local" if self.contact else "off",
        )
        free_angles = {} if free_angles is None else dict(free_angles)
        if any(
            not isinstance(k, (int, np.integer))
            or isinstance(k, bool)
            or not 0 <= k < self.total_units
            or not np.isfinite(float(v))
            for k, v in free_angles.items()
        ):
            raise ValueError(
                "free_angles maps calculated unit indices to finite radians"
            )
        incoming = drive
        for index in range(self.total_units):
            cell = self.cells[index % self.N]
            old = (
                previous[index]
                if previous is not None and index < len(previous)
                else None
            )
            mode = old.mode_id if old is not None else cell["mode_id"]
            bid = old.branch_id if old is not None else cell["branch_id"]
            try:
                if self.contact and abs(incoming) > np.pi + 1e-10:
                    raise PathError("DOMAIN_END", "contact input domain")
                if old is None:
                    q, tangent, mode, bid, free = self._seed(
                        cell, incoming, free_angles.get(index)
                    )
                else:
                    q, tangent, mode, bid, free = self._follow(
                        cell, incoming, old, free_angles.get(index)
                    )
                state = self._make_state(index, cell, q, tangent, mode, bid, free)
                if old is not None:
                    events = []
                    if np.any(
                        np.rint((state.folds - wrap_angle(state.folds)) / (2 * np.pi))
                        != np.rint((old.folds - wrap_angle(old.folds)) / (2 * np.pi))
                    ):
                        events.append("ANGLE_WRAP")
                    if np.any(
                        np.sign(wrap_angle(state.folds))
                        != np.sign(wrap_angle(old.folds))
                    ):
                        events.append("MV_CHANGE")
                    state.events = tuple(events)
                _, raw = cell["solver"].get_3d_geometry(q)
                local = self._apply_lengths_and_scale(
                    raw, index, cell["lengths"], cell["CLV"]
                )
                prev_cell = self.cells[(index - 1) % self.N] if index else None
                gv = self._place_unit_global(
                    local,
                    result.vertices[-1] if index else None,
                    cell["CLV"],
                    prev_cell["iout"] if index else None,
                    prev_cell["CLV"] if index else None,
                )
                if index:
                    result.joints.append(
                        self._joint_report(
                            local, gv, result.vertices[-1], cell, prev_cell
                        )
                    )
                vertices = [gv[cell["CLV"][k]] for k in range(4)]
                result.states.append(state)
                result.vertices.append(gv)
                result.faces.append(
                    [[gv[0], vertices[k], vertices[(k + 1) % 4]] for k in range(4)]
                )
                incoming = float(state.folds[cell["iout"]])
            except PathError as error:
                result.termination = dict(
                    reason=error.reason,
                    detail=str(error),
                    unit_index=index,
                    cell_index=index % self.N,
                    input_angle=incoming,
                    mode_id=mode,
                    branch_id=bid,
                )
                break
        return result

    def _commit(self, result):
        self.last_result = result
        self.unit_states = result.states
        self.rhos = np.full((self.total_units, 4), np.nan)
        for state in result.states:
            self.rhos[state.unit_index] = state.solver_rhos
        self.global_vertices = result.vertices
        self.global_faces = result.faces
        self.unit_iouts = [self.cells[s.cell_index]["iout"] for s in result.states]

    def compute_strip_kinematics(self, initial_rho0, *, track=False, free_angles=None):
        """Deterministic snapshot, or explicit temporal update; return prefix count."""
        if track:
            return self.update(initial_rho0, free_angles=free_angles).valid_units
        result = self._evaluate(initial_rho0, free_angles=free_angles)
        self.last_attempt = result
        self.last_error = None if result.complete else result.termination["detail"]
        self._commit(result)
        return result.valid_units

    def _follow(self, cell, drive, old, free_angle=None):
        model = cell["solver"]
        if old.family_free_crease is not None:
            if abs(drive - old.input_angle) > 1e-10:
                raise PathError(
                    "UNDERDETERMINED",
                    "leaving a fixed-drive family needs an explicit regular path",
                )
            free = (
                old.folds[old.family_free_crease]
                if free_angle is None
                else float(free_angle)
            )
            packet = model.configuration_solutions(drive)
            families = [f for f in packet["families"] if f["mode_id"] == old.mode_id]
            if len(families) != 1:
                raise PathError("AMBIGUOUS", "fixed-drive family changed")
            family = families[0]
            q = old.solver_rhos.copy()
            start = float(old.folds[old.family_free_crease])
            idx = family["free_crease"] - 1
            for goal in np.linspace(
                start, free, max(1, int(np.ceil(abs(free - start) / 0.035))) + 1
            )[1:]:
                candidate = family["evaluate"](float(goal))
                if candidate is None:
                    raise PathError("NUMERICAL_FAILURE", "family continuation")
                q = lift_angles(candidate.rhos, q)
                q[model.drive_index] = drive
                q[idx] = goal
                if self.contact:
                    self._check_contact(cell, q)
            return q, None, old.mode_id, old.branch_id, old.family_free_crease
        tangent = old.tangent
        if tangent is None and abs(drive - old.input_angle) > 1e-12:
            # Establish a local tangent before taking a larger correction.
            delta = np.copysign(1e-5, drive - old.input_angle)
            near, _, _ = model.continue_configuration(
                old.solver_rhos, old.input_angle + delta
            )
            tangent = (near - old.solver_rhos) / delta
        q, tangent, _ = model.continue_configuration(
            old.solver_rhos, drive, tangent, contact=self.contact
        )
        return q, tangent, old.mode_id, old.branch_id, None

    def _trace(self, target, *, free_angles=None):
        target = float(target)
        if not np.isfinite(target):
            raise ValueError("finite input angle required")
        if self.last_result is None or not self.last_result.complete:
            return self._evaluate(target, free_angles=free_angles)
        start = self.last_result
        count = 0

        def step(previous, goal, depth=0):
            nonlocal count
            count += 1
            result = self._evaluate(goal, previous.states, free_angles)
            if result.complete:
                return result
            if (
                result.status in ("AMBIGUOUS", "NUMERICAL_FAILURE")
                and depth < 10
                and abs(goal - previous.input_angle) > 1e-7
            ):
                mid = step(previous, (previous.input_angle + goal) / 2, depth + 1)
                if not mid.complete:
                    return mid
                return step(mid, goal, depth + 1)
            return result

        result = start
        for goal in np.linspace(
            start.input_angle,
            target,
            max(1, int(np.ceil(abs(target - start.input_angle) / 0.035))) + 1,
        )[1:]:
            result = step(result, float(goal))
            if not result.complete:
                break
        # Even an unchanged root input can move an explicitly specified family.
        result.requested_input = target
        result.continuation_steps = count
        return result

    def update(self, initial_rho0, *, free_angles=None):
        """Atomic temporal update. Failure retains the last accepted full strip."""
        result = self._trace(initial_rho0, free_angles=free_angles)
        self.last_attempt = result
        self.last_error = None if result.complete else result.termination["detail"]
        if result.complete:
            self._commit(result)
        return result

    def reset(self):
        """Discard path history explicitly; the next query uses initialization hints."""
        self.last_result = self.last_attempt = None
        self.last_error = None
        self.unit_states = []
        self.global_vertices = []
        self.global_faces = []
        self.unit_iouts = []
        self.rhos = np.full((self.total_units, 4), np.nan)

    def set_contact(self, enabled):
        """Change local contact policy only if the current whole strip passes."""
        enabled = bool(enabled)
        if enabled:
            for state in self.unit_states:
                try:
                    self._check_contact(self.cells[state.cell_index], state.solver_rhos)
                except PathError as error:
                    self.last_error = f"Unit {state.unit_index}: {error}"
                    return False
        self.contact = enabled
        self._curve_cache.clear()
        if self.last_result is not None:
            self.last_result.contact_scope = "unit_local" if enabled else "off"
        self.last_error = None
        return True

    def preview(self, initial_rho0, *, free_angles=None, track=None):
        """Query without changing accepted strip folds, meshes or path history."""
        if track is None:
            track = self.last_result is not None and self.last_result.complete
        return (
            self._trace(initial_rho0, free_angles=free_angles)
            if track
            else self._evaluate(initial_rho0, free_angles=free_angles)
        )

    def get_fold_angles(self, t, *, wrapped=False, order="tess"):
        if not isinstance(t, (int, np.integer)) or t < 0 or t >= len(self.unit_states):
            raise IndexError("unit was not successfully calculated")
        if order not in ("tess", "solver"):
            raise ValueError("order must be tess or solver")
        state = self.unit_states[t]
        q = state.folds if order == "tess" else state.solver_rhos
        return wrap_angle(q) if wrapped else q.copy()

    def _get_rho_for_crease(self, t, k):
        if k not in (0, 1, 2, 3):
            raise ValueError("crease must be c0..c3")
        return float(self.get_fold_angles(t)[k])

    def get_mv_pattern(self, t, tol=1e-4):
        return "".join(
            "V" if q > tol else "M" if q < -tol else "F"
            for q in self.get_fold_angles(t, wrapped=True)
        )

    def validate_with_algebraic_solver(self, unit_index=0, verbose=True):
        q = self.get_fold_angles(unit_index)
        cell = self.cells[unit_index % self.N]
        t0, t1, t2, t3 = cell["alphas"]
        lhs = np.cos(t1) * np.cos(t2) - np.sin(t1) * np.sin(t2) * np.cos(q[2])
        rhs = np.cos(t3) * np.cos(t0) - np.sin(t3) * np.sin(t0) * np.cos(q[0])
        ok = bool(
            abs(lhs - rhs) < 2e-8
            and closure_error(cell["solver"].original_alphas, self.rhos[unit_index])
            < 2e-8
        )
        if verbose:
            print(f"Unit {unit_index}: spherical identity and rotation closure: {ok}")
        return ok

    def _display_angles(self, values):
        values = np.asarray(values, float)
        q = wrap_angle(values)
        if self.contact:
            q = np.where((values > 0) & (abs(q + np.pi) < 2e-14), np.pi, q)
        return q

    def _pieces(self, rows):
        rows = np.asarray(rows, float)
        if not len(rows):
            return []
        q = self._display_angles(rows)
        cuts = np.flatnonzero(np.any(abs(np.diff(q, axis=0)) > np.pi, axis=1)) + 1
        return [part for part in np.split(q, cuts) if len(part)]

    def cobweb_data(self, *, resolution=None):
        """All distinct model paths/families as polylines, never an interpolant.

        Background paths are not sigma-filtered: sigma is an initialization
        hint, not a global branch label. Orange steps use the shared evaluator.
        resolution caps rendered rows; it never changes the path-selection atlas.
        """
        if resolution is not None and (
            not isinstance(resolution, (int, np.integer)) or resolution < 2
        ):
            raise ValueError("resolution must be an integer >=2")
        data = []
        for index, cell in enumerate(self.cells):
            key = (index, self.contact, resolution)
            if key not in self._curve_cache:
                curves = []
                for bid, b in self._ensure_atlas(cell).items():
                    sources = b.physical_segments if self.contact else [b.states]
                    for source in sources:
                        canonical = np.asarray(
                            [self._canonical(cell, q) for q in source]
                        )
                        for piece in self._pieces(canonical):
                            if resolution and len(piece) > resolution:
                                piece = piece[
                                    np.unique(
                                        np.linspace(
                                            0, len(piece) - 1, resolution, dtype=int
                                        )
                                    )
                                ]
                            curves.append(
                                dict(
                                    cell_index=index,
                                    branch_id=bid,
                                    mode_id=self._mode_name(cell, b.mode_id),
                                    family=b.family is not None,
                                    free_crease=(
                                        None
                                        if b.family is None
                                        else cell["fold_indices"].index(
                                            b.family["free_crease"] - 1
                                        )
                                    ),
                                    states=piece,
                                    xy=piece[:, [0, cell["iout"]]],
                                )
                            )
                self._curve_cache[key] = curves
            data.extend(deepcopy(self._curve_cache[key]))
        return data

    def plot_cobweb(self, rho0_deg, ax=None, color="orange", *, resolution=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 8))
        data = self.cobweb_data(resolution=resolution)
        keys = sorted({(p["cell_index"], p["mode_id"] or p["branch_id"]) for p in data})
        palette = ("#4477aa", "#228833", "#cc3311", "#aa3377", "#66ccee", "#332288")
        for part in data:
            xy = part["xy"]
            key = (part["cell_index"], part["mode_id"] or part["branch_id"])
            ax.plot(
                xy[:, 0],
                xy[:, 1],
                lw=1.0,
                alpha=0.45,
                color=palette[keys.index(key) % len(palette)],
                marker="o" if len(xy) == 1 else None,
                label=f"cell{part['cell_index']} {part['mode_id']} [{part['branch_id']}]",
            )
        result = self.preview(np.deg2rad(rho0_deg))
        for state in result.states:
            cell = self.cells[state.cell_index]
            x, y = self._display_angles([state.input_angle, state.folds[cell["iout"]]])
            ax.plot([x, x, y], [x, y, y], color=color, lw=1.8)
        if result.termination:
            t = result.termination
            x = float(self._display_angles(t["input_angle"]))
            ax.annotate(t["reason"], (x, x), color="firebrick")
        ax.plot([-np.pi, np.pi], [-np.pi, np.pi], "k--", alpha=0.4)
        ax.set(
            xlim=(-np.pi, np.pi),
            ylim=(-np.pi, np.pi),
            xlabel="Input fold [rad]",
            ylabel="Output fold [rad]",
        )
        ax.grid(alpha=0.3)
        ax.set_aspect("equal")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                dict(zip(labels, handles)).values(),
                dict(zip(labels, handles)).keys(),
                fontsize=6,
            )
        ax.origami_result = result
        ax.origami_curves = data
        return ax

    def debug_kinematics(self, rho0_deg, stop_on_flip=True):
        """Read-only trace from the same selector. Scalar FLIP is not a stop rule."""
        result = self.preview(np.deg2rad(rho0_deg))
        print("unit cell input_deg output_deg mode branch closure")
        sequence = []
        for state in result.states:
            output = float(state.folds[self.cells[state.cell_index]["iout"]])
            values = (
                state.unit_index,
                float(np.rad2deg(state.input_angle)),
                float(np.rad2deg(output)),
            )
            sequence.append(values)
            print(
                state.unit_index,
                state.cell_index,
                values[1],
                values[2],
                state.mode_id,
                state.branch_id,
                state.residual,
            )
        print(result.status, result.termination or "")
        return sequence

    def _print_init_summary(self):
        if not self.verbose:
            return
        print(
            f"OrigamiTesellator1D: {self.N} cells x {self.num_periods} = {self.total_units} units"
        )
        print(
            f"Local contact={self.contact}; joint policy={self.joint_policy}; global collision unchecked"
        )
        for index, cell in enumerate(self.cells):
            g = cell["geometry"]
            print(
                f"cell {index}: {g['case']}, sum={sum(cell['alphas_deg']):g} deg, "
                f"iout=c{cell['iout']}, sigma={cell['sigma']:+d} ({cell['sigma_role']}), "
                f"flat-foldable={cell['is_flat_foldable']}, collinear={g['collinear']}"
            )
            if cell["A"] is not None:
                print(f"  cos(rho2)={cell['A']:.9g}*cos(rho0)+({cell['B']:.9g})")

    def _rodrigues(self, axis, angle):
        norm = np.linalg.norm(axis)
        if norm < 1e-9:
            return np.eye(3)
        axis = axis / norm
        K = np.array(
            [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
        )
        return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    def _get_rot_from_vecs(self, u, v):
        u = u / np.linalg.norm(u)
        v = v / np.linalg.norm(v)
        c = np.cross(u, v)
        d = np.dot(u, v)
        nc = np.linalg.norm(c)

        if nc < 1e-9:
            if d > 0:
                return np.eye(3)  # parallel & same dir: identity
            else:
                # ── parallel & opposite dir → 180° proper rotation w.r.t. normal axis──
                # choose axis perpendicular to u
                perp = np.array([1.0, 0.0, 0.0])
                if (
                    abs(np.dot(u, perp)) > 0.9
                ):  # if u closer to x-axis, use y-axis instead
                    perp = np.array([0.0, 1.0, 0.0])
                axis = np.cross(u, perp)
                axis = axis / np.linalg.norm(axis)
                return self._rodrigues(axis, np.pi)  # det=+1 guarantee

        return self._rodrigues(c / nc, np.arccos(np.clip(d, -1.0, 1.0)))

    def _apply_lengths_and_scale(self, local_verts, unit_index, cell_lengths, CLV):
        lv = local_verts.copy()
        O = lv[0]
        for k in (1, 2, 3):  # tess creases e1, e2, e3
            vi = CLV[k]
            vec = lv[vi] - O
            n = np.linalg.norm(vec)
            if n > 1e-9:
                lv[vi] = O + (vec / n) * cell_lengths[k - 1]
        try:
            with np.errstate(over="raise", invalid="raise", under="ignore"):
                result = lv * (self.scale_factor**unit_index)
            if not np.isfinite(result).all() or not np.any(result[1:] - result[0]):
                raise ValueError("scale outside numerical range")
            return result
        except (OverflowError, FloatingPointError, ValueError) as error:
            raise PathError(
                "NUMERICAL_FAILURE", "scaled geometry: " + str(error)
            ) from error

    def _place_unit_global(self, local_verts, prev_gv, curr_CLV, prev_iout, prev_CLV):
        """Using rotation and translation, patch local unit to global tesellated strips

        Eq. (A4) in Imada(2026):
            o_{n+1}   = o_n + l^{iout} · c^{iout}_n     [position]
            c^0_{n+1} = -c^{iout}_n                     [input crease reversal]
            n^0_{n+1} = n^{iout-1}_n                    [face normal continuity]

        Parameters
        ----------
        curr_CLV  : current  cell's CLV  (for local_verts idx)
        prev_iout : previous cell's output crease index
        prev_CLV  : previous cell's CLV  (for prev_gv idx)

        """
        if prev_gv is None:
            return local_verts

        rotation, origin = self._joint_transform(
            local_verts, prev_gv, curr_CLV, prev_iout, prev_CLV
        )
        return (local_verts - local_verts[0]) @ rotation.T + origin

    @staticmethod
    def _joint_transform(local, previous, current_map, output, previous_map):
        c0 = local[current_map[0]] - local[0]
        c1 = local[current_map[1]] - local[0]
        ck = previous[previous_map[output]] - previous[0]
        before = previous[previous_map[(output - 1) % 4]] - previous[0]
        try:
            source = crease_frame(-c0, np.cross(c0, c1))
            target = crease_frame(ck, np.cross(before, ck))
        except ValueError as error:
            raise PathError(
                "NUMERICAL_FAILURE", "joint frame: " + str(error)
            ) from error
        return target @ source.T, previous[previous_map[output]]

    def _joint_report(self, local, gv, previous, cell, previous_cell):
        cm, pm = cell["CLV"], previous_cell["CLV"]
        output = previous_cell["iout"]
        rotation, origin = self._joint_transform(local, previous, cm, output, pm)
        v = gv[cm[0]] - gv[0]
        old = previous[pm[output]] - previous[0]

        def unit(x):
            return x / np.linalg.norm(x)

        normal = np.cross(v, gv[cm[1]] - gv[0])
        oldnormal = np.cross(previous[pm[(output - 1) % 4]] - previous[0], old)
        edge_error = float(np.linalg.norm(gv[cm[0]] - previous[0]))
        report = dict(
            origin_error=float(np.linalg.norm(gv[0] - origin)),
            axis_error=float(np.linalg.norm(unit(v) + unit(old))),
            normal_error=float(np.linalg.norm(unit(normal) - unit(oldnormal))),
            rotation=rotation,
            orthogonality_error=float(
                np.linalg.norm(rotation.T @ rotation - np.eye(3))
            ),
            determinant=float(np.linalg.det(rotation)),
            edge_error=edge_error,
            edge_shared=bool(edge_error <= 1e-8 * (1 + np.linalg.norm(old))),
            policy=self.joint_policy,
        )
        if not all(
            np.isfinite(report[k])
            for k in (
                "origin_error",
                "axis_error",
                "normal_error",
                "orthogonality_error",
                "determinant",
                "edge_error",
            )
        ):
            raise PathError("NUMERICAL_FAILURE", "nonfinite joint geometry")
        if (
            report["axis_error"] > 1e-8
            or report["normal_error"] > 1e-8
            or report["orthogonality_error"] > 1e-8
            or abs(report["determinant"] - 1) > 1e-8
            or report["origin_error"] > 1e-8 * (1 + np.linalg.norm(origin))
        ):
            raise PathError("NUMERICAL_FAILURE", "joint geometry residual")
        if self.joint_policy == "shared_edge" and not report["edge_shared"]:
            raise PathError(
                "JOINT_INCOMPATIBLE",
                f"finite crease endpoints differ by {edge_error:g}",
            )
        return report

    _CELL_PALETTES = [
        ["#CCCCCC", "#A8C4E0", "#A8D4A8", "#E0B8B0"],
        ["#F0E0A0", "#D8C878", "#E8D8A8", "#C8B880"],
        ["#D0C0F0", "#C0A8E0", "#D8B8F0", "#B0A0D8"],
        ["#A8E4E4", "#88CCCC", "#A0DCDC", "#78C4C4"],
    ]

    def _build_crease_lines(self):
        """
        From global_vertices, backbone / side / cross line coordinate calculation.

        Returns
        -------
        backbone : (M,3)       vertex origins + last output crease end-point
        e1_pts   : (M-1, 3)   tess-e1 end-point of each vertex
        e3_pts   : (M-1, 3)   tess-e3 end-point of each vertex
        cross    : (cx,cy,cz)  4-spoke crease line (NaN 분리)
        """
        if not self.global_vertices:
            return None, None, None, None

        # Backbone: every origin + last output end-point
        backbone = [gv[0] for gv in self.global_vertices]
        if self.unit_iouts:
            last_t = len(self.global_vertices) - 1
            last_CLV = self.cells[last_t % self.N]["CLV"]
            backbone.append(self.global_vertices[-1][last_CLV[self.unit_iouts[-1]]])
        backbone = np.array(backbone)

        # Side lines: tess-e1, tess-e3 end-point
        e1_pts, e3_pts = [], []
        for t, gv in enumerate(self.global_vertices):
            CLV = self.cells[t % self.N]["CLV"]
            e1_pts.append(gv[CLV[1]])
            e3_pts.append(gv[CLV[3]])
        e1_pts = np.array(e1_pts)
        e3_pts = np.array(e3_pts)

        # Cross spokes: origin → 4 crease end-point
        cx, cy, cz = [], [], []
        for t, gv in enumerate(self.global_vertices):
            CLV = self.cells[t % self.N]["CLV"]
            O = gv[0]
            for k in range(4):
                ep = gv[CLV[k]]
                cx += [O[0], ep[0], np.nan]  # to make each spoke as independent line
                cy += [O[1], ep[1], np.nan]
                cz += [O[2], ep[2], np.nan]

        return backbone, e1_pts, e3_pts, (cx, cy, cz)

    def _render_faces_and_lines(
        self, ax, valid, poly_col, line_bb, line_e1, line_e3, line_cr
    ):
        """Renew Poly3DCollection + Line3D object of current result."""

        # faces
        if self.global_faces and valid > 0:
            flat = [f for unit in self.global_faces for f in unit]
            colors = []
            for t in range(valid):
                pal = self._CELL_PALETTES[t % self.N % len(self._CELL_PALETTES)]
                colors.extend(pal)
            poly_col.set_verts(flat)
            poly_col.set_facecolors(colors)
        else:
            poly_col.set_verts([])

        # crease lines
        backbone, e1_pts, e3_pts, cross = self._build_crease_lines()

        def _set(line, pts):
            line.set_data(pts[:, 0], pts[:, 1])
            line.set_3d_properties(pts[:, 2])

        if backbone is not None:
            cx, cy, cz = cross
            _set(line_bb, backbone)
            _set(line_e1, e1_pts)
            _set(line_e3, e3_pts)
            line_cr.set_data(cx, cy)
            line_cr.set_3d_properties(cz)
        else:
            for line in (line_bb, line_e1, line_e3, line_cr):
                line.set_data([], [])
                line.set_3d_properties([])

        # axis range
        rng = max(valid * 1.6, 2.0)
        ax.set_xlim(-rng, rng)
        ax.set_ylim(-rng, rng)
        ax.set_zlim(-rng, rng)

    def plot_3d(self, rho0_deg, figsize=(11, 8), *, show=True):
        """
        Plot static strip 3D shape for given input angle.

        Parameters
        ----------
        rho0_deg : float   input fold angle [deg]
        figsize  : tuple
        """
        valid = self.compute_strip_kinematics(np.deg2rad(rho0_deg))

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        if valid == 0:
            ax.set_title(f"{self.last_result.status} at rho0 = {rho0_deg} deg")
            if show:
                plt.show()
            return fig

        poly = Poly3DCollection([], edgecolors="#555555", linewidths=0.5, alpha=0.85)
        ax.add_collection3d(poly)
        (line_bb,) = ax.plot([], [], [], color="crimson", lw=2.5, label="Backbone")
        (line_e1,) = ax.plot([], [], [], color="navy", lw=1.5, ls="--", label="e1 tips")
        (line_e3,) = ax.plot(
            [], [], [], color="darkgreen", lw=1.5, ls="--", label="e3 tips"
        )
        (line_cr,) = ax.plot([], [], [], color="black", lw=0.6, alpha=0.35)

        self._render_faces_and_lines(
            ax, valid, poly, line_bb, line_e1, line_e3, line_cr
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend(fontsize=8, loc="upper left")

        cell_desc = "  |  ".join(
            f"cell{i} iout={c['iout']} sigma={c['sigma']:+d}"
            for i, c in enumerate(self.cells)
        )
        ax.set_title(
            f"rho0 = {rho0_deg} deg  |  {valid}/{self.total_units} units\n{cell_desc}",
            fontsize=9,
        )
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    def setup_interactive_viewer(self, *, show=True, initial_rho0_deg=45.0):
        """Temporal strip viewer; rejected moves restore the accepted full pose."""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")
        plt.subplots_adjust(bottom=0.25)
        slider = Slider(
            fig.add_axes([0.18, 0.12, 0.58, 0.035]),
            "rho0 (deg)",
            -180,
            180,
            valinit=initial_rho0_deg,
        )
        free_slider = Slider(
            fig.add_axes([0.18, 0.18, 0.58, 0.035]),
            "Free fold (deg)",
            -180,
            180,
            valinit=0,
        )
        free_slider.set_active(False)
        check = CheckButtons(
            fig.add_axes([0.81, 0.08, 0.18, 0.12]), ["Local contact"], [self.contact]
        )
        status = fig.text(0.15, 0.025, "", fontsize=9)
        poly = Poly3DCollection([], edgecolors="#555", alpha=0.85)
        ax.add_collection3d(poly)
        (line_bb,) = ax.plot([], [], [], c="crimson", label="Backbone")
        (line_e1,) = ax.plot([], [], [], c="navy", ls="--", label="c1 tips")
        (line_e3,) = ax.plot([], [], [], c="darkgreen", ls="--", label="c3 tips")
        (line_cr,) = ax.plot([], [], [], c="gray", lw=0.6)
        ax.set(xlabel="X", ylabel="Y", zlabel="Z")
        ax.legend(fontsize=8)
        guard = {"active": False, "updates": 0}

        def silent(widget, value):
            eventson = widget.eventson
            widget.eventson = False
            try:
                widget.set_val(value)
            finally:
                widget.eventson = eventson

        def render():
            valid = len(self.unit_states)
            self._render_faces_and_lines(
                ax, valid, poly, line_bb, line_e1, line_e3, line_cr
            )
            if valid:
                silent(slider, float(np.rad2deg(self.unit_states[0].input_angle)))
            family = self.unit_states[0].family_free_crease if valid else None
            free_slider.ax.set_visible(family is not None)
            free_slider.set_active(family is not None)
            slider.set_active(family is None)
            if family is not None:
                free_slider.label.set_text(f"Free c{family} unit0 (deg)")
                silent(
                    free_slider, float(np.rad2deg(self.unit_states[0].folds[family]))
                )
            ax.set_title(f"{valid}/{self.total_units} units | contact={self.contact}")
            status.set_text(
                self.last_error
                or "Selected paths retained. Contact checks individual vertices."
            )
            status.set_color("firebrick" if self.last_error else "#333")
            fig.canvas.draw_idle()

        def move(_value=None):
            if guard["active"]:
                return
            guard["active"] = True
            try:
                guard["updates"] += 1
                free = {0: np.deg2rad(free_slider.val)} if free_slider.active else None
                self.update(np.deg2rad(slider.val), free_angles=free)
                render()
            finally:
                guard["active"] = False

        def contact(_label):
            if guard["active"]:
                return
            guard["active"] = True
            try:
                self.set_contact(bool(check.get_status()[0]))
                if bool(check.get_status()[0]) != self.contact:
                    eventson = check.eventson
                    check.eventson = False
                    try:
                        check.set_active(0)
                    finally:
                        check.eventson = eventson
                render()
            finally:
                guard["active"] = False

        slider.on_changed(move)
        free_slider.on_changed(move)
        check.on_clicked(contact)
        fig.origami_strip = self
        fig.origami_slider = slider
        fig.origami_free_slider = free_slider
        fig.origami_contact = check
        fig.origami_status = status
        fig.origami_guard = guard
        self._viewer = fig
        move()
        if show:
            plt.show()
        return fig

    def export_to_mat(self, filepath, rho0_deg, *, track=False, allow_partial=False):
        """Read-only snapshot export. c-order folds and explicit failure metadata."""
        from scipy.io import savemat

        result = self.preview(np.deg2rad(rho0_deg), track=track)
        if not result.complete and not allow_partial:
            raise PathError(result.status, result.termination["detail"])
        triangles = [tri for unit in result.faces for tri in unit]
        vertices = np.asarray(triangles, float).reshape(-1, 3)
        faces = np.arange(1, len(vertices) + 1, dtype=int).reshape(-1, 3)
        backbone = [v[0] for v in result.vertices]
        c1 = []
        c3 = []
        for state, gv in zip(result.states, result.vertices):
            cell = self.cells[state.cell_index]
            c1.append(gv[cell["CLV"][1]])
            c3.append(gv[cell["CLV"][3]])
        if result.states:
            last = result.states[-1]
            cell = self.cells[last.cell_index]
            backbone.append(result.vertices[-1][cell["CLV"][cell["iout"]]])
        data = dict(
            vertices=vertices,
            faces=faces,
            backbone=np.asarray(backbone).reshape(-1, 3),
            e1_pts=np.asarray(c1).reshape(-1, 3),
            e3_pts=np.asarray(c3).reshape(-1, 3),
            fold_angles=np.asarray([s.folds for s in result.states]).reshape(-1, 4),
            solver_rhos=np.asarray([s.solver_rhos for s in result.states]).reshape(
                -1, 4
            ),
            mode_ids=np.array([s.mode_id for s in result.states], dtype=object),
            branch_ids=np.array([s.branch_id for s in result.states], dtype=object),
            meta=dict(
                rho0_deg=float(rho0_deg),
                valid_units=result.valid_units,
                total_units=self.total_units,
                N_period=self.N,
                status=result.status,
                contact_scope=result.contact_scope,
                global_contact_checked=False,
                joint_policy=self.joint_policy,
            ),
        )
        savemat(filepath, data)
        if self.verbose:
            print(
                f"[SAVE] {filepath}: {result.valid_units}/{self.total_units} units, {result.status}"
            )
        return data


import scipy.io


def export_debug_placement_mat(tessellator, filepath, rho0_deg, t_flip=10):
    """Read-only export with the actual proper joint rotation and residuals."""
    from scipy.io import savemat

    result = tessellator.preview(np.deg2rad(rho0_deg))
    if not result.complete:
        raise PathError(result.status, result.termination["detail"])
    if not 1 <= t_flip < result.valid_units:
        raise ValueError("t_flip must identify a calculated joint after unit 0")
    data = tessellator.export_to_mat(
        filepath, rho0_deg, track=tessellator.last_result is not None
    )
    data["joint"] = result.joints[t_flip - 1]
    data["meta"].update(t_flip=t_flip, t_prev=t_flip - 1)
    savemat(filepath, data)
    print(
        f"[SAVE] {filepath}: joint {t_flip-1}->{t_flip}, residuals={result.joints[t_flip-1]}"
    )
    return data


if __name__ == "__main__":

    # Flat-foldable & helical case
    # cell_configs = [
    #     {"alphas": [95, 60, 85, 120], "sigma": 1, "iout": 2},
    # ]

    # tessellator = OrigamiTesellator1D(
    #     cell_configs=cell_configs,
    #     num_periods=8,
    #     lengths=(1.5, 1.0, 1.5),
    #     scale_factor=1.0,
    # )

    C_left = OrigamiTesellator1D(
        cell_configs=[
            {"alphas": [70, 70, 120, 110], "sigma": 1, "iout": 3},
        ],
        num_periods=2,
        lengths=(1.0, 1.0, 1.0),
    )

    flat_foldable_helix = OrigamiTesellator1D(
        cell_configs=[
            {"alphas": [70, 60, 110, 120], "sigma": 1, "iout": 2},
        ],
        num_periods=1,
        lengths=(0.8, 1.5, 0.8),
    )

    C_left.plot_cobweb(rho0_deg=-20)
    plt.title("1D Strip Tessellation Propagation")
    plt.show()

    seq = C_left.debug_kinematics(rho0_deg=-20)
    # C_left.plot_3d(rho0_deg=-30)
    # flat_foldable_helix.plot_3d(rho0_deg=20)
    # flat_foldable_helix.setup_interactive_viewer()
    # C_left.setup_interactive_viewer()
    # C_left.export_to_mat("strip_rho45.mat", rho0_deg=120)
    export_debug_placement_mat(C_left, "strip_debug.mat", rho0_deg=-20, t_flip=1)
