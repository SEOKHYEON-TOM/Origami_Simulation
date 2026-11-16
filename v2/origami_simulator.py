"""Degree-four origami candidates, connected actuation branches and sessions.

All public crease indices are original one-based indices; angles are radians
unless named *_deg. Contact checks angle bounds and transverse panel penetration
for the four zero-thickness, vertex-sharing triangular panels.
"""

from dataclasses import dataclass, field
import time
import numpy as np
from origami_kinematics import Kinematics, closure_error, rx, rz, rotation

TAU = 2 * np.pi


def wrap_angle(a):
    return (np.asarray(a) + np.pi) % TAU - np.pi


def lift_angles(q, ref):
    return np.asarray(ref) + wrap_angle(np.asarray(q) - ref)


def panel_penetrations(vertices, tolerance=1e-10):
    """Proper intersections of opposite radial panels; touching is allowed.

    Each triangle is (O,u,v), with sectors strictly between 0 and pi. The
    cofactors of [u,v,-w,-z] give its null vector. Four strictly same-sign
    coefficients give a point in both triangle interiors after normalization.
    Rank-two/coplanar overlap is boundary contact, not transverse penetration.
    Returned panel numbers match get_3d_geometry's one-based face order.
    """
    vertices = np.asarray(vertices, float)
    if vertices.shape != (5, 3) or not np.isfinite(vertices).all():
        raise ValueError("five finite 3D vertices required")
    rays = vertices[1:] - vertices[0]
    panels = rays[[[3, 0], [0, 1], [1, 2], [2, 3]]]
    hits = []
    for i, j in ((0, 2), (1, 3)):
        u, v = panels[i]
        w, z = panels[j]
        uv, wz = np.cross(u, v), np.cross(w, z)
        k = np.array([v @ wz, -u @ wz, -z @ uv, w @ uv])
        scale = np.linalg.norm(k)
        if scale <= tolerance:
            continue
        k /= scale
        if np.all(k > tolerance) or np.all(k < -tolerance):
            weights = abs(k) / np.sum(abs(k))
            point = vertices[0] + weights[0] * u + weights[1] * v
            hits.append({"panels": (i + 1, j + 1), "point": point})
    return hits


@dataclass
class Endpoint:
    angle: float
    reason: str
    detail: str = ""


@dataclass
class State:
    rhos: np.ndarray
    drive_unwrapped: float
    drive_wrapped: float
    winding_count: int
    branch_id: str
    residual: float
    events: tuple = ()
    mv_pattern: str = ""


@dataclass
class Branch:
    id: str
    mode_id: str
    states: np.ndarray
    seed_state: np.ndarray
    segments: list = field(default_factory=list)
    endpoints: list = field(default_factory=list)
    family: object = None
    physical_segments: list = field(default_factory=list)
    seed_checks: int = 0
    reachable_intervals: list = field(default_factory=list)
    admissible_domain: tuple | None = None
    seam_connections: dict = field(default_factory=dict)


class PathError(ValueError):
    def __init__(self, reason, detail=""):
        self.reason = reason
        super().__init__(reason + (": " + detail if detail else ""))


class OrigamiDegree4Simulator:
    def __init__(
        self,
        alphas_deg,
        contact=True,
        verbose=True,
        driving_crease=None,
        seed_count=10,
        max_seeds=50,
        wrap_margin=0.1,
        residual_tol=2e-8,
    ):
        if not 10 <= int(seed_count) <= int(max_seeds) <= 50:
            raise ValueError("Require 10 <= seed_count <= max_seeds <= 50")
        if not np.isfinite(residual_tol) or residual_tol <= 0:
            raise ValueError("residual_tol must be finite and positive")
        self._core = Kinematics(
            np.array(alphas_deg, float, copy=True),
            driving_crease=driving_crease,
            residual_tol=residual_tol,
        )
        self.original_alphas_deg = np.array(alphas_deg, float)
        self.original_alphas = np.deg2rad(self.original_alphas_deg)
        self.geometry = self._core.geometry
        self.alphas = self._core.internal_alphas
        for angles in (self.original_alphas_deg, self.original_alphas, self.alphas):
            angles.setflags(write=False)
        self._driving_crease = self._core.driving_crease
        self.shift_amount = self._core.shift_amount
        self.is_euclidean = self.geometry["case"] == "developable"
        self.geom_type = {
            "developable": "Euclidean (Flat)",
            "elliptic": "Elliptic (Cone-like)",
            "hyperbolic": "Hyperbolic (Saddle-like)",
        }[self.geometry["case"]]
        self.contact, self.verbose = bool(contact), bool(verbose)
        self.seed_count, self.max_seeds = int(seed_count), int(max_seeds)
        self.wrap_margin = float(wrap_margin)
        if not 0 < self.wrap_margin < np.pi:
            raise ValueError("wrap_margin must be in (0,pi)")
        self.TOLERANCE, self.SINGULARITY_THRESHOLD = 1e-9, 1e-15
        self.branches = {}
        self._cache = {}
        self._status_cache = {}
        self._viewer = None
        for name in (
            "branch1",
            "branch2",
            "math_branch1",
            "math_branch2",
            "full_math_data",
        ):
            setattr(self, name, np.empty((0, 4)))
        self.singular_states = []
        self.metrics = {}
        self.api_metrics = dict(
            candidate_queries=0, candidate_solves=0, continuation_calls=0
        )
        self._configuration_cache = {}

    @property
    def driving_crease(self):
        """Original input index. Construct a new simulator to change geometry/driver."""
        return self._driving_crease

    @property
    def drive_index(self):
        return self.driving_crease - 1

    def solve_states(self, drive):
        return self._core.solve(float(drive))

    def configuration_solutions(self, drive):
        """Atomic candidate/status/family query for consumers such as a strip.

        Angles and candidates use the original public crease order. Families
        explicitly describe underdetermined fixed-drive states. Returned arrays
        are copies, so consumers cannot corrupt a shared immutable model cache.
        """
        from copy import deepcopy

        drive = float(drive)
        if not np.isfinite(drive):
            raise ValueError("finite drive required")
        t = float(wrap_angle(drive))
        if drive > 0 and abs(t + np.pi) < 2e-14:
            t = float(np.pi)
        key = (round(t, 15), self._core.residual_tol)
        self.api_metrics["candidate_queries"] += 1
        if key not in self._configuration_cache:
            if len(self._configuration_cache) >= 4096:
                self._configuration_cache.pop(next(iter(self._configuration_cache)))
            cs = self._core.solve(t)
            self._configuration_cache[key] = dict(
                candidates=tuple(deepcopy(cs)),
                status=self._core.last_status,
                families=tuple(dict(f) for f in self._core.last_families),
                details=dict(self._core.last_details),
            )
            self.api_metrics["candidate_solves"] += 1
        item = self._configuration_cache[key]
        return dict(
            candidates=deepcopy(item["candidates"]),
            status=item["status"],
            families=tuple(dict(f) for f in item["families"]),
            details=dict(item["details"]),
        )

    def kinematic_intervals(self):
        """Representative drive domains, without building a sampled branch map."""
        if not hasattr(self, "_drive_domains"):
            self._drive_domains = self._domains()
        return tuple(self._drive_domains)

    def continue_configuration(self, rhos, target, tangent=None, *, contact=False):
        """Continue four original-order folds without changing the driving crease.

        This adapter preserves exact incoming limits at fixed-drive continua:
        A,C=O(delta^2), B=O(delta) imply p*q=0, hence free fold 0 or pi.
        It leaves the established solver/map APIs and their samples unchanged.
        """
        q = np.asarray(rhos, float).copy()
        if q.shape != (4,) or not np.isfinite(q).all() or not np.isfinite(target):
            raise ValueError("four finite folds and a finite target required")
        if closure_error(self.original_alphas, q) > self._core.residual_tol:
            raise PathError("NUMERICAL_FAILURE", "initial rotation closure")
        self.api_metrics["continuation_calls"] += 1
        t, target = float(q[self.drive_index]), float(target)
        self._check_drive_path(t, target)
        if contact:
            self._require_contact(q)
        if abs(target - t) < 1e-13:
            return q, None if tangent is None else np.asarray(tangent).copy(), None
        goals = list(
            np.linspace(t, target, max(1, int(np.ceil(abs(target - t) / 0.035))) + 1)[
                1:
            ]
        )
        lo, hi = sorted((t, target))
        goals.extend(
            k * np.pi
            for k in range(int(np.floor(lo / np.pi)) - 1, int(np.ceil(hi / np.pi)) + 2)
            if lo + 1e-12 < k * np.pi < hi - 1e-12
        )
        goals = sorted(set(goals), reverse=target < t)
        candidate = None
        for goal in goals:
            if abs(goal - t) < 1e-13:
                q[self.drive_index] = goal
                t = float(goal)
                continue
            packet = (
                self.configuration_solutions(goal)
                if abs(np.sin(goal)) <= 32 * np.finfo(float).eps
                else None
            )
            if packet is not None and packet["families"]:
                dt = goal - t
                predicted = q if tangent is None else q + np.asarray(tangent) * dt
                options = [
                    f["evaluate"](free)
                    for f in packet["families"]
                    for free in (0.0, np.pi)
                ]
                options = [c for c in options if c is not None]
                lifted = [lift_angles(c.rhos, predicted) for c in options]
                scores = [np.linalg.norm(v - predicted) for v in lifted]
                ids = np.argsort(scores)
                if not len(ids) or (
                    len(ids) > 1 and abs(scores[ids[0]] - scores[ids[1]]) < 1e-8
                ):
                    raise PathError("AMBIGUOUS", "incoming continuum limit")
                v, candidate = lifted[ids[0]], options[ids[0]]
                v[self.drive_index] = goal
                tangent = (v - q) / dt if abs(dt) > 1e-13 else tangent
                q = v
            else:
                q, tangent, candidate = self._step(q, t, float(goal), tangent)
            t = float(goal)
            if closure_error(self.original_alphas, q) > self._core.residual_tol:
                raise PathError("NUMERICAL_FAILURE", "continued rotation closure")
            if contact:
                self._require_contact(q)
        return q, tangent, candidate

    def _solve(self, drive):
        key = (round(float(wrap_angle(drive)), 15), self._core.residual_tol)
        if key not in self._cache:
            self.metrics["solve_calls"] = self.metrics.get("solve_calls", 0) + 1
            self._cache[key] = list(self._core.solve(key[0]))
            self._status_cache[key] = self._core.last_status
        self._core.last_status = self._status_cache[key]
        return self._cache[key]

    def _solve_quadratic_rho1(self, rho_drive):
        return self._core.quadratic_roots(float(rho_drive))

    def solve_full_rhos_from_drive_and_rho1(self, rho_drive, rho1):
        c = self._core.recover(float(rho_drive), float(rho1))
        return None if c is None else c.rhos.copy()

    def get_mv_pattern(self, rhos, tolerance=1e-7):
        return "".join(
            "V" if r > tolerance else "M" if r < -tolerance else "F"
            for r in wrap_angle(rhos)
        )

    def _domains(self):
        a1, a2, a3, a4 = self.alphas
        aa = np.cos(a1 + a3 - a4) - np.cos(a2)
        ac = np.cos(a1 - a3 - a4) - np.cos(a2)
        ca = np.cos(a1 - a3 + a4) - np.cos(a2)
        cc = np.cos(a1 + a3 + a4) - np.cos(a2)
        bb = 4 * np.sin(a1) * np.sin(a3)
        poly = np.array([-4 * aa * ca, bb * bb - 4 * (aa * cc + ac * ca), -4 * ac * cc])
        scale = max(abs(poly).max(), 1e-30)
        nz = np.flatnonzero(abs(poly) > scale * 1e-12)
        points = [-np.pi, np.pi]
        if len(nz):
            for z in np.roots(poly[nz[0] :] / scale):
                if abs(z.imag) < 1e-10 and z.real > 1e-12:
                    t = float(2 * np.arctan(np.sqrt(z.real)))
                    if 1e-7 < t < np.pi - 1e-7:
                        points.extend([-t, t])
        points = sorted(set(round(t, 13) for t in points))
        return [
            (a, b)
            for a, b in zip(points[:-1], points[1:])
            if any(self._solve(a + f * (b - a)) for f in (0.5, 0.371, 0.629))
        ]

    def _step(self, q, t, target, tangent=None, depth=0):
        dt = target - t
        predicted = q if tangent is None else q + tangent * dt
        cs = self._solve(target)
        if not cs and self._core.last_status == "CONTINUUM":
            # An exact special-family seam can contain the limit of this mode.
            # Its incoming tangent selects that limit, never an arbitrary sample.
            if tangent is None:
                raise PathError(
                    "AMBIGUOUS", "a fixed-drive continuum needs an incoming path"
                )
            self._core.solve(float(wrap_angle(target)))
            cs = [
                f["evaluate"](float(predicted[f["free_crease"] - 1]))
                for f in self._core.last_families
            ]
            cs = [c for c in cs if c is not None]
        if not cs:
            status = self._core.last_status
            raise PathError(
                "DRIVE_EXTREMUM" if status == "NO_SOLUTION" else "NUMERICAL_FAILURE",
                status,
            )
        lifted = [lift_angles(c.rhos, predicted) for c in cs]
        for v in lifted:
            v[self.drive_index] = target
        scores = np.array([np.linalg.norm(v - predicted) for v in lifted])
        ids = np.argsort(scores)
        best = int(ids[0])
        ambiguous = (
            len(ids) > 1
            and abs(scores[ids[1]] - scores[best]) < 1e-9
            and np.linalg.norm(lifted[ids[1]] - lifted[best]) > 1e-6
        )
        if (ambiguous or scores[best] > 0.16) and depth < 12 and abs(dt) > 1e-7:
            mid = (t + target) / 2
            qm, tm, _ = self._step(q, t, mid, tangent, depth + 1)
            return self._step(qm, mid, target, tm, depth + 1)
        if ambiguous:
            raise PathError("AMBIGUOUS", "indistinguishable continuations")
        if scores[best] > 0.8:
            raise PathError("NUMERICAL_FAILURE", "correction too large")
        v = lifted[best]
        return v, (v - q) / dt if abs(dt) > 1e-14 else tangent, cs[best]

    def _check_drive_path(self, start, target):
        lo, hi = sorted((start, target))
        domains = getattr(self, "_drive_domains", None)
        if domains is None:
            domains = self._domains()
            self._drive_domains = domains
        intervals = sorted(
            (a + k * TAU, b + k * TAU)
            for k in range(int(np.floor(lo / TAU)) - 2, int(np.ceil(hi / TAU)) + 3)
            for a, b in domains
        )
        cursor = lo
        for a, b in intervals:
            if b < cursor - 1e-10:
                continue
            if a > cursor + 1e-10:
                break
            cursor = max(cursor, b)
            if cursor >= hi - 1e-10:
                return
        raise PathError(
            "DRIVE_EXTREMUM", "movement crosses an infeasible drive interval"
        )

    def _advance(self, q, t, target, tangent=None, contact=False):
        self._check_drive_path(t, target)
        if contact:
            self._require_contact(q)
        if abs(target - t) < 1e-13:
            return q.copy(), tangent, None
        c = None
        for goal in np.linspace(
            t, target, max(1, int(np.ceil(abs(target - t) / 0.035))) + 1
        )[1:]:
            q, tangent, c = self._step(q, t, float(goal), tangent)
            t = float(goal)
            if contact:
                self._require_contact(q)
        return q, tangent, c

    def _name(self, c, rank, component):
        if self.is_euclidean and c.mode_id and c.mode_id.startswith("unique_e"):
            return "E_" + c.mode_id
        return (
            c.mode_id or ("E" if self.is_euclidean else "NE") + "_mode%d" % (rank + 1)
        ) + "_c%d" % component

    def _seed_checks(self, lo, hi, anchor, reference):
        if self.geometry["euclidean_flat_foldable"] or (
            reference
            and all((c.mode_id or "").startswith("collinear_") for c in reference)
        ):
            return 0
        good = 0
        for t in np.linspace(
            lo + 0.13 * (hi - lo), hi - 0.13 * (hi - lo), self.max_seeds
        ):
            self.metrics["seed_attempts"] += 1
            cs = self._solve(t)
            try:
                reached = []
                for c in cs:
                    q, _, _ = self._advance(c.rhos, float(t), anchor)
                    distances = [
                        np.linalg.norm(wrap_angle(q - r.rhos)) for r in reference
                    ]
                    if not distances or min(distances) > 1e-6:
                        raise PathError("AMBIGUOUS", "seed disagrees with reference")
                    reached.append(int(np.argmin(distances)))
                if len(set(reached)) != len(reached):
                    raise PathError("AMBIGUOUS", "seeds collapse onto one path")
            except PathError:
                continue
            if cs:
                good += 1
                self.metrics["seed_checks"] += 1
            if good >= self.seed_count:
                return good
        raise PathError("NUMERICAL_FAILURE", "insufficient valid interior seeds")

    def run_simulation(self, resolution=1000):
        if (
            isinstance(resolution, bool)
            or int(resolution) != resolution
            or resolution < 8
        ):
            raise ValueError("resolution must be an integer >=8")
        start = time.perf_counter()
        self.resolution = int(resolution)
        self.branches = {}
        self._cache.clear()
        self._status_cache.clear()
        self.metrics = dict(solve_calls=0, seed_attempts=0, seed_checks=0)
        grid = np.linspace(-np.pi + 1e-6, np.pi - 1e-6, int(resolution))
        all_rows = []
        canonical = min(
            range(4), key=lambda k: tuple(np.roll(self.original_alphas_deg, -k))
        )
        order = np.roll(np.arange(4), -canonical)
        self._drive_domains = self._domains()
        for component, (lo, hi) in enumerate(self._drive_domains):
            anchor = float(np.clip(0.071, lo + 0.2 * (hi - lo), hi - 0.2 * (hi - lo)))
            cs = sorted(
                self._solve(anchor), key=lambda c: tuple(np.round(c.rhos[order], 11))
            )
            checks = self._seed_checks(lo, hi, anchor, cs)
            for rank, c in enumerate(cs):
                bid = self._name(c, rank, component)
                if bid in self.branches:
                    bid += "_c%d" % component
                seed = c.rhos.copy()
                rows = [seed]
                ends = []
                reached = {}
                for direction in (-1, 1):
                    q, t, tangent = seed.copy(), anchor, None
                    targets = sorted(
                        [
                            x
                            for x in grid
                            if lo < x < hi and direction * (x - anchor) > 0
                        ],
                        reverse=direction < 0,
                    )
                    for goal in targets:
                        try:
                            q, tangent, _ = self._advance(q, t, float(goal), tangent)
                            t = float(goal)
                            rows.append(q.copy())
                        except PathError as exc:
                            ends.append(Endpoint(t, exc.reason, str(exc)))
                            reached[direction] = t
                            break
                    else:
                        boundary = lo if direction < 0 else hi
                        reached[direction] = boundary
                        ends.append(
                            Endpoint(
                                boundary,
                                (
                                    "DOMAIN_END"
                                    if abs(abs(boundary) - np.pi) < 1e-8
                                    else "DRIVE_EXTREMUM"
                                ),
                            )
                        )
                rows = np.array(sorted(rows, key=lambda q: q[self.drive_index]))
                all_rows.extend(rows)
                physical, contact_ends = self._contact_path(
                    rows, (reached[-1], reached[1])
                )
                ends.extend(contact_ends)
                self.branches[bid] = Branch(
                    bid,
                    c.mode_id or bid,
                    rows,
                    seed,
                    self._split(rows),
                    ends,
                    physical_segments=physical,
                    seed_checks=checks,
                    reachable_intervals=[(reached[-1], reached[1])],
                    admissible_domain=(lo, hi),
                )
        self._add_families(int(resolution))
        self._connect_seams()
        # Raw kinematic space is independent of branch tracing failures.
        raw = [c.rhos for t in grid for c in self._solve(t)]
        raw.extend(
            q for b in self.branches.values() if b.family is not None for q in b.states
        )
        self.full_math_data = np.asarray(raw).reshape(-1, 4)
        regular = [b for b in self.branches.values() if b.family is None]
        for i in range(2):
            b = regular[i] if i < len(regular) else None
            setattr(
                self,
                "math_branch%d" % (i + 1),
                b.states.copy() if b else np.empty((0, 4)),
            )
            setattr(
                self,
                "branch%d" % (i + 1),
                (
                    np.vstack(b.physical_segments)
                    if b and b.physical_segments
                    else np.empty((0, 4))
                ),
            )
        self.metrics["run_seconds"] = time.perf_counter() - start
        if self.verbose:
            print(
                self.geom_type,
                self.geometry,
                "branches",
                list(self.branches),
                self.metrics,
            )
        return self

    def _connect_seams(self):
        """Record verified periodic adjacency without merging actuation segments."""
        regular = [b for b in self.branches.values() if b.family is None]
        parents = {b.id: b.id for b in regular}

        def find(key):
            while parents[key] != key:
                key = parents[key]
            return key

        eps = 1e-4
        for b in regular:
            if b.reachable_intervals[0][1] < np.pi - 1e-8 or len(b.states) < 2:
                continue
            q0, q1 = b.states[-2:]
            t0, t1 = q0[self.drive_index], q1[self.drive_index]
            tangent = (q1 - q0) / (t1 - t0)
            try:
                continued, _, _ = self._advance(q1, t1, np.pi + eps, tangent)
            except PathError:
                continue
            matches = []
            for other in regular:
                if (
                    other.reachable_intervals[0][0] > -np.pi + 1e-8
                    or len(other.states) < 2
                ):
                    continue
                a, c = other.states[:2]
                ta, tc = a[self.drive_index], c[self.drive_index]
                try:
                    opposite, _, _ = self._advance(
                        a, ta, -np.pi + eps, (c - a) / (tc - ta)
                    )
                except PathError:
                    continue
                if np.linalg.norm(wrap_angle(continued - opposite)) < 1e-6:
                    matches.append(other)
            if len(matches) == 1:
                other = matches[0]
                b.seam_connections["positive"] = other.id
                other.seam_connections["negative"] = b.id
                left, right = find(b.id), find(other.id)
                parents[max(left, right)] = min(left, right)
        for b in regular:
            b.mode_id = find(b.id)

    def _add_families(self, resolution):
        seen = set()
        for t in (-np.pi, 0.0, np.pi):
            self._core.solve(t)
            for f in self._core.last_families:
                key = (f["mode_id"], round(float(t), 9))
                if key in seen:
                    continue
                seen.add(key)
                rows = []
                for free in np.linspace(-np.pi, np.pi, resolution):
                    c = f["evaluate"](float(free))
                    if c is not None:
                        q = c.rhos.copy()
                        q[self.drive_index] = float(t)
                        rows.append(q)
                if not rows:
                    continue
                rows = np.array(rows)
                bid = f["mode_id"] + (
                    "_minus_pi" if t < -3 else "_plus_pi" if t > 3 else ""
                )
                self.branches[bid] = Branch(
                    bid,
                    bid,
                    rows,
                    rows[len(rows) // 2].copy(),
                    [rows],
                    [Endpoint(t, "DRIVE_EXTREMUM", "fixed-drive continuum")],
                    f,
                    self._contact_pieces(rows),
                    reachable_intervals=[(t, t)],
                )

    def _split(self, rows):
        if not len(rows):
            return []
        principal = wrap_angle(rows)
        if len(rows) == 1:
            return [principal]
        dif = np.diff(rows[:, self.drive_index])
        gap = max(0.05, 3 * np.median(abs(dif)))
        cuts = (
            np.flatnonzero(
                (np.max(abs(np.diff(principal, axis=0)), axis=1) > np.pi)
                | (abs(dif) > gap)
            )
            + 1
        )
        return [p for p in np.split(principal, cuts) if len(p)]

    def _contact_pieces(self, rows):
        if not len(rows):
            return []
        good = np.array([self._contact_reason(q) is None for q in rows])
        runs = np.split(
            np.arange(len(rows)), np.flatnonzero(np.diff(good.astype(int))) + 1
        )
        return [rows[ix] for ix in runs if len(ix) and good[ix[0]]]

    def _contact_reason(self, q):
        if np.any(abs(q) > np.pi + 1e-8):
            return "continuous fold bound"
        hits = panel_penetrations(self._fold_vertices(q))
        if hits:
            return "panel penetration: " + ", ".join(
                f"P{hit['panels'][0]}/P{hit['panels'][1]}" for hit in hits
            )
        return None

    def _require_contact(self, q):
        reason = self._contact_reason(q)
        if reason is not None:
            raise PathError("CONTACT_LIMIT", reason)

    def contact_report(self, rhos_rad):
        """Inspect bounded angles and opposite-panel penetration independently."""
        _, vertices = self.get_3d_geometry(rhos_rad)
        hits = panel_penetrations(vertices)
        bounded = bool(np.all(abs(np.asarray(rhos_rad)) <= np.pi + 1e-8))
        return {
            "angle_bounds": bounded,
            "penetrations": hits,
            "admissible": bounded and not hits,
        }

    def _contact_path(self, rows, reached):
        """Filter contact intervals and retain exact isolated boundary states.

        This augments only contact data. Mathematical samples, seeds and IDs
        remain untouched. Boundary refinement is local, not a CCD certificate.
        """
        points = list(rows)
        ends = []
        for target in (reached[0], 0.0, reached[1]):
            if abs(abs(target) - np.pi) < 1e-12:
                target = float(np.copysign(np.pi, target))
            if not reached[0] - 1e-12 <= target <= reached[1] + 1e-12:
                continue
            ids = np.argsort(abs(rows[:, self.drive_index] - target))[:2]
            if len(ids) < 2:
                continue
            a, z = rows[ids]
            dt = z[self.drive_index] - a[self.drive_index]
            if abs(dt) < 1e-12:
                continue
            try:
                if self._core._is_continuum(target):
                    # At a continuum seam A=C=0, A,C=O(delta^2),
                    # B=K*delta+O(delta^3), K!=0 for 0<alpha_i<pi.
                    # Dividing A*p^2+B*p*q+C*q^2=0 by delta gives
                    # p*q=0. Isolated incoming modes therefore limit to
                    # the family's free angle 0 or pi, not the arbitrary
                    # first-order extrapolation along the entire continuum.
                    family = self._core._family(target)
                    prediction = a + (z - a) / dt * (target - a[self.drive_index])
                    limits = [family["evaluate"](free) for free in (0.0, np.pi)]
                    candidates = [
                        lift_angles(c.rhos, prediction) for c in limits if c is not None
                    ]
                    candidates.sort(key=lambda q: np.linalg.norm(q - prediction))
                    if not candidates or (
                        len(candidates) > 1
                        and abs(
                            np.linalg.norm(candidates[0] - prediction)
                            - np.linalg.norm(candidates[1] - prediction)
                        )
                        < 1e-7
                    ):
                        raise PathError("AMBIGUOUS", "continuum boundary limit")
                    q = candidates[0]
                    q[self.drive_index] = float(target)
                else:
                    if np.min(abs(rows[:, self.drive_index] - target)) < 1e-12:
                        continue
                    q, _, _ = self._advance(
                        a, float(a[self.drive_index]), float(target), (z - a) / dt
                    )
                points = [
                    p for p in points if abs(p[self.drive_index] - target) > 1e-12
                ]
                points.append(q)
            except PathError as error:
                ends.append(
                    Endpoint(
                        float(target),
                        error.reason,
                        "contact boundary evaluation: " + str(error),
                    )
                )
                continue
        points.sort(key=lambda q: q[self.drive_index])
        good = [self._contact_reason(q) is None for q in points]
        refined = []
        for ix in np.flatnonzero(np.diff(np.array(good, int))):
            inside, outside = (
                (points[ix], points[ix + 1])
                if good[ix]
                else (points[ix + 1], points[ix])
            )
            initial = inside
            ti, to = float(inside[self.drive_index]), float(outside[self.drive_index])
            tangent = (outside - inside) / (to - ti)
            reason = self._contact_reason(outside)
            resolved = True
            for _ in range(25):
                if abs(to - ti) < 1e-8:
                    break
                mid = (ti + to) / 2
                try:
                    qm, _, _ = self._advance(inside, ti, mid, tangent)
                except PathError as error:
                    ends.append(Endpoint(ti, error.reason, str(error)))
                    resolved = False
                    break
                if self._contact_reason(qm) is None:
                    inside, ti = qm, mid
                else:
                    to = mid
            else:
                ends.append(Endpoint(ti, "NUMERICAL_FAILURE", "contact refinement"))
                continue
            if not resolved:
                continue
            if abs(ti - initial[self.drive_index]) < 1e-7:
                ti = float(initial[self.drive_index])
            else:
                refined.append(inside)
            ends.append(Endpoint(ti, "CONTACT_LIMIT", reason or "panel contact"))
        points.extend(refined)
        points.sort(key=lambda q: q[self.drive_index])
        return self._contact_pieces(np.array(points)), ends

    def plot_data(self, branch_id, contact=None):
        b = self.branches[branch_id]
        bounded = self.contact if contact is None else contact
        return [
            s
            for p in (b.physical_segments if bounded else [b.states])
            for s in self._split(p)
        ]

    def create_session(self, branch=None, contact=None):
        if not self.branches:
            self.run_simulation()
        if not self.branches:
            raise PathError("NO_SOLUTION")
        branch = next(iter(self.branches)) if branch is None else branch
        if branch not in self.branches:
            raise ValueError("Unknown branch " + str(branch))
        return SimulationSession(
            self, branch, self.contact if contact is None else contact
        )

    def compute_folding_angles_from_input_output(
        self, input_angle_deg, output_index, sign=1, verbose=False, branch=None
    ):
        if not isinstance(output_index, (int, np.integer)) or output_index not in range(
            4
        ):
            raise ValueError("output_index must be 0..3")
        if output_index == self.drive_index or sign not in (-1, 1):
            raise ValueError("output must differ from drive; sign must be +/-1")
        angles = np.deg2rad(np.atleast_1d(input_angle_deg).astype(float))
        if not np.all(np.isfinite(angles)):
            raise ValueError("finite angles required")
        solutions = []
        for t in angles:
            if branch is None:
                cs = [c.rhos for c in self._solve(t)]
            else:
                try:
                    cs = [self.create_session(branch).move_to(t).rhos]
                except PathError:
                    cs = []
            solutions.append(
                [
                    q
                    for q in cs
                    if abs(t) < 1e-12 or np.sign(q[output_index]) == sign * np.sign(t)
                ]
            )
        count = max([len(s) for s in solutions] + [0])
        out = np.full((len(angles), count), np.nan)
        full = np.full((len(angles), count, 4), np.nan)
        for i, ss in enumerate(solutions):
            for j, q in enumerate(ss):
                out[i, j] = q[output_index]
                full[i, j] = q
        return dict(
            input_angles=angles,
            output_angles=out,
            folding_angles_all=full,
            num_solutions=np.array([len(s) for s in solutions]),
        )

    def _unwrap_filter(self, path, step_size):
        q = np.asarray(path, float).reshape(-1, 4)
        return q[np.all(abs(q) <= np.pi + 1e-8, axis=1)]

    def _rodrigues_rotation_matrix(self, axis, angle):
        return rotation(axis, angle)

    def get_3d_geometry(self, rhos_rad):
        q = np.asarray(rhos_rad, float)
        if q.shape != (4,) or not np.isfinite(q).all():
            raise ValueError("four finite folds required")
        if closure_error(self.original_alphas, q) > max(self._core.residual_tol, 1e-7):
            raise ValueError("folding angles do not satisfy rotation closure")
        v = self._fold_vertices(q)
        return [
            [v[0], v[4], v[1]],
            [v[0], v[1], v[2]],
            [v[0], v[2], v[3]],
            [v[0], v[3], v[4]],
        ], v

    def _fold_vertices(self, q):
        """Construct the same panel mesh after the caller validated closure."""
        frame = rz(self.original_alphas[3])
        vertices = [np.zeros(3), frame[:, 0].copy()]
        for i in range(3):
            frame = frame @ rx(q[i]) @ rz(self.original_alphas[i])
            vertices.append(frame[:, 0].copy())
        return np.array(vertices)

    def plot_2d_static(self, branch=None, contact=None, show=True):
        import matplotlib.pyplot as plt

        if not self.branches:
            self.run_simulation()
        fig, ax = plt.subplots(figsize=(8, 8))
        for bid in ([branch] if branch else self.branches):
            for p in self.plot_data(bid, contact):
                for j in range(4):
                    if j != self.drive_index:
                        ax.plot(
                            p[:, self.drive_index],
                            p[:, j],
                            label=bid + " e%d" % (j + 1),
                            marker="o" if len(p) == 1 else None,
                        )
        ax.set(
            xlim=(-np.pi, np.pi),
            ylim=(-np.pi, np.pi),
            xlabel="rho%d [rad]" % self.driving_crease,
            ylabel="fold angle [rad]",
        )
        ax.grid(True)
        if show:
            plt.show()
        return fig

    def show_animated_2d_plot(self, interval=0.001, skip_step=10):
        import matplotlib.pyplot as plt

        fig = self.plot_2d_static(show=False)
        for b in self.branches.values():
            for p in self.plot_data(b.id):
                for q in p[::skip_step]:
                    fig.axes[0].scatter(
                        [q[self.drive_index]] * 3, np.delete(q, self.drive_index), s=5
                    )
                    plt.pause(interval)
        plt.show()
        return fig

    def show_3d_interactive(
        self, branch=None, contact=None, show=True, wrap_margin=None, random_seed=None
    ):
        from origami_viewer import create_viewer

        if not self.branches:
            self.run_simulation()
        self._viewer = create_viewer(
            self,
            branch,
            contact,
            self.wrap_margin if wrap_margin is None else wrap_margin,
            show=show,
            random_seed=random_seed,
        )
        return self._viewer


class SimulationSession:
    def __init__(self, sim, branch, contact):
        self.sim, self.branch_id, self.contact = sim, branch, bool(contact)
        self.branch = sim.branches[branch]
        self.is_family = self.branch.family is not None
        self.free_angle = 0.0
        self.tangent = None
        self.last_error = None
        self.contact_reference = None
        self._contact_reference_tangent = None
        self.contact_reference_history = []
        self._contact_plot_cache = None
        self.contact_plot_endpoints = []
        self._initial_contact_piece = None
        q = self.branch.seed_state.copy()
        if self.contact:
            choices = [
                (float(np.linalg.norm(row - q)), row, piece)
                for piece in self.branch.physical_segments
                for row in piece
            ]
            if not choices:
                raise PathError(
                    "CONTACT_LIMIT", "this path has no admissible contact start"
                )
            _, q, self._initial_contact_piece = min(choices, key=lambda item: item[0])
            q = q.copy()
            if not self.is_family:
                ids = np.argsort(
                    abs(self.branch.states[:, sim.drive_index] - q[sim.drive_index])
                )[:2]
                if len(ids) == 2:
                    a, z = self.branch.states[ids]
                    dt = z[sim.drive_index] - a[sim.drive_index]
                    if abs(dt) > 1e-12:
                        self.tangent = (z - a) / dt
        if self.is_family:
            self.free_angle = float(q[self.branch.family["free_crease"] - 1])
        self.state = self._state(q, float(q[sim.drive_index]), ())

    @property
    def drive_locked(self):
        """The current contact component contains only one resolved drive value."""
        if not self.contact or self.is_family:
            return False
        pieces = self.plot_data()
        if not pieces:
            return True
        drives = np.concatenate([p[:, self.sim.drive_index] for p in pieces])
        return bool(np.ptp(drives) < 1e-7)

    def _state(self, q, t, events):
        wrapped = float(wrap_angle(t))
        if t > 0 and abs(wrapped + np.pi) < 2e-14:
            wrapped = float(np.pi)
        winding = int(round((t - wrapped) / TAU))
        return State(
            q.copy(),
            float(t),
            wrapped,
            winding,
            self.branch_id,
            float(closure_error(self.sim.original_alphas, q)),
            tuple(events),
            self.sim.get_mv_pattern(q),
        )

    def move_to(self, target):
        target = float(target)
        if not np.isfinite(target):
            raise ValueError("finite drive required")
        if self.is_family:
            if (
                abs(float(wrap_angle(target - self.branch.family["fixed_drive"])))
                > 1e-8
            ):
                raise PathError("DRIVE_EXTREMUM", "fixed-drive family")
            return self.state
        if self.contact and abs(target) > np.pi + 1e-10:
            raise PathError("DOMAIN_END", "contact drive domain")
        if (
            self.contact
            and self.drive_locked
            and abs(target - self.state.drive_unwrapped) > 1e-12
        ):
            raise PathError(
                "CONTACT_LIMIT", "isolated contact state; driving fold is fixed"
            )
        old = self.state
        q, tangent, _ = self.sim._advance(
            old.rhos, old.drive_unwrapped, target, self.tangent, self.contact
        )
        new = self._state(q, target, ())
        if new.residual > 1e-7:
            raise PathError("NUMERICAL_FAILURE", "closure residual")
        events = []
        if new.winding_count != old.winding_count:
            events.append("ANGLE_WRAP")
        if new.mv_pattern != old.mv_pattern:
            events.append("MV_CHANGE")
        new.events = tuple(events)
        self.state, self.tangent = new, tangent
        return new

    def move_slider(self, raw_display):
        return self.move_to(
            float(raw_display) + (0 if self.contact else TAU * self.state.winding_count)
        )

    def move_family(self, free_angle):
        if not self.is_family:
            raise ValueError("not a fixed-drive family")
        target = float(free_angle)
        if not np.isfinite(target):
            raise ValueError("finite free angle required")
        q = self.state.rhos.copy()
        index = self.branch.family["free_crease"] - 1
        for goal in np.linspace(
            self.free_angle,
            target,
            max(1, int(np.ceil(abs(target - self.free_angle) / 0.035))) + 1,
        )[1:]:
            c = self.branch.family["evaluate"](float(goal))
            if c is None:
                raise PathError("NUMERICAL_FAILURE", "family state")
            q = lift_angles(c.rhos, q)
            q[index] = float(goal)
            if self.contact:
                self.sim._require_contact(q)
        self.free_angle = target
        self.state = self._state(q, self.state.drive_unwrapped, ("FAMILY_MOVE",))
        return self.state

    def _copy_state(self):
        s = self.state
        return State(
            s.rhos.copy(),
            s.drive_unwrapped,
            s.drive_wrapped,
            s.winding_count,
            s.branch_id,
            s.residual,
            s.events,
            s.mv_pattern,
        )

    def set_contact(self, enabled, *, reanchor=False):
        """Enable bounds strictly, or explicitly start a new bounded path here.

        Reanchoring changes the angle lift, never the shape or branch. Past
        unbounded motion is not retrospectively declared contact-admissible.
        The viewer requests this policy explicitly; the legacy API is strict.
        """
        enabled = bool(enabled)
        if enabled == self.contact:
            self.last_error = None
            return True
        if enabled:
            # Reanchoring cannot remove intersections: it only changes angles
            # by multiples of 2*pi, which preserve the panel geometry.
            hits = panel_penetrations(self.sim._fold_vertices(self.state.rhos))
            if hits:
                self.last_error = "CONTACT_LIMIT: current shape has panel penetration"
                return False
        if enabled and (
            abs(self.state.drive_unwrapped) > np.pi + 1e-10
            or np.any(abs(self.state.rhos) > np.pi + 1e-8)
        ):
            if not reanchor:
                self.last_error = "CONTACT_LIMIT: continuous state outside bounds"
                return False
            old = self._copy_state()
            q = wrap_angle(old.rhos)
            at_positive_pi = (old.rhos > 0) & (abs(q + np.pi) < 2e-14)
            q[at_positive_pi] = np.pi
            q[self.sim.drive_index] = old.drive_wrapped
            residual = float(closure_error(self.sim.original_alphas, q))
            if (
                not np.isfinite(residual)
                or residual > max(self.sim._core.residual_tol, 1e-7)
                or np.linalg.norm(wrap_angle(q - old.rhos)) > 1e-9
            ):
                self.last_error = "NUMERICAL_FAILURE: cannot preserve shape while resetting contact reference"
                return False
            self.state = self._state(q, old.drive_wrapped, ("CONTACT_REFERENCE_RESET",))
            self.contact_reference_history.append(old)
            if self.is_family:
                self.free_angle = float(q[self.branch.family["free_crease"] - 1])
        self.contact = enabled
        if enabled:
            self.contact_reference = self._copy_state()
            self._contact_reference_tangent = (
                None if self.tangent is None else self.tangent.copy()
            )
            self._contact_plot_cache = None
        self.last_error = None
        return True

    def plot_data(self):
        """Display the current bounded path from its actual reference state."""
        if not self.contact:
            return [
                p
                for b in self.sim.branches.values()
                if b.mode_id == self.branch.mode_id
                for p in self.sim.plot_data(b.id, False)
            ]
        if self.contact_reference is None:
            return self.sim._split(self._initial_contact_piece)
        if self._contact_plot_cache is not None:
            return self._contact_plot_cache
        anchor = self.contact_reference
        rows = [anchor.rhos.copy()]
        self.contact_plot_endpoints = []
        count = max(81, len(self.branch.states))
        free_index = self.branch.family["free_crease"] - 1 if self.is_family else None
        origin = anchor.rhos[free_index] if self.is_family else anchor.drive_unwrapped
        for direction in (-1, 1):
            q = anchor.rhos.copy()
            t = float(origin)
            tangent = self._contact_reference_tangent if not self.is_family else None
            goals = sorted(
                (
                    x
                    for x in np.linspace(-np.pi, np.pi, count)
                    if direction * (x - origin) > 1e-12
                ),
                reverse=direction < 0,
            )
            for goal in goals:
                try:
                    if self.is_family:
                        for sub in np.linspace(
                            t, goal, max(1, int(np.ceil(abs(goal - t) / 0.035))) + 1
                        )[1:]:
                            c = self.branch.family["evaluate"](float(sub))
                            if c is None:
                                raise PathError("NUMERICAL_FAILURE", "family plot")
                            q = lift_angles(c.rhos, q)
                            q[free_index] = float(sub)
                            q[self.sim.drive_index] = anchor.drive_unwrapped
                            self.sim._require_contact(q)
                    else:
                        q, tangent, _ = self.sim._advance(
                            q, t, float(goal), tangent, True
                        )
                    t = float(goal)
                    rows.append(q.copy())
                except PathError as exc:
                    self.contact_plot_endpoints.append(
                        Endpoint(t, exc.reason, str(exc))
                    )
                    break
            else:
                self.contact_plot_endpoints.append(Endpoint(t, "DOMAIN_END"))
        order = free_index if self.is_family else self.sim.drive_index
        data = np.array(sorted(rows, key=lambda q: q[order]))
        self._contact_plot_cache = self.sim._split(data)
        return self._contact_plot_cache


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alphas", nargs=4, type=float, default=[70, 20, 110, 160])
    p.add_argument("--drive", type=int, choices=range(1, 5))
    p.add_argument("--resolution", type=int, default=1000)
    p.add_argument("--branch")
    p.add_argument("--no-contact", action="store_true")
    p.add_argument("--no-show", action="store_true")
    p.add_argument(
        "--random-seed", type=int, help="Reproduce the no-contact viewer start."
    )
    args = p.parse_args()
    sim = OrigamiDegree4Simulator(
        args.alphas, contact=not args.no_contact, driving_crease=args.drive
    )
    sim.run_simulation(args.resolution)
    sim.show_3d_interactive(
        branch=args.branch, show=not args.no_show, random_seed=args.random_seed
    )
