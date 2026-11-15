"""Degree-four spherical kinematics, independent of plotting and branch tracing.

The general solver retains Foschi et al. PRE 106, 055001 (2022), Eq. 6.
Its homogeneous coordinates retain angular roots at infinity.  Recovery uses
oriented panel normals rather than guessing signs of squared half tangents.
"""

from dataclasses import dataclass, field
import numpy as np

PI = np.pi
TAU = 2 * PI


def wrap_angle(value):
    """Principal representative in [-pi, pi); works on arrays and scalars."""
    return (np.asarray(value) + PI) % TAU - PI


def rx(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotation(axis, angle):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def closure_matrix(alphas_rad, rhos):
    M = np.eye(3)
    for a, r in zip(alphas_rad, rhos):
        M = M @ rx(r) @ rz(a)
    return M


def closure_error(alphas_rad, rhos):
    return float(np.linalg.norm(closure_matrix(alphas_rad, rhos) - np.eye(3)))


def closure_jacobian(alphas_rad, rhos):
    M, cols = np.eye(3), []
    for a, r in zip(alphas_rad, rhos):
        cols.append(M[:, 0].copy())
        M = M @ rx(r) @ rz(a)
    return np.column_stack(cols)


def classify_geometry(alphas_deg):
    a = np.asarray(alphas_deg, dtype=float)
    if a.shape != (4,) or not np.all(np.isfinite(a)):
        raise ValueError("alphas_deg must contain four finite sector angles")
    if np.any(a <= 0) or np.any(a >= 360):
        raise ValueError("Sector angles must lie strictly between 0 and 360 degrees")
    # A sector of 180 degrees makes its panel-normal convention degenerate.
    # Preserve the input classification; the solver reports it explicitly.
    exact_tol = 64 * np.finfo(float).eps * max(360.0, float(np.sum(a)))
    near_tol = 1e-3
    close = lambda x: abs(float(x)) <= exact_tol
    surplus = float(np.sum(a) - 360.0)
    case = (
        "developable"
        if close(surplus)
        else ("elliptic" if surplus < 0 else "hyperbolic")
    )
    rotations, reflections = [], []
    for k in range(4):
        perm = [(j + k) % 4 for j in range(4)]
        if np.max(abs(a - a[perm])) <= exact_tol:
            rotations.append(
                {
                    "shift": k,
                    "sector_map": [p + 1 for p in perm],
                    "crease_map": [((j + k) % 4) + 1 for j in range(4)],
                }
            )
        perm = [(k - j) % 4 for j in range(4)]
        if np.max(abs(a - a[perm])) <= exact_tol:
            reflections.append(
                {
                    "axis": k,
                    "sector_map": [p + 1 for p in perm],
                    "crease_map": [((k - j + 1) % 4) + 1 for j in range(4)],
                }
            )
    collinear = sorted(
        {
            tuple(sorted((i + 1, (i + 2) % 4 + 1)))
            for i in range(4)
            if close(a[i] + a[(i + 1) % 4] - 180)
        }
    )
    equal_pairs = [
        (i + 1, j + 1) for i in range(4) for j in range(i + 1, 4) if close(a[i] - a[j])
    ]
    ff = close(a[0] + a[2] - a[1] - a[3])
    euclidean_ff = case == "developable" and ff
    near = {
        "euclidean": exact_tol < abs(surplus) <= near_tol,
        "flat_foldable": not ff and abs(a[0] + a[2] - a[1] - a[3]) <= near_tol,
        "collinear": any(
            exact_tol < abs(a[i] + a[(i + 1) % 4] - 180) <= near_tol for i in range(4)
        ),
        "equal_angles": any(
            exact_tol < abs(a[i] - a[j]) <= near_tol
            for i in range(4)
            for j in range(i + 1, 4)
        ),
    }
    unique = (
        [
            i + 1
            for i in range(4)
            if a[(i - 1) % 4] + a[i] < a[(i + 1) % 4] + a[(i + 2) % 4] - exact_tol
        ]
        if case == "developable"
        else []
    )
    special = bool(ff or collinear or equal_pairs or len(rotations) > 1 or reflections)
    return {
        "case": case,
        "angular_surplus_deg": surplus,
        "flat_foldable": bool(ff),
        "euclidean_flat_foldable": bool(euclidean_ff),
        "collinear": bool(collinear),
        "collinear_pairs": collinear,
        "rotations": rotations,
        "reflections": reflections,
        "rotational_symmetry": len(rotations) > 1,
        "reflection_symmetry": bool(reflections),
        "equal_angle_pairs": equal_pairs,
        "unique_folds": unique,
        "special": special,
        "near": near,
        "exact_tolerance_deg": exact_tol,
        "near_tolerance_deg": near_tol,
        "degenerate_sectors": [i + 1 for i in range(4) if close(a[i] - 180)],
    }


@dataclass
class Candidate:
    rhos: np.ndarray
    residual: float
    mode_id: str | None = None
    events: tuple = ()
    status: str = "REGULAR"
    jacobian_singular_values: np.ndarray = field(default_factory=lambda: np.empty(0))


class Kinematics:
    """Pure-state solver; driving_crease is an ORIGINAL one-based crease index.

    last_status distinguishes NO_SOLUTION, NUMERICAL_FAILURE, CONTINUUM and OK.
    In a fixed-drive continuum, last_families contains explicit evaluators;
    solve does not replace a continuum with two arbitrarily chosen samples.
    """

    def __init__(self, alphas_deg, driving_crease=None, residual_tol=2e-8):
        self.alphas_deg = np.asarray(alphas_deg, float)
        self.geometry = classify_geometry(self.alphas_deg)
        self.alphas_rad = np.deg2rad(self.alphas_deg)
        if driving_crease is None:
            driving_crease = 1 if self.geometry["case"] == "elliptic" else 4
        if isinstance(driving_crease, bool) or driving_crease not in (1, 2, 3, 4):
            raise ValueError(
                "driving_crease must be one of original e1,e2,e3,e4 (1..4)"
            )
        self.driving_crease = int(driving_crease)
        self.shift_amount = 4 - self.driving_crease
        self.internal_alphas = np.roll(self.alphas_rad, self.shift_amount)
        self.residual_tol = float(residual_tol)
        self.last_status = "NOT_RUN"
        self.last_families = []
        self.last_details = {}
        self.mode_rules = []
        if self.geometry["euclidean_flat_foldable"]:
            self.mode_rules.append(
                {
                    "type": "flat_foldable",
                    "condition": "a3=pi-a1, a4=pi-a2 (internal cyclic indices)",
                    "equation": "sin(a1+a2)*y^2-2*sin(a1)*x*y+sin(a1-a2)*x^2=0",
                    "factors": [
                        "cos((a1+a2)/2)*p*c-cos((a1-a2)/2)*q*s=0",
                        "sin((a1+a2)/2)*p*c-sin((a1-a2)/2)*q*s=0",
                    ],
                    "coordinates": "x=tan(d/2), y=tan(r1/2); p/q=y, s/c=x",
                    "limitations": "zero homogeneous pairs are explicit fixed-drive families",
                }
            )
        if self.geometry["collinear"]:
            self.mode_rules.append(
                {
                    "type": "collinear",
                    "pairs": self.geometry["collinear_pairs"],
                    "equation": "substitute adjacent sector sum pi in the general homogeneous quadratic",
                    "limitations": "families at d=0 or pi evaluated geometrically; no epsilon perturbation",
                }
            )
        if self.geometry["special"] and not (
            self.geometry["euclidean_flat_foldable"] or self.geometry["collinear"]
        ):
            self.mode_rules.append(
                {
                    "type": "symmetric_general_fallback",
                    "limitations": "symmetry alone does not add a derived closed-form mode rule; general equation used explicitly",
                }
            )

    @staticmethod
    def _cosdiff(u, v):
        p, q = np.sin((u + v) / 2), np.sin((u - v) / 2)
        # Snap trigonometric zeros of input geometry, never the drive itself.
        if abs(p) < 2e-15:
            p = np.longdouble(0)
        if abs(q) < 2e-15:
            q = np.longdouble(0)
        return -2 * p * q

    def coefficients(self, drive):
        a1, a2, a3, a4 = np.asarray(self.internal_alphas, np.longdouble)
        s, c = np.sin(np.longdouble(drive) / 2), np.cos(np.longdouble(drive) / 2)
        cd = self._cosdiff
        A = s * s * cd(a1 + a3 - a4, a2) + c * c * cd(a1 - a3 - a4, a2)
        B = 4 * s * c * np.sin(a1) * np.sin(a3)
        C = c * c * cd(a1 + a3 + a4, a2) + s * s * cd(a1 - a3 + a4, a2)
        if self.geometry["case"] == "developable":
            # sum(alpha)=2pi makes the constant C term identically zero.
            C = s * s * cd(a1 - a3 + a4, a2)
        return np.array([A, B, C], dtype=np.longdouble)

    def _is_continuum(self, drive, abc=None):
        # Substitute the exact angular seam first, then test geometry-only
        # trigonometric identities.  Includes generalized bird's feet as
        # well as Euclidean collinearity; no perturbation of input geometry.
        if abs(np.sin(float(drive))) > 32 * np.finfo(float).eps:
            return False
        a1, a2, a3, a4 = np.asarray(self.internal_alphas, np.longdouble)
        cd = self._cosdiff
        if np.cos(float(drive)) > 0:
            A, C = cd(a1 - a3 - a4, a2), cd(a1 + a3 + a4, a2)
        else:
            A, C = cd(a1 + a3 - a4, a2), cd(a1 - a3 + a4, a2)
        return bool(A == 0 and C == 0)

    @property
    def family_drive_values(self):
        return [d for d in (0.0, PI) if self._is_continuum(d)]

    def _family(self, drive):
        free = ((0 - self.shift_amount) % 4) + 1
        fixed_repr = 0 if abs(np.sin(drive / 2)) < 1e-10 else 180
        name = f"family_drive_e{self.driving_crease}_{fixed_repr}_free_e{free}"

        def evaluate(angle):
            candidate = self.recover(drive, angle)
            if candidate is not None:
                candidate.mode_id = name
                candidate.events += ("FIXED_DRIVE_FAMILY",)
            return candidate

        return {
            "fixed_drive": float(drive),
            "free_crease": free,
            "evaluate": evaluate,
            "mode_id": name,
            "valid_intervals": [(-PI, PI)],
            "rule": "A=B=C=0 after exact drive seam and sector identities substitution",
        }

    def quadratic_roots(self, drive):
        """Internal rho1 roots. Projective infinity is the finite angle pi."""
        self.last_families = []
        self._discriminant_roundoff = False
        if not np.isfinite(drive):
            raise ValueError("drive must be finite radians")
        if self.geometry["degenerate_sectors"]:
            self.last_status = "UNSUPPORTED_DEGENERATE_SECTOR"
            return []
        abc = self.coefficients(drive)
        if self._is_continuum(drive, abc):
            self.last_status = "CONTINUUM"
            self.last_families = [self._family(drive)]
            return []
        if self.geometry["euclidean_flat_foldable"]:
            u, v = self.internal_alphas[:2]
            s, c = np.sin(drive / 2), np.cos(drive / 2)
            clean = lambda t: 0.0 if abs(t) < 2e-15 else t
            pairs = [
                (clean(np.cos((u - v) / 2)) * s, clean(np.cos((u + v) / 2)) * c),
                (clean(np.sin((u - v) / 2)) * s, clean(np.sin((u + v) / 2)) * c),
            ]
            self.last_status = "OK"
            roots = [
                float(wrap_angle(2 * np.arctan2(p, q)))
                for p, q in pairs
                if abs(p) + abs(q) > 1e-16
            ]
            return self._dedup_angles(roots)
        scale = np.max(abs(abc))
        if scale == 0:
            self.last_status = "UNRESOLVED_DEGENERACY"
            return []
        A, B, C = abc / scale
        # Exact coefficient zeros may be represented with float input errors.
        tol = np.longdouble(2e-15)
        roots = []
        if abs(A) <= tol:
            roots.append(PI)
            if abs(B) > tol:
                roots.append(float(2 * np.arctan2(-C, B)))
        else:
            delta = B * B - 4 * A * C
            # Geometry and the driving angle originate as float64, even when
            # intermediate arithmetic uses longdouble. At a repeated root,
            # using only longdouble epsilon rejects representable closed states.
            dtol = 64 * np.finfo(float).eps * (B * B + abs(4 * A * C) + 1)
            if delta < -dtol:
                self.last_status = "NO_SOLUTION"
                return []
            self._discriminant_roundoff = bool(delta < 0)
            delta = max(np.longdouble(0), delta)
            q = -np.longdouble(0.5) * (B + np.copysign(np.sqrt(delta), B))
            if abs(q) > np.longdouble(1e-30):
                roots.extend([float(2 * np.arctan2(q, A)), float(2 * np.arctan2(C, q))])
            else:
                roots.append(0.0)
        self.last_status = "OK"
        return self._dedup_angles(roots)

    @staticmethod
    def _dedup_angles(roots):
        out = []
        for root in roots:
            root = float(wrap_angle(root))
            if all(abs(float(wrap_angle(root - old))) > 2e-10 for old in out):
                out.append(root)
        return sorted(out)

    def recover(self, drive, rho1):
        """Recover all angles from oriented normals; both arguments are radians."""
        if not np.isfinite(drive) or not np.isfinite(rho1):
            return None
        a1, a2, a3, a4 = self.internal_alphas
        if min(abs(np.sin([a1, a2, a3, a4]))) < 1e-13:
            return None
        v1 = np.array([np.cos(a4), np.sin(a4), 0.0])
        n0 = np.array([0.0, 0.0, 1.0])
        R1 = rotation(v1, rho1)
        v2 = R1 @ np.array([np.cos(a4 + a1), np.sin(a4 + a1), 0.0])
        R4 = rx(-drive)
        v3 = R4 @ np.array([np.cos(a3), -np.sin(a3), 0.0])
        if abs(float(np.dot(v2, v3) - np.cos(a2))) > self.residual_tol:
            return None
        n1 = R1 @ n0
        n2 = np.cross(v2, v3) / np.sin(a2)
        n2 = n2 / np.linalg.norm(n2)
        n3 = R4 @ n0
        signed = lambda axis, n, m: np.arctan2(
            np.dot(axis, np.cross(n, m)), np.dot(n, m)
        )
        rho2 = signed(v2, n1, n2)
        rho3 = signed(v3, n2, n3)
        internal = np.array([float(wrap_angle(rho1)), rho2, rho3, float(drive)])
        rhos = np.roll(internal, -self.shift_amount)
        residual = closure_error(self.alphas_rad, rhos)
        if not np.isfinite(residual) or residual > self.residual_tol:
            return None
        jac = closure_jacobian(self.alphas_rad, rhos)
        sv = np.linalg.svd(jac, compute_uv=False)
        dependent = np.delete(jac, self.driving_crease - 1, axis=1)
        dependent_sv = np.linalg.svd(dependent, compute_uv=False)
        events = []
        if sv[-1] >= 1e-8 and dependent_sv[-1] < 1e-8:
            events.append("DRIVE_PROJECTION_SINGULARITY")
        if sv[-1] < 1e-8:
            events.append("KINEMATIC_SINGULARITY")
        if any(abs(abs(float(wrap_angle(r))) - PI) < 1e-9 for r in rhos):
            events.append("ANGLE_CHART_SEAM")
        mode = None
        if self.geometry["case"] == "developable" and not self.geometry["special"]:
            signs = np.sign(rhos)
            for i in self.geometry["unique_folds"]:
                other = np.delete(signs, i - 1)
                if (
                    abs(rhos[i - 1]) > 1e-9
                    and np.all(abs(np.delete(rhos, i - 1)) > 1e-9)
                    and np.all(other == -signs[i - 1])
                ):
                    mode = f"unique_e{i}"
                    break
        return Candidate(
            rhos,
            residual,
            mode,
            tuple(events),
            "KINEMATIC_SINGULARITY" if sv[-1] < 1e-8 else "REGULAR",
            sv,
        )

    def solve(self, drive):
        roots = self.quadratic_roots(float(drive))
        candidates = []
        if self.geometry["euclidean_flat_foldable"] and self.last_status == "OK":
            u, v = self.internal_alphas[:2]
            s, c = np.sin(drive / 2), np.cos(drive / 2)
            clean = lambda t: 0.0 if abs(t) < 2e-15 else t
            pairs = [
                (
                    "ff_plus",
                    clean(np.cos((u - v) / 2)) * s,
                    clean(np.cos((u + v) / 2)) * c,
                ),
                (
                    "ff_minus",
                    clean(np.sin((u - v) / 2)) * s,
                    clean(np.sin((u + v) / 2)) * c,
                ),
            ]
            for name, p, q in pairs:
                if abs(p) + abs(q) <= 1e-16:
                    continue
                candidate = self.recover(float(drive), 2 * np.arctan2(p, q))
                if candidate is not None:
                    candidate.mode_id = name
                    candidates.append(candidate)
            if (
                len(candidates) == 2
                and np.linalg.norm(wrap_angle(candidates[0].rhos - candidates[1].rhos))
                < 1e-9
            ):
                for candidate in candidates:
                    candidate.events += ("COINCIDENT_MODES",)
        else:
            for root in roots:
                candidate = self.recover(float(drive), root)
                if candidate is not None:
                    candidates.append(candidate)
            if self.geometry["collinear"] and candidates:
                # Collinearity can make C identically zero or A identically
                # zero. Factor y(A*y+B)=0 or z(B*y+C*z)=0 projectively.
                c0, cp = self.coefficients(0), self.coefficients(PI)
                internal_index = (0 - self.shift_amount) % 4
                if abs(c0[2]) + abs(cp[2]) < 2e-14:
                    for candidate in candidates:
                        null = (
                            abs(float(wrap_angle(candidate.rhos[internal_index])))
                            < 1e-9
                        )
                        candidate.mode_id = (
                            "collinear_null" if null else "collinear_other"
                        )
                elif abs(c0[0]) + abs(cp[0]) < 2e-14:
                    for candidate in candidates:
                        bound = (
                            abs(
                                abs(float(wrap_angle(candidate.rhos[internal_index])))
                                - PI
                            )
                            < 1e-9
                        )
                        candidate.mode_id = (
                            "collinear_bound" if bound else "collinear_other"
                        )
        if roots and not candidates:
            self.last_status = "NUMERICAL_FAILURE"
        if self._discriminant_roundoff:
            for candidate in candidates:
                candidate.events += ("DISCRIMINANT_ROUNDOFF",)
        self.last_details = {
            "roots": len(roots),
            "accepted": len(candidates),
            "families": len(self.last_families),
            "discriminant_roundoff": self._discriminant_roundoff,
        }
        return candidates
