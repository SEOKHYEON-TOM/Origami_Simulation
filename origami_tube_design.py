import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize

from origami_tesellator_1D import OrigamiTesellator1D
from origami_forward_kinematics import get_se3_frame
from origami_helix_inverse_design import (
    build_and_analyze,
    compute_AB,
    compute_fixed_point,
    _extract_helix_params,
)


def rotation_angle_from_matrix(R):
    """Rotation angle omega [rad] of a 3x3 rotation matrix R."""
    cos_val = (np.trace(R) - 1.0) / 2.0
    return float(np.arccos(np.clip(cos_val, -1.0, 1.0)))


def period_omega(tessellator, idx0=0, idx1=1):
    """
    Rotation angle omega [rad] of the screw motion between unit idx0 and
    unit idx1 of an already-computed tessellator (compute_strip_kinematics
    must have been run already). Returns None if either index is out of
    range of the successfully computed units.

    Frame convention mirrors origami_forward_kinematics.extract_relative_se3's
    is_base=True branch exactly (same c0/c1 vectors from CLV[0]/CLV[1], same
    tessellator.rhos[t][0] as the rho_val passed to get_se3_frame) applied
    uniformly at both idx0 and idx1, instead of re-deriving the crease-index
    mapping independently.
    """
    valid_count = len(tessellator.global_vertices)
    if idx0 < 0 or idx1 < 0 or valid_count <= max(idx0, idx1):
        return None

    def _frame_rotation(t):
        cell = tessellator.cells[t % tessellator.N]
        CLV = cell["CLV"]
        gv = tessellator.global_vertices[t]
        origin = gv[0]
        c0 = gv[CLV[0]] - origin
        c1 = gv[CLV[1]] - origin
        rho_val = tessellator.rhos[t][0]
        R, _ = get_se3_frame(origin, c0, c1, rho_val, is_base=True)
        return R

    R0 = _frame_rotation(idx0)
    R1 = _frame_rotation(idx1)
    R_rel = R0.T @ R1
    return rotation_angle_from_matrix(R_rel)
