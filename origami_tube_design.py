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


def find_ff_tube_design(target_N, r_target, g_target, w_omega=3.0,
                         sigma=1, iout=2, num_periods=8,
                         lengths=(1.0, 1.0, 1.0), seed=42, verbose=True):
    """
    Search (theta0, theta1, rho0) so that the Kawasaki strip
    alphas = [theta0, theta1, 180-theta0, 180-theta1] hits target r, g and
    closes an N-gon tube cross-section (omega == 2*pi/target_N) at rho0.
    """
    target_omega = 2.0 * np.pi / target_N

    def objective(params):
        theta0, theta1, rho0 = params
        if not (3.0 < theta0 < 87.0 and 3.0 < theta1 < 87.0 and 5.0 < rho0 < 175.0):
            return 1e6
        alphas = [theta0, theta1, 180.0 - theta0, 180.0 - theta1]
        res = build_and_analyze(alphas, rho0, sigma, iout, num_periods, lengths)
        if res is None or res["helix"] is None:
            return 1e6
        omega = period_omega(res["tessellator"])
        if omega is None:
            return 1e6
        r, g = res["helix"]["r"], res["helix"]["g"]
        return ((r - r_target) ** 2 / (r_target ** 2 + 1e-4)
                + (g - g_target) ** 2
                + w_omega * (omega - target_omega) ** 2 / (target_omega ** 2 + 1e-4))

    bounds = [(5.0, 85.0), (5.0, 85.0), (5.0, 175.0)]

    best_val, best_x = np.inf, None

    # Stage 1: Differential Evolution
    try:
        res_de = differential_evolution(
            objective, bounds, seed=seed, maxiter=200, tol=1e-7,
            popsize=14, mutation=(0.5, 1.5), recombination=0.7,
        )
        if res_de.fun < best_val:
            best_val, best_x = res_de.fun, res_de.x
    except Exception as e:
        if verbose:
            print(f"  [Warning] Differential Evolution failed: {e}")

    # Stage 2: Nelder-Mead (only if DE found a valid starting point)
    if best_x is not None:
        try:
            res_nm = minimize(
                objective, best_x, method="Nelder-Mead",
                options={"xatol": 0.02, "fatol": 1e-9, "maxiter": 2000},
            )
            if res_nm.fun < best_val:
                best_val, best_x = res_nm.fun, res_nm.x
        except Exception as e:
            if verbose:
                print(f"  [Warning] Nelder-Mead failed: {e}")

    if best_x is None:
        if verbose:
            print(f"  [Error] Optimization failed completely")
        result = {
            "theta0_deg": None, "theta1_deg": None, "rho0_deg": None,
            "alphas_deg": None,
            "omega_achieved_deg": None,
            "r_achieved": None,
            "g_achieved": None,
            "residual": float(best_val),
            "tessellator": None,
            "sigma": sigma, "iout": iout, "num_periods": num_periods, "lengths": lengths,
        }
        return result

    theta0, theta1, rho0 = [float(v) for v in best_x]
    alphas_opt = [theta0, theta1, 180.0 - theta0, 180.0 - theta1]
    final = build_and_analyze(alphas_opt, rho0, sigma, iout, num_periods, lengths)
    omega_achieved = period_omega(final["tessellator"]) if final else None

    result = {
        "theta0_deg": theta0, "theta1_deg": theta1, "rho0_deg": rho0,
        "alphas_deg": alphas_opt,
        "omega_achieved_deg": np.degrees(omega_achieved) if omega_achieved is not None else None,
        "r_achieved": final["helix"]["r"] if final and final["helix"] else None,
        "g_achieved": final["helix"]["g"] if final and final["helix"] else None,
        "residual": float(best_val),
        "tessellator": final["tessellator"] if final else None,
        "sigma": sigma, "iout": iout, "num_periods": num_periods, "lengths": lengths,
    }

    if verbose:
        print(f"[FF tube] theta=({theta0:.2f}, {theta1:.2f}, "
              f"{180-theta0:.2f}, {180-theta1:.2f}) deg, rho0={rho0:.2f} deg")
        omega_str = f"{result['omega_achieved_deg']:.3f}" if result['omega_achieved_deg'] is not None else "N/A"
        r_str = f"{result['r_achieved']:.4f}" if result['r_achieved'] is not None else "N/A"
        g_str = f"{result['g_achieved']:.4f}" if result['g_achieved'] is not None else "N/A"
        print(f"  omega={omega_str} deg (target {np.degrees(target_omega):.1f} deg)")
        print(f"  r={r_str} (target {r_target}), g={g_str} (target {g_target})")
        print(f"  residual={best_val:.6f}")

    return result


def plot_ff_tube_transition(result, save_path=None, n_points=60):
    """
    Sweep rho0 across (5, 175) deg for the alphas found by
    find_ff_tube_design, plotting omega(rho0), r(rho0), g(rho0).
    Expected: g -> 0 near both ends (flat/flat-folded, zero torsion),
    while omega crosses the target value near result["rho0_deg"].
    """
    alphas = result["alphas_deg"]
    sigma = result.get("sigma", 1)
    iout = result.get("iout", 2)
    num_periods = result.get("num_periods", 8)
    lengths = result.get("lengths", (1.0, 1.0, 1.0))

    rho0_vals = np.linspace(5.0, 175.0, n_points)
    omega_vals, r_vals, g_vals = [], [], []

    for rho0 in rho0_vals:
        res = build_and_analyze(alphas, rho0, sigma, iout, num_periods, lengths)
        if res is None or res["helix"] is None:
            omega_vals.append(np.nan)
            r_vals.append(np.nan)
            g_vals.append(np.nan)
            continue
        omega = period_omega(res["tessellator"])
        omega_vals.append(np.degrees(omega) if omega is not None else np.nan)
        r_vals.append(res["helix"]["r"])
        g_vals.append(res["helix"]["g"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].plot(rho0_vals, omega_vals, 'o-', ms=3)
    axes[0].axvline(result["rho0_deg"], color='red', ls='--', label='found rho0')
    axes[0].axhline(result["omega_achieved_deg"], color='gray', ls=':', label='achieved omega')
    axes[0].set_xlabel('rho0 [deg]'); axes[0].set_ylabel('omega [deg]')
    axes[0].set_title('Period rotation angle vs rho0'); axes[0].legend(fontsize=8)

    axes[1].plot(rho0_vals, r_vals, 'o-', ms=3, color='seagreen')
    axes[1].axvline(result["rho0_deg"], color='red', ls='--')
    axes[1].set_xlabel('rho0 [deg]'); axes[1].set_ylabel('r')
    axes[1].set_title('Helix radius vs rho0')

    axes[2].plot(rho0_vals, g_vals, 'o-', ms=3, color='darkorange')
    axes[2].axvline(result["rho0_deg"], color='red', ls='--')
    axes[2].set_xlabel('rho0 [deg]'); axes[2].set_ylabel('g')
    axes[2].set_title('Helix pitch (torsion) vs rho0')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[saved] {save_path}")
    else:
        plt.show()

    return fig, {
        "rho0_vals": rho0_vals,
        "omega_vals": np.array(omega_vals),
        "r_vals": np.array(r_vals),
        "g_vals": np.array(g_vals),
    }
