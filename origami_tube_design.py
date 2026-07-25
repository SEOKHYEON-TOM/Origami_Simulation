import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import Slider
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
            "target_N": target_N, "target_omega_deg": np.degrees(target_omega),
            "r_target": r_target, "g_target": g_target,
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
        "target_N": target_N, "target_omega_deg": np.degrees(target_omega),
        "r_target": r_target, "g_target": g_target,
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
    if result.get("rho0_deg") is not None:
        axes[0].axvline(result["rho0_deg"], color='red', ls='--', label='found rho0')
    if result.get("omega_achieved_deg") is not None:
        axes[0].axhline(result["omega_achieved_deg"], color='gray', ls=':', label='achieved omega')
    if result.get("target_omega_deg") is not None:
        axes[0].axhline(result["target_omega_deg"], color='purple', ls='-.', alpha=0.7, label='target omega')
    axes[0].set_xlabel('rho0 [deg]'); axes[0].set_ylabel('omega [deg]')
    axes[0].set_title('Period rotation angle vs rho0'); axes[0].legend(fontsize=8)

    axes[1].plot(rho0_vals, r_vals, 'o-', ms=3, color='seagreen')
    if result.get("rho0_deg") is not None:
        axes[1].axvline(result["rho0_deg"], color='red', ls='--')
    axes[1].set_xlabel('rho0 [deg]'); axes[1].set_ylabel('r')
    axes[1].set_title('Helix radius vs rho0')

    axes[2].plot(rho0_vals, g_vals, 'o-', ms=3, color='darkorange')
    if result.get("rho0_deg") is not None:
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


def find_ne_tube_design(target_N, r_target, g_target, w_omega=3.0,
                         w_stable=10.0, geom_type="hyperbolic",
                         sigma=1, iout=2, num_periods=8,
                         lengths=(1.0, 1.0, 1.0), seed=42, verbose=True):
    """
    Search (alpha1(=alpha2), alpha3, alpha4) for a bird's-foot non-Euclidean
    strip whose fixed point rho* closes an N-gon tube cross-section
    (omega(rho*) == 2*pi/target_N) while hitting target r, g, preferring
    Class II (stable, |A| < 1) designs.

    Note: empirically, the alpha1=alpha2 (bird's-foot) symmetry constrains
    the converged fixed-point helix to g_achieved =~ 0 (observed magnitude
    1e-16 to 1e-7 across many configurations) -- g_target is effectively
    unreachable for nonzero values with this parametrization. Treat
    `residual` here as primarily an omega-closure + r-fit + stability metric,
    not a genuine (r, g) fit. The default targets from the plan (r_target=2.0,
    g_target=0.3) sit near the edge of what this family can achieve; a nearby
    target (e.g. r_target=1.8, g_target=0.25) may be needed in practice.
    """
    target_omega = 2.0 * np.pi / target_N

    def objective(params):
        a1, a3, a4 = params
        total = 2 * a1 + a3 + a4
        if geom_type == "hyperbolic" and total <= 360.5:
            return 1e6 + 100 * (360.5 - total) ** 2
        if geom_type == "elliptic" and total >= 359.5:
            return 1e6 + 100 * (total - 359.5) ** 2
        for a in (a1, a3, a4):
            if not (5.0 < a < 175.0):
                return 1e6

        alphas = [a1, a1, a3, a4]
        A, B = compute_AB(alphas)
        if A is None:
            return 1e6
        cos_star, rho_star = compute_fixed_point(A, B)
        if rho_star is None:
            return 1e6 + 1000.0

        res = build_and_analyze(alphas, rho_star, sigma, iout, num_periods, lengths)
        if res is None or res["helix"] is None:
            return 1e6
        omega = period_omega(res["tessellator"])
        if omega is None:
            return 1e6

        r, g = res["helix"]["r"], res["helix"]["g"]
        return ((r - r_target) ** 2 / (r_target ** 2 + 1e-4)
                + (g - g_target) ** 2
                + w_omega * (omega - target_omega) ** 2 / (target_omega ** 2 + 1e-4)
                + w_stable * max(0.0, abs(A) - 1.0) ** 2)

    if geom_type == "hyperbolic":
        bounds = [(91.0, 165.0), (91.0, 165.0), (91.0, 165.0)]
    else:
        bounds = [(10.0, 85.0), (10.0, 85.0), (10.0, 85.0)]

    best_val, best_x = np.inf, None

    # Stage 1: Differential Evolution
    try:
        res_de = differential_evolution(
            objective, bounds, seed=seed, maxiter=400, tol=1e-8,
            popsize=18, mutation=(0.5, 1.5), recombination=0.7,
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
            "alpha1_deg": None, "alpha3_deg": None, "alpha4_deg": None,
            "alphas_deg": None, "rho_star_deg": None,
            "A": None, "B": None,
            "omega_achieved_deg": None,
            "r_achieved": None,
            "g_achieved": None,
            "residual": float(best_val),
            "target_N": target_N, "target_omega_deg": np.degrees(target_omega),
            "r_target": r_target, "g_target": g_target,
            "tessellator": None,
            "sigma": sigma, "iout": iout, "num_periods": num_periods, "lengths": lengths,
        }
        return result

    a1, a3, a4 = [float(v) for v in best_x]
    alphas_opt = [a1, a1, a3, a4]
    A, B = compute_AB(alphas_opt)
    cos_star, rho_star = compute_fixed_point(A, B)
    final = (build_and_analyze(alphas_opt, rho_star, sigma, iout, num_periods, lengths)
             if rho_star is not None else None)
    omega_achieved = period_omega(final["tessellator"]) if final else None

    result = {
        "alpha1_deg": a1, "alpha3_deg": a3, "alpha4_deg": a4,
        "alphas_deg": alphas_opt, "rho_star_deg": rho_star,
        "A": A, "B": B,
        "omega_achieved_deg": np.degrees(omega_achieved) if omega_achieved is not None else None,
        "r_achieved": final["helix"]["r"] if final and final["helix"] else None,
        "g_achieved": final["helix"]["g"] if final and final["helix"] else None,
        "residual": float(best_val),
        "target_N": target_N, "target_omega_deg": np.degrees(target_omega),
        "r_target": r_target, "g_target": g_target,
        "tessellator": final["tessellator"] if final else None,
        "sigma": sigma, "iout": iout, "num_periods": num_periods, "lengths": lengths,
    }

    if verbose:
        print(f"[NE tube] alphas=({a1:.2f}, {a1:.2f}, {a3:.2f}, {a4:.2f}) deg, "
              f"rho*={rho_star}")
        print(f"  A={A:.4f} ({'Class II stable' if abs(A) < 1 else 'Class III unstable'})")
        omega_str = f"{result['omega_achieved_deg']:.3f}" if result['omega_achieved_deg'] is not None else "N/A"
        r_str = f"{result['r_achieved']:.4f}" if result['r_achieved'] is not None else "N/A"
        g_str = f"{result['g_achieved']:.4f}" if result['g_achieved'] is not None else "N/A"
        print(f"  omega={omega_str} deg (target {np.degrees(target_omega):.1f} deg)")
        print(f"  r={r_str} (target {r_target}), g={g_str} (target {g_target})")
        print(f"  residual={best_val:.6f}")

    return result


def plot_ne_tube_convergence(result, rho0_start_deg, num_cells=40, window=8,
                              save_path=None):
    """
    Build a long strip (num_cells periods) starting away from result's
    rho_star_deg, then plot per-cell omega_t (exact) and windowed local
    r_t, g_t (approximate, via _extract_helix_params on a sliding window of
    the backbone) against cell index t. Expected: all three converge toward
    result's achieved (omega, r, g) as t grows, since rho_t converges to
    rho_star along the strip for a Class II (stable) design.

    Note: for bird's-foot (alpha1=alpha2) NE designs, g_achieved is
    typically ~= 0 regardless of the original g_target (see
    find_ne_tube_design's docstring for details) -- the g_t panel here
    will usually converge toward ~0, not toward any nonzero target.
    """
    alphas = result["alphas_deg"]
    sigma = result.get("sigma", 1)
    iout = result.get("iout", 2)
    lengths = result.get("lengths", (1.0, 1.0, 1.0))

    tess = OrigamiTesellator1D(
        cell_configs=[{"alphas": alphas, "sigma": sigma, "iout": iout, "lengths": lengths}],
        num_periods=num_cells, verbose=False,
    )
    valid = tess.compute_strip_kinematics(np.deg2rad(rho0_start_deg))
    backbone = tess._build_crease_lines()[0]

    t_vals, omega_vals, r_vals, g_vals = [], [], [], []
    for t in range(1, valid):
        omega = period_omega(tess, t - 1, t)
        t_vals.append(t)
        omega_vals.append(np.degrees(omega) if omega is not None else np.nan)

        lo = max(0, t - window)
        window_pts = backbone[lo:t + 1]
        helix = _extract_helix_params(window_pts) if len(window_pts) >= 4 else None
        r_vals.append(helix["r"] if helix else np.nan)
        g_vals.append(helix["g"] if helix else np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].plot(t_vals, omega_vals, 'o-', ms=3)
    if result.get("omega_achieved_deg") is not None:
        axes[0].axhline(result["omega_achieved_deg"], color='red', ls='--', label='achieved (at rho*)')
    if result.get("target_omega_deg") is not None:
        axes[0].axhline(result["target_omega_deg"], color='purple', ls='-.', alpha=0.7, label='target omega')
    axes[0].set_xlabel('cell index t'); axes[0].set_ylabel('omega_t [deg]')
    axes[0].set_title('Per-cell rotation angle'); axes[0].legend(fontsize=8)

    axes[1].plot(t_vals, r_vals, 'o-', ms=3, color='seagreen')
    if result.get("r_achieved") is not None:
        axes[1].axhline(result["r_achieved"], color='red', ls='--')
    axes[1].set_xlabel('cell index t'); axes[1].set_ylabel('r_t (windowed)')
    axes[1].set_title('Local helix radius')

    axes[2].plot(t_vals, g_vals, 'o-', ms=3, color='darkorange')
    if result.get("g_achieved") is not None:
        axes[2].axhline(result["g_achieved"], color='red', ls='--')
    axes[2].set_xlabel('cell index t'); axes[2].set_ylabel('g_t (windowed)')
    axes[2].set_title('Local helix pitch')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[saved] {save_path}")
    else:
        plt.show()

    return fig, {
        "t_vals": np.array(t_vals),
        "omega_vals": np.array(omega_vals),
        "r_vals": np.array(r_vals),
        "g_vals": np.array(g_vals),
    }


def _ring_marker_points(tess, target_N):
    """
    Origin coordinates of every target_N-th unit (ring boundaries).

    If the strip's per-period screw rotation is exactly 2*pi/target_N, these
    points trace one smooth helix/circle with uniform spacing -- the visual
    signature of a genuinely closed N-gon tube cross-section.
    """
    origins = [tess.global_vertices[t][0]
               for t in range(0, len(tess.global_vertices), target_N)]
    return np.array(origins)


def _visualize_tube(result, rho0_deg, num_rings=10, interactive=False, save_path=None):
    """
    Render the N-gon tube for a find_ff_tube_design/find_ne_tube_design
    result as an actual 3D OrigamiTesellator1D strip, with a ring-boundary
    marker line (every target_N units) overlaid to show N-gon closure.

    rho0_deg: the fold state to render (caller picks rho0_deg vs
    rho_star_deg depending on which search produced `result`).
    """
    if result.get("alphas_deg") is None or rho0_deg is None:
        raise ValueError(
            "Cannot visualize: this result has no valid design "
            "(alphas_deg or the fold-state field is None -- likely a "
            "total search failure from find_ff_tube_design/find_ne_tube_design)."
        )

    alphas = result["alphas_deg"]
    sigma = result.get("sigma", 1)
    iout = result.get("iout", 2)
    lengths = result.get("lengths", (1.0, 1.0, 1.0))
    target_N = result["target_N"]

    tess = OrigamiTesellator1D(
        cell_configs=[{"alphas": alphas, "sigma": sigma, "iout": iout, "lengths": lengths}],
        num_periods=target_N * num_rings, verbose=False,
    )

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection='3d')

    poly = Poly3DCollection([], edgecolors='#555555', linewidths=0.5, alpha=0.85)
    ax.add_collection3d(poly)
    line_bb, = ax.plot([], [], [], color='crimson', lw=2.5, label='Backbone')
    line_e1, = ax.plot([], [], [], color='navy', lw=1.5, ls='--', label='e1 tips')
    line_e3, = ax.plot([], [], [], color='darkgreen', lw=1.5, ls='--', label='e3 tips')
    line_cr, = ax.plot([], [], [], color='black', lw=0.6, alpha=0.35)
    line_ring, = ax.plot([], [], [], color='magenta', lw=2.5, marker='o', ms=5,
                         label=f'ring boundary (every {target_N} units)')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend(fontsize=8, loc='upper left')

    def _title(rho0_val, valid):
        return (f"N={target_N} tube  |  rho0={rho0_val:.1f} deg  |  "
                f"{valid}/{tess.total_units} units  |  "
                f"omega={result.get('omega_achieved_deg')}, "
                f"r={result.get('r_achieved')}, g={result.get('g_achieved')}")

    def _redraw(rho0_val):
        valid = tess.compute_strip_kinematics(np.deg2rad(rho0_val))
        tess._render_faces_and_lines(ax, valid, poly, line_bb, line_e1, line_e3, line_cr)
        ring_pts = _ring_marker_points(tess, target_N)
        if len(ring_pts) > 0:
            line_ring.set_data(ring_pts[:, 0], ring_pts[:, 1])
            line_ring.set_3d_properties(ring_pts[:, 2])
        ax.set_title(_title(rho0_val, valid), fontsize=8)
        return valid

    if interactive:
        if save_path:
            print("[warning] save_path is ignored when interactive=True")
        plt.subplots_adjust(bottom=0.18)
        ax_sl = plt.axes([0.20, 0.06, 0.60, 0.03])
        slider = Slider(ax_sl, 'rho0 (deg)', -175, 175, valinit=rho0_deg)

        def _update(val=None):
            _redraw(slider.val)
            fig.canvas.draw_idle()

        slider.on_changed(_update)
        _update()
        plt.show()
    else:
        _redraw(rho0_deg)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[saved] {save_path}")
        else:
            plt.show()

    return fig, tess


def visualize_ff_tube(result, num_rings=10, interactive=False, save_path=None):
    """
    Render the N-gon tube found by find_ff_tube_design, at its found rho0.
    See _visualize_tube for parameter details.
    """
    return _visualize_tube(result, rho0_deg=result.get("rho0_deg"),
                            num_rings=num_rings, interactive=interactive,
                            save_path=save_path)


def visualize_ne_tube(result, num_rings=10, interactive=False, save_path=None):
    """
    Render the N-gon tube found by find_ne_tube_design, at its fixed point
    rho_star_deg. See _visualize_tube for parameter details.

    Note: for bird's-foot (alpha1=alpha2) designs, g_achieved is typically
    ~= 0 (see find_ne_tube_design's docstring) -- the rendered tube will
    usually look like a flat stack of rings, not a pitched helix.
    """
    return _visualize_tube(result, rho0_deg=result.get("rho_star_deg"),
                            num_rings=num_rings, interactive=interactive,
                            save_path=save_path)


if __name__ == "__main__":
    TARGET_N = 4
    R_TARGET = 2.0
    G_TARGET = 0.3

    print("=" * 60)
    print("FF tube design search")
    print("=" * 60)
    ff_result = find_ff_tube_design(TARGET_N, R_TARGET, G_TARGET)
    plot_ff_tube_transition(ff_result, save_path="ff_tube_transition.png")

    print("=" * 60)
    print("NE tube design search")
    print("=" * 60)
    ne_result = find_ne_tube_design(TARGET_N, R_TARGET, G_TARGET)
    if ne_result["rho_star_deg"] is not None:
        plot_ne_tube_convergence(
            ne_result, rho0_start_deg=ne_result["rho_star_deg"] - 20.0,
            num_cells=40, save_path="ne_tube_convergence.png",
        )
    else:
        print("[skip] NE search did not converge to a valid fixed point; skipping convergence plot")
