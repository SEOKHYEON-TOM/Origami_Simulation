# Origami N-gon Tube Cross-Section Screw-Closure Tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a single new module, `origami_tube_design.py`, that searches for
Kawasaki (FF) and non-Kawasaki bird's-foot (NE) sector-angle designs whose
per-period screw motion closes an N-gon tube cross-section (`omega = 2*pi/N`)
while hitting target helix radius/pitch `(r, g)`, plus verification plots for
both variants.

**Architecture:** Pure-addition module that imports from the existing
`origami_tesellator_1D.py`, `origami_forward_kinematics.py`, and
`origami_helix_inverse_design.py` without modifying any of them (see spec
`2026-07-24-origami-tube-cross-section-design.md`). Six functions are added
incrementally, each one runnable/checkable as soon as it exists.

**Tech Stack:** numpy, matplotlib, scipy.optimize (differential_evolution +
Nelder-Mead), reusing `OrigamiTesellator1D`, `get_se3_frame`,
`build_and_analyze`, `compute_AB`, `compute_fixed_point`, `_extract_helix_params`.

**Testing note:** This project has no pytest suite — every existing file
verifies itself via `__main__` demo blocks and console/plot inspection (see
spec section 5). This plan follows that convention: "Verify" steps run
short inline Python snippets via `python - <<'EOF' ... EOF` (not saved to
the repo) instead of pytest test files, and assert concrete numeric
invariants wherever one is available.

---

### Task 1: Module skeleton + `rotation_angle_from_matrix`

**Files:**
- Create: `origami_tube_design.py`

- [ ] **Step 1: Create the file with imports and the first utility function**

```python
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
```

- [ ] **Step 2: Verify with a known rotation matrix**

Run:
```bash
cd "c:/python/test/Origami_Simulation_Helix"
python - <<'EOF'
import numpy as np
from origami_tube_design import rotation_angle_from_matrix

R90 = np.array([[0.0, -1.0, 0.0],
                [1.0,  0.0, 0.0],
                [0.0,  0.0, 1.0]])
angle = rotation_angle_from_matrix(R90)
assert abs(angle - np.pi / 2) < 1e-9, angle

R_identity = np.eye(3)
assert rotation_angle_from_matrix(R_identity) < 1e-9

print("OK: rotation_angle_from_matrix")
EOF
```
Expected: prints `OK: rotation_angle_from_matrix`, no traceback.

- [ ] **Step 3: Commit**

```bash
git add origami_tube_design.py
git commit -m "Add origami_tube_design.py skeleton with rotation_angle_from_matrix"
```

---

### Task 2: `period_omega`

**Files:**
- Modify: `origami_tube_design.py` (append at end of file)

- [ ] **Step 1: Append the function**

```python
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
    if valid_count <= max(idx0, idx1):
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
```

- [ ] **Step 2: Verify the fixed-point invariant**

At a strip's exact fixed point `rho*`, every cell sees the identical local
geometry forever, so the screw between *any* two consecutive units must be
the same rigid motion. This gives a self-consistency check that needs no
hand-computed expected value.

Run:
```bash
cd "c:/python/test/Origami_Simulation_Helix"
python - <<'EOF'
import numpy as np
from origami_helix_inverse_design import compute_AB, compute_fixed_point
from origami_tesellator_1D import OrigamiTesellator1D
from origami_tube_design import period_omega

alphas = [100.0, 100.0, 120.0, 140.0]
A, B = compute_AB(alphas)
cos_star, rho_star_deg = compute_fixed_point(A, B)
assert rho_star_deg is not None, "no fixed point for this test alphas"

tess = OrigamiTesellator1D(
    cell_configs=[{"alphas": alphas, "sigma": 1, "iout": 2}],
    num_periods=10, verbose=False,
)
valid = tess.compute_strip_kinematics(np.deg2rad(rho_star_deg))
assert valid >= 8, f"only {valid} valid units, expected >= 8"

omega_01 = period_omega(tess, 0, 1)
omega_56 = period_omega(tess, 5, 6)
print(f"rho*={rho_star_deg:.4f} deg, omega(0,1)={np.degrees(omega_01):.6f} deg, "
      f"omega(5,6)={np.degrees(omega_56):.6f} deg")
assert omega_01 is not None and omega_56 is not None
assert abs(omega_01 - omega_56) < 1e-6, "omega should be constant at the fixed point"

print("OK: period_omega fixed-point invariant")
EOF
```
Expected: prints the rho*/omega values then `OK: period_omega fixed-point invariant`.
If `assert rho_star_deg is not None` fails, this specific alphas example has
no real fixed point — retry with `alphas = [95.0, 95.0, 115.0, 150.0]` instead.

- [ ] **Step 3: Commit**

```bash
git add origami_tube_design.py
git commit -m "Add period_omega: per-period screw rotation angle"
```

---

### Task 3: `find_ff_tube_design`

**Files:**
- Modify: `origami_tube_design.py` (append at end of file)

- [ ] **Step 1: Append the function**

```python
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

    res_de = differential_evolution(
        objective, bounds, seed=seed, maxiter=200, tol=1e-7,
        popsize=14, mutation=(0.5, 1.5), recombination=0.7,
    )
    best_val, best_x = res_de.fun, res_de.x

    res_nm = minimize(
        objective, best_x, method="Nelder-Mead",
        options={"xatol": 0.02, "fatol": 1e-9, "maxiter": 2000},
    )
    if res_nm.fun < best_val:
        best_val, best_x = res_nm.fun, res_nm.x

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
        print(f"  omega={result['omega_achieved_deg']:.3f} deg "
              f"(target {np.degrees(target_omega):.1f} deg)")
        print(f"  r={result['r_achieved']:.4f} (target {r_target}), "
              f"g={result['g_achieved']:.4f} (target {g_target})")
        print(f"  residual={best_val:.6f}")

    return result
```

- [ ] **Step 2: Verify with a concrete target**

Run:
```bash
cd "c:/python/test/Origami_Simulation_Helix"
python - <<'EOF'
from origami_tube_design import find_ff_tube_design

result = find_ff_tube_design(target_N=4, r_target=2.0, g_target=0.3)
assert result["omega_achieved_deg"] is not None, "search produced no valid design"
omega_err = abs(result["omega_achieved_deg"] - 90.0)
print(f"residual={result['residual']:.6f}, omega_err={omega_err:.3f} deg")
assert result["residual"] < 0.3, f"residual too large: {result['residual']}"
assert omega_err < 8.0, f"omega too far from 90 deg: {omega_err}"
print("OK: find_ff_tube_design")
EOF
```
Expected: prints residual/omega_err then `OK: find_ff_tube_design`.
If either assertion fails, the (r_target=2.0, g_target=0.3) combination may
not be reachable for an N=4 closure with this alphas family — retry with
`r_target` in `1.5..3.0` and `g_target` in `0.1..0.6` (see spec section 6,
"해 존재성"). This is an optimization search, not a deterministic
computation, so a different `seed` may also help.

- [ ] **Step 3: Commit**

```bash
git add origami_tube_design.py
git commit -m "Add find_ff_tube_design: FF sector-angle search for N-gon closure"
```

---

### Task 4: `plot_ff_tube_transition`

**Files:**
- Modify: `origami_tube_design.py` (append at end of file)

- [ ] **Step 1: Append the function**

```python
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
```

- [ ] **Step 2: Verify the g -> 0 endpoint behavior**

Use the Agg backend so the check runs without opening a window.

Run:
```bash
cd "c:/python/test/Origami_Simulation_Helix"
python - <<'EOF'
import matplotlib
matplotlib.use("Agg")
import os
import numpy as np
from origami_tube_design import find_ff_tube_design, plot_ff_tube_transition

result = find_ff_tube_design(target_N=4, r_target=2.0, g_target=0.3, verbose=False)
fig, data = plot_ff_tube_transition(result, save_path="_check_ff_transition.png")
assert os.path.exists("_check_ff_transition.png")

g_vals = data["g_vals"]
finite = ~np.isnan(g_vals)
g_at_ends = np.abs(g_vals[finite])[[0, -1]]
g_at_middle = abs(g_vals[finite][len(g_vals[finite]) // 2])
print(f"|g| near ends: {g_at_ends}, |g| near middle: {g_at_middle:.4f}")
assert g_at_ends.max() < 0.05, "g should be near 0 at rho0 close to 5/175 deg"

os.remove("_check_ff_transition.png")
print("OK: plot_ff_tube_transition")
EOF
```
Expected: prints the g values then `OK: plot_ff_tube_transition`. The
middle `|g|` value is not asserted against a fixed threshold (it depends on
whatever the search converged to) — only the near-flat endpoint behavior is
checked here.

- [ ] **Step 3: Commit**

```bash
git add origami_tube_design.py
git commit -m "Add plot_ff_tube_transition: rho0 sweep verification plot"
```

---

### Task 5: `find_ne_tube_design`

**Files:**
- Modify: `origami_tube_design.py` (append at end of file)

- [ ] **Step 1: Append the function**

```python
def find_ne_tube_design(target_N, r_target, g_target, w_omega=3.0,
                         w_stable=10.0, geom_type="hyperbolic",
                         sigma=1, iout=2, num_periods=8,
                         lengths=(1.0, 1.0, 1.0), seed=42, verbose=True):
    """
    Search (alpha1(=alpha2), alpha3, alpha4) for a bird's-foot non-Euclidean
    strip whose fixed point rho* closes an N-gon tube cross-section
    (omega(rho*) == 2*pi/target_N) while hitting target r, g, preferring
    Class II (stable, |A| < 1) designs.
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

    res_de = differential_evolution(
        objective, bounds, seed=seed, maxiter=400, tol=1e-8,
        popsize=18, mutation=(0.5, 1.5), recombination=0.7,
    )
    best_val, best_x = res_de.fun, res_de.x

    res_nm = minimize(
        objective, best_x, method="Nelder-Mead",
        options={"xatol": 0.02, "fatol": 1e-9, "maxiter": 2000},
    )
    if res_nm.fun < best_val:
        best_val, best_x = res_nm.fun, res_nm.x

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
        "tessellator": final["tessellator"] if final else None,
        "sigma": sigma, "iout": iout, "num_periods": num_periods, "lengths": lengths,
    }

    if verbose:
        print(f"[NE tube] alphas=({a1:.2f}, {a1:.2f}, {a3:.2f}, {a4:.2f}) deg, "
              f"rho*={rho_star}")
        print(f"  A={A:.4f} ({'Class II stable' if abs(A) < 1 else 'Class III unstable'})")
        print(f"  omega={result['omega_achieved_deg']:.3f} deg "
              f"(target {np.degrees(target_omega):.1f} deg)")
        print(f"  r={result['r_achieved']:.4f} (target {r_target}), "
              f"g={result['g_achieved']:.4f} (target {g_target})")
        print(f"  residual={best_val:.6f}")

    return result
```

- [ ] **Step 2: Verify with a concrete target**

Run:
```bash
cd "c:/python/test/Origami_Simulation_Helix"
python - <<'EOF'
from origami_tube_design import find_ne_tube_design

result = find_ne_tube_design(target_N=4, r_target=2.0, g_target=0.3)
assert result["omega_achieved_deg"] is not None, "search produced no valid design"
omega_err = abs(result["omega_achieved_deg"] - 90.0)
print(f"residual={result['residual']:.6f}, omega_err={omega_err:.3f} deg, A={result['A']:.4f}")
assert result["residual"] < 0.5, f"residual too large: {result['residual']}"
assert omega_err < 10.0, f"omega too far from 90 deg: {omega_err}"
print("OK: find_ne_tube_design")
if abs(result["A"]) >= 1.0:
    print("NOTE: design landed in Class III (unstable) -- report this, do not treat as failure")
EOF
```
Expected: prints residual/omega_err/A then `OK: find_ne_tube_design`. If the
assertions fail, retry with different `r_target`/`g_target` in the same
ranges as Task 3, per spec section 6. Landing in Class III (`|A| >= 1`) is
not a hard failure — `w_stable` only softly prefers Class II — report it if
it happens rather than treating it as a bug.

- [ ] **Step 3: Commit**

```bash
git add origami_tube_design.py
git commit -m "Add find_ne_tube_design: NE bird's-foot search for N-gon closure"
```

---

### Task 6: `plot_ne_tube_convergence`

**Files:**
- Modify: `origami_tube_design.py` (append at end of file)

- [ ] **Step 1: Append the function**

```python
def plot_ne_tube_convergence(result, rho0_start_deg, num_cells=40, window=8,
                              save_path=None):
    """
    Build a long strip (num_cells periods) starting away from result's
    rho_star_deg, then plot per-cell omega_t (exact) and windowed local
    r_t, g_t (approximate, via _extract_helix_params on a sliding window of
    the backbone) against cell index t. Expected: all three converge toward
    result's achieved (omega, r, g) as t grows, since rho_t converges to
    rho_star along the strip for a Class II (stable) design.
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
    axes[0].axhline(result["omega_achieved_deg"], color='red', ls='--', label='target (at rho*)')
    axes[0].set_xlabel('cell index t'); axes[0].set_ylabel('omega_t [deg]')
    axes[0].set_title('Per-cell rotation angle'); axes[0].legend(fontsize=8)

    axes[1].plot(t_vals, r_vals, 'o-', ms=3, color='seagreen')
    axes[1].axhline(result["r_achieved"], color='red', ls='--')
    axes[1].set_xlabel('cell index t'); axes[1].set_ylabel('r_t (windowed)')
    axes[1].set_title('Local helix radius')

    axes[2].plot(t_vals, g_vals, 'o-', ms=3, color='darkorange')
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
```

- [ ] **Step 2: Verify the convergence trend**

Run:
```bash
cd "c:/python/test/Origami_Simulation_Helix"
python - <<'EOF'
import matplotlib
matplotlib.use("Agg")
import os
import numpy as np
from origami_tube_design import find_ne_tube_design, plot_ne_tube_convergence

result = find_ne_tube_design(target_N=4, r_target=2.0, g_target=0.3, verbose=False)
assert result["rho_star_deg"] is not None

start = result["rho_star_deg"] - 20.0
fig, data = plot_ne_tube_convergence(result, rho0_start_deg=start, num_cells=40,
                                      save_path="_check_ne_convergence.png")
assert os.path.exists("_check_ne_convergence.png")

omega_vals = data["omega_vals"]
finite = ~np.isnan(omega_vals)
target_omega = result["omega_achieved_deg"]
err = np.abs(omega_vals[finite] - target_omega)
first_half_err = err[: len(err) // 2].mean()
second_half_err = err[len(err) // 2 :].mean()
print(f"mean |omega_t - target| first half={first_half_err:.4f}, "
      f"second half={second_half_err:.4f}")
assert second_half_err <= first_half_err, "omega_t should not diverge from target over cells"

os.remove("_check_ne_convergence.png")
print("OK: plot_ne_tube_convergence")
EOF
```
Expected: prints the two mean-error values (second half less than or equal
to first half) then `OK: plot_ne_tube_convergence`. If the search in Task 5
landed in Class III (unstable, printed as a NOTE there), this convergence
check may legitimately fail since `rho_t` diverges away from `rho*` instead
of converging — that is expected behavior for an unstable design, not a bug
in this function; re-run Task 5's verification with a different `seed` to
get a Class II design first if that happens.

- [ ] **Step 3: Commit**

```bash
git add origami_tube_design.py
git commit -m "Add plot_ne_tube_convergence: per-cell convergence verification plot"
```

---

### Task 7: `__main__` demo block

**Files:**
- Modify: `origami_tube_design.py` (append at end of file)

- [ ] **Step 1: Append the demo block**

```python
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
    plot_ne_tube_convergence(
        ne_result, rho0_start_deg=ne_result["rho_star_deg"] - 20.0,
        num_cells=40, save_path="ne_tube_convergence.png",
    )
```

- [ ] **Step 2: Run the full demo end-to-end**

```bash
cd "c:/python/test/Origami_Simulation_Helix"
python origami_tube_design.py
```
Expected: both search sections print their `[FF tube]`/`[NE tube]` summary
lines and residuals, `ff_tube_transition.png` and `ne_tube_convergence.png`
are written to the project root, no traceback.

- [ ] **Step 3: Remove the generated demo images (not committed) and commit the code**

```bash
rm -f ff_tube_transition.png ne_tube_convergence.png
git add origami_tube_design.py
git commit -m "Add __main__ demo block for origami_tube_design.py"
```

---

## Plan Self-Review Notes

- **Spec coverage:** Task 1-2 cover spec 4.1 (utilities), Task 3-4 cover
  spec 4.2 (FF variant + verification plot), Task 5-6 cover spec 4.3 (NE
  variant + verification plot), Task 7 covers spec section 5 (`__main__`
  demo pattern matching existing files). Spec section 3 "제외" items
  (multi-strip 3D tube rendering, arbitrary-N auto-scan, mirror/4-cell
  period) are intentionally not tasked — out of scope per spec.
- **No existing files modified** — matches spec section 3's explicit
  constraint; every task only creates or appends to
  `origami_tube_design.py`.
- **Type/name consistency check:** `result` dict keys used consistently
  across tasks — `"alphas_deg"`, `"tessellator"`, `"sigma"`, `"iout"`,
  `"num_periods"`, `"lengths"` are produced by both `find_ff_tube_design`
  and `find_ne_tube_design` with the same names, and consumed identically
  by `plot_ff_tube_transition`/`plot_ne_tube_convergence`. `period_omega`'s
  `(idx0, idx1)` signature (Task 2) is used both at its default `(0, 1)`
  (Tasks 3-4) and explicitly at `(t-1, t)` (Task 6) — no signature drift.
