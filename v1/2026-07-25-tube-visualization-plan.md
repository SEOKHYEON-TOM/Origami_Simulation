# Tube Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `visualize_ff_tube`/`visualize_ne_tube` to `origami_tube_design.py`
so a `find_ff_tube_design`/`find_ne_tube_design` result can be rendered as an
actual 3D N-gon tube (static or interactive), with a ring-boundary marker
line that visually confirms N-gon closure.

**Architecture:** Pure addition to the existing `origami_tube_design.py`
(see spec `2026-07-25-tube-visualization-design.md`). Three new pieces,
built bottom-up: a ring-coordinate helper, a shared internal render
function (static + interactive, reusing `OrigamiTesellator1D`'s existing
`_render_faces_and_lines` internal helper), and two thin public wrappers.
No existing file is modified.

**Tech Stack:** numpy, matplotlib (`mpl_toolkits.mplot3d.art3d.Poly3DCollection`,
`matplotlib.widgets.Slider` — both new imports for this file, already used
by `origami_tesellator_1D.py`), reusing `OrigamiTesellator1D`.

**Testing note:** Same convention as the rest of this project/file — no
pytest, "Verify" steps run inline `python - <<'EOF' ... EOF` snippets, not
saved test files.

---

### Task 1: `_ring_marker_points` helper

**Files:**
- Modify: `origami_tube_design.py` (append at end of file)

- [ ] **Step 1: Append the function**

```python
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
```

- [ ] **Step 2: Verify against a known strip**

Run:
```bash
cd "c:/python/test/Origami_Simulation_Helix"
python - <<'EOF'
import numpy as np
from origami_tesellator_1D import OrigamiTesellator1D
from origami_tube_design import _ring_marker_points

tess = OrigamiTesellator1D(
    cell_configs=[{"alphas": [70, 60, 110, 120], "sigma": 1, "iout": 2}],
    num_periods=8, verbose=False,
)
valid = tess.compute_strip_kinematics(np.deg2rad(20))
assert valid == 8, f"expected 8 valid units, got {valid}"

pts = _ring_marker_points(tess, target_N=4)
assert pts.shape == (2, 3), f"expected 2 ring points, got shape {pts.shape}"
assert np.allclose(pts[0], tess.global_vertices[0][0])
assert np.allclose(pts[1], tess.global_vertices[4][0])
print("OK: _ring_marker_points")
EOF
```
Expected: prints `OK: _ring_marker_points`, no traceback. (Verified
empirically that `alphas=[70,60,110,120]` (Kawasaki flat-foldable),
`rho0_deg=20` produces 8/8 valid units at `num_periods=8` — the original
plan draft assumed `alphas=[70,70,120,110]`/`rho0_deg=-20` would too, but
that combination only yields 2/8 valid units in practice; corrected here.)

- [ ] **Step 3: Commit**

```bash
git add origami_tube_design.py
git commit -m "Add _ring_marker_points: ring-boundary coordinates for tube visualization"
```

---

### Task 2: `_visualize_tube` (shared static + interactive renderer)

**Files:**
- Modify: `origami_tube_design.py` — add two imports near the top, append the function at the end

- [ ] **Step 1: Add the two new imports**

Find the current import block at the top of the file:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
```
Change it to:
```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import Slider
from scipy.optimize import differential_evolution, minimize
```

- [ ] **Step 2: Append the function**

```python
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
```

- [ ] **Step 3: Verify static mode, interactive mode, and the precondition check**

Run:
```bash
cd "c:/python/test/Origami_Simulation_Helix"
python - <<'EOF'
import matplotlib
matplotlib.use("Agg")
import os
from origami_tube_design import find_ff_tube_design, _visualize_tube

result = find_ff_tube_design(target_N=4, r_target=2.0, g_target=0.3, verbose=False)

# static
fig, tess = _visualize_tube(result, rho0_deg=result["rho0_deg"], num_rings=2,
                             interactive=False, save_path="_check_tube_static.png")
assert os.path.exists("_check_tube_static.png")
os.remove("_check_tube_static.png")
print("OK: _visualize_tube static")

# interactive (Agg backend makes plt.show() a no-op; this confirms the
# slider/figure construction code runs without raising)
fig2, tess2 = _visualize_tube(result, rho0_deg=result["rho0_deg"], num_rings=2,
                               interactive=True)
print("OK: _visualize_tube interactive (no crash)")

# precondition check
bad_result = dict(result)
bad_result["alphas_deg"] = None
try:
    _visualize_tube(bad_result, rho0_deg=None, num_rings=2)
    raise AssertionError("expected ValueError, none was raised")
except ValueError as e:
    print(f"OK: ValueError raised as expected: {e}")
EOF
```
Expected: prints `OK: _visualize_tube static`, `OK: _visualize_tube
interactive (no crash)`, then `OK: ValueError raised as expected: ...`, no
unhandled traceback. (`find_ff_tube_design` takes ~15-30s; this is
expected, not a hang.)

- [ ] **Step 4: Commit**

```bash
git add origami_tube_design.py
git commit -m "Add _visualize_tube: shared static/interactive N-gon tube renderer"
```

---

### Task 3: `visualize_ff_tube` / `visualize_ne_tube` public wrappers

**Files:**
- Modify: `origami_tube_design.py` (append at end of file)

- [ ] **Step 1: Append the two functions**

```python
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
```

- [ ] **Step 2: Verify both wrappers, including the interactive+save_path warning path**

Run:
```bash
cd "c:/python/test/Origami_Simulation_Helix"
python - <<'EOF'
import matplotlib
matplotlib.use("Agg")
import os
from origami_tube_design import (
    find_ff_tube_design, find_ne_tube_design,
    visualize_ff_tube, visualize_ne_tube,
)

ff_result = find_ff_tube_design(target_N=4, r_target=2.0, g_target=0.3, verbose=False)
fig, tess = visualize_ff_tube(ff_result, num_rings=2, save_path="_check_ff_tube.png")
assert os.path.exists("_check_ff_tube.png")
os.remove("_check_ff_tube.png")
print("OK: visualize_ff_tube")

ne_result = find_ne_tube_design(target_N=4, r_target=1.8, g_target=0.25, verbose=False)
fig2, tess2 = visualize_ne_tube(ne_result, num_rings=2, save_path="_check_ne_tube.png")
assert os.path.exists("_check_ne_tube.png")
os.remove("_check_ne_tube.png")
print("OK: visualize_ne_tube")

# interactive + save_path should warn and ignore save_path (not write the file)
fig3, tess3 = visualize_ff_tube(ff_result, num_rings=2, interactive=True,
                                 save_path="_should_be_ignored.png")
assert not os.path.exists("_should_be_ignored.png"), \
    "save_path should be ignored in interactive mode"
print("OK: interactive save_path ignored correctly")
EOF
```
Expected: prints `OK: visualize_ff_tube`, `OK: visualize_ne_tube`, a
`[warning] save_path is ignored when interactive=True` line, then `OK:
interactive save_path ignored correctly`, no traceback.

- [ ] **Step 3: Commit**

```bash
git add origami_tube_design.py
git commit -m "Add visualize_ff_tube/visualize_ne_tube public wrappers"
```

---

## Plan Self-Review Notes

- **Spec coverage:** Task 1 covers the spec's "Ring 경계 계산" section, Task 2
  covers "렌더링 방식" (both static and interactive branches, plus the
  precondition-failure `ValueError`), Task 3 covers the "API" section's two
  public functions including the `save_path`-ignored-in-interactive-mode
  behavior. No spec section is left uncovered.
- **No existing files modified** — all three tasks only touch
  `origami_tube_design.py` (Task 2 adds two new top-of-file imports; no
  other file changes).
- **Type/name consistency check:** `_visualize_tube`'s `(result, rho0_deg,
  num_rings, interactive, save_path)` signature is used identically by both
  `visualize_ff_tube` and `visualize_ne_tube` in Task 3, differing only in
  which `result` key is passed as `rho0_deg`. `_ring_marker_points(tess,
  target_N)` from Task 1 is called with the same two positional args inside
  `_visualize_tube`'s `_redraw` closure in Task 2 — no signature drift.
