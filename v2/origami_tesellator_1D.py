import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import interp1d

from origami_simulator import OrigamiDegree4Simulator

class OrigamiTesellator1D:
    def __init__(self, cell_configs, num_periods,
                 lengths=(1.0, 1.0, 1.0),
                 scale_factor=1.0,
                 contact=True,
                 verbose=True):
        """
        N-periodic (maybe not) 1D Origami Strip tessellator.
        
        Parameters
        ----------
        cell_configs : list[dict]
            N vertices list consisting of one period.
            dict 
                necessary keys:
                "alphas"  : [a0,a1,a2,a3] [deg]
                optional (use Default value when not declared):
                "sigma"   : 1 or -1             output fold angle sign (default: 1)
                "iout"    : 1, 2, or 3          output crease index  (default: 2)
                "lengths" : (l_e1, l_e2, l_e3)  crease length         (default: global lengths)
                
        num_periods : int
        
        lengths : tuple(3)  or  list[tuple(3)]
            global crease lengths (e1, e2, e3)
            can be single tuple applied to all cells
            or N length lists applied for each cell
            
        scale_factor : float
        
        contact : bool
            Physical physical bound (-pi~pi) in local panel. ### NEED TO EXTEND FOR GLOBAL CONTACT SENSING ###

        Examples
        --------
        # Single cell
        OrigamiTesellator1D(
            cell_configs=[{"alphas":[95,60,85,120], "sigma":-1, "iout":2}],
            num_periods=8,
        )
 
        # 2-cell N-periodic strip
        OrigamiTesellator1D(
            cell_configs=[
                {"alphas":[95,60,85,120], "sigma":-1, "iout":2},
                {"alphas":[80,100,100,80], "sigma":1, "iout":1},
            ],
            num_periods=4,
            lengths=(1.5, 1.0, 1.5),
            scale_factor=1.0,
        )
        """
        # 1. Input Validation
        if not isinstance(cell_configs, (list, tuple)) or len(cell_configs) == 0:
            raise ValueError("cell_configs는 1개 이상의 dict 리스트여야 합니다.")
        for i, cfg in enumerate(cell_configs):
            if not isinstance(cfg, dict):
                raise TypeError(
                    f"cell_configs[{i}]는 dict이어야 합니다. 현재: {type(cfg).__name__}"
                )
            if "alphas" not in cfg:
                raise KeyError(f"cell_configs[{i}]에 필수 키 'alphas'가 없습니다.")
            if len(cfg["alphas"]) != 4:
                raise ValueError(
                    f"cell_configs[{i}]['alphas']는 정확히 4개여야 합니다. "
                    f"현재: {len(cfg['alphas'])}개"
                )
            sigma = cfg.get("sigma", 1)
            iout  = cfg.get("iout",  2)
            if sigma not in (1, -1):
                raise ValueError(
                    f"cell_configs[{i}]['sigma']는 1 또는 -1이어야 합니다. 현재: {sigma!r}"
                )
            if iout not in (1, 2, 3):
                raise ValueError(
                    f"cell_configs[{i}]['iout']는 1, 2, 3 중 하나여야 합니다. 현재: {iout!r}"
                )
        if int(num_periods) < 1:
            raise ValueError(f"num_periods should be over or equal to 1. Value: {num_periods!r}")
        
        # 2. Global parameter
        self.N            = len(cell_configs)
        self.cell_configs = cell_configs
        self.num_periods  = int(num_periods)
        self.total_units  = self.N * self.num_periods
        self.num_units    = self.total_units
        self.scale_factor = float(scale_factor)
        self.contact      = contact
        self.verbose      = verbose
        self.lengths      = lengths
        
        # 3. Global Lengths Regularization
        _lengths_0 = lengths[0]
        if isinstance(_lengths_0, (int, float)):
            # single tuple 
            _global_lengths = [tuple(float(v) for v in lengths)] * self.N
        else:
            # list of tuples
            if len(lengths) != self.N:
                raise ValueError(
                    f"list lengths({len(lengths)})is not identical with ({self.N})."
                )
            _global_lengths = [tuple(float(v) for v in l) for l in lengths]
                
        # 4. Cell-specific internal data generation (solver, A/B, CLV, etc.)
        self.cells = []
        for i, cfg in enumerate(cell_configs):
            # First lengths in cell_configs, if not global lengths
            cell_lengths = (
            tuple(float(v) for v in cfg["lengths"])
            if "lengths" in cfg
            else _global_lengths[i]
            )
            self.cells.append(self._init_cell(cfg, cell_lengths))
        
        # 5. Summary of cells' information
        self._print_init_summary()
        
        
        self.rhos            = np.zeros((self.total_units, 4))
        self.global_faces    = []
        self.global_vertices = []
        self.unit_iouts      = []

    # ================================================================
    #  __init__ supplementary methods
    # ================================================================
    
    def _init_cell(self, cfg, lengths):
        """
        Process single cell config dict
        
        Return
        --------
        alphas_deg, alphas      : sector angles (deg, rad)
        sigma                   : ±1
        iout                    : 1|2|3
        lengths                 : (l_e1, l_e2, l_e3)
        geom                    : "euclidean" | "elliptic" | "hyperbolic"
        roll                    : shift value for using 'np.roll' rearranging alpha angles
        solver                  : OrigamiDegree4Simulator (single unit cell solver)
        output_solver_idx       : output crease idx on solve_full lists
        CLV                     : crease k → get_3d_geometry lv idx
        A, B                    : Coefficient of linear propagation when iout = 2 (iout≠2 None)
        is_flat_foldable        : checking Kawasaki condition
        """
        
        alphas_deg = np.asarray(cfg["alphas"], dtype=float)
        alphas     = np.deg2rad(alphas_deg)
        sigma      = int(cfg.get("sigma", 1)) # if sigma not assigned, sigma = +1
        iout       = int(cfg.get("iout",  2)) # if iout not assigned, iout = 2
        
        # Geo-type classification and roll assignment
        # ─ Elliptic  (sum<2π): driving=e1 → roll= 0, θ0→α1
        # ─ Euclidean (sum=2π): driving=e4 → roll=-1, θ0→α4
        # ─ Hyperbolic(sum>2π): driving=e4 → roll=-1 (same as Euclidean)
        total_angle = float(alphas.sum())
        if np.isclose(total_angle, 2 * np.pi):
            geom, roll = "euclidean",  -1
        elif total_angle < 2 * np.pi:
            geom, roll = "elliptic",    0
        else:
            geom, roll = "hyperbolic", -1
 
        solver = OrigamiDegree4Simulator(
            np.roll(alphas_deg, roll).tolist(),
            contact=False, verbose=False,
        )
        
        # solve_full arrays → tessellator crease idex mapping
        #
        # Euclidean/Hyperbolic (roll=-1, shift_amount=0):
        #   [rho_tess-e1, rho_tess-e2, rho_tess-e3, rho_tess-e0]
        #   tess crease k → return idx (k-1)%4
        #
        # Elliptic (roll=0, shift_amount=3):
        #   [ρ_tess-e0, ρ_tess-e1, ρ_tess-e2, ρ_tess-e3]
        #   tess crease k → return k

        if geom == "elliptic":
            output_solver_idx = iout % 4
            CLV = {0: 1, 1: 2, 2: 3, 3: 4}
        else:
            output_solver_idx = (iout - 1) % 4
            CLV = {0: 4, 1: 1, 2: 2, 3: 3}
            
        # A, B coefficients (Imada 2025 Eq.1)
        # valid only when iout=2. If not, nonlinear → None
        A, B = None, None
        if iout == 2:
            t0, t1, t2, t3 = alphas
            denom = np.sin(t1) * np.sin(t2)
            if abs(denom) > 1e-12:
                A = np.sin(t3) * np.sin(t0) / denom
                B = (np.cos(t1) * np.cos(t2) - np.cos(t3) * np.cos(t0)) / denom
                
        # Kawasaki Condition (flat-foldability): θ0+θ2 = θ1+θ3 = π
        t0, t1, t2, t3 = alphas
        is_flat_foldable = (
            np.isclose(t0 + t2, np.pi, atol=1e-3)
            and np.isclose(t1 + t3, np.pi, atol=1e-3)
        )
        
        return {
            "alphas_deg":        alphas_deg,
            "alphas":            alphas,
            "sigma":             sigma,
            "iout":              iout,
            "lengths":           lengths,
            "geom":              geom,
            "roll":              roll,
            "solver":            solver,
            "output_solver_idx": output_solver_idx,
            "CLV":               CLV,
            "A":                 A,
            "B":                 B,
            "is_flat_foldable":  is_flat_foldable,
        }
        
    def _print_init_summary(self):
        if self.verbose:
        #print cell configurations summary
            _GEOM_SYM = {"euclidean": "⬡ Euc", "elliptic": "△ Ell", "hyperbolic": "▽ Hyp"}
    
            print(f"\n{'='*68}")
            print(f"  OrigamiTesellator1D  |  N={self.N} cell(s) × "
                f"{self.num_periods} period(s) = {self.total_units} vertices"
                f"  |  scale={self.scale_factor}")
            print(f"{'─'*68}")
            print(f"  {'#':>2}  {'Geom':>8}  {'iout':>4}  {'σ':>2}  "
                f"{'FF':>2}  {'Propagation':<18}  {'A':>8}  {'B':>8}")
            print(f"  {'─'*2}  {'─'*8}  {'─'*4}  {'─'*2}  "
                f"{'─'*2}  {'─'*18}  {'─'*8}  {'─'*8}")
            
            for i, cell in enumerate(self.cells):
                sym = _GEOM_SYM.get(cell["geom"], "? ???")
                ff  = "✓" if cell["is_flat_foldable"] else "✗"
    
                if cell["A"] is not None:
                    A, B = cell["A"], cell["B"]
                    if np.isclose(A, 1.0, atol=1e-2) and np.isclose(B, 0.0, atol=1e-2):
                        cls = "linear (Class I)"
                    elif np.isclose(abs(A), 1.0, atol=1e-2):
                        cls = "linear (Class IV)"
                    elif abs(A) < 1.0:
                        cls = "linear (Class II)"
                    else:
                        cls = "linear (Class III)"
                    A_str = f"{A:8.4f}"
                    B_str = f"{B:8.4f}"
                else:
                    iout = cell["iout"]
                    cls  = f"nonlinear (e{iout} adj.)"
                    A_str, B_str = "     N/A", "     N/A"
    
                print(f"  {i:2d}  {sym:>8}  {cell['iout']:4d}  {cell['sigma']:+2d}  "
                    f"{ff:>2}  {cls:<18}  {A_str}  {B_str}")
    
            print(f"{'='*68}\n")
        else:
            return
    
    def _rodrigues(self, axis, angle):
        norm = np.linalg.norm(axis)
        if norm < 1e-9:
            return np.eye(3)
        axis = axis / norm
        K = np.array([[       0, -axis[2],  axis[1]],
                      [ axis[2],        0, -axis[0]],
                      [-axis[1],  axis[0],        0]])
        return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    
    def _get_rot_from_vecs(self, u, v):
        u  = u / np.linalg.norm(u)
        v  = v / np.linalg.norm(v)
        c  = np.cross(u, v)
        d  = np.dot(u, v)
        nc = np.linalg.norm(c)

        if nc < 1e-9:
            if d > 0:
                return np.eye(3)          # parallel & same dir: identity
            else:
                # ── parallel & opposite dir → 180° proper rotation w.r.t. normal axis──
                # choose axis perpendicular to u
                perp = np.array([1., 0., 0.])
                if abs(np.dot(u, perp)) > 0.9:   # if u closer to x-axis, use y-axis instead
                    perp = np.array([0., 1., 0.])
                axis = np.cross(u, perp)
                axis = axis / np.linalg.norm(axis)
                return self._rodrigues(axis, np.pi)   # det=+1 guarantee

        return self._rodrigues(c / nc, np.arccos(np.clip(d, -1.0, 1.0)))

    def _apply_lengths_and_scale(self, local_verts, unit_index, cell_lengths, CLV):
        lv = local_verts.copy()
        O  = lv[0]
        for k in (1, 2, 3):           # tess creases e1, e2, e3
            vi  = CLV[k]
            vec = lv[vi] - O
            n   = np.linalg.norm(vec)
            if n > 1e-9:
                lv[vi] = O + (vec / n) * cell_lengths[k - 1]
        return lv * (self.scale_factor ** unit_index)
        
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

        k       = prev_iout
        k_idx   = prev_CLV[k]
        km1_idx = prev_CLV[(k - 1) % 4]
        
        # Eq. A4 (i): next origin = previous output crease end point
        origin = prev_gv[k_idx]
        
        # previous cell: c^{iout}, c^{iout-1} → n^{iout-1}
        ck    = prev_gv[k_idx]   - prev_gv[0]
        ckm1  = prev_gv[km1_idx] - prev_gv[0]
        n_prev = np.cross(ckm1, ck)
        
        # current cell: c^0, c^1 → n^0
        c0_new = local_verts[curr_CLV[0]] - local_verts[0]
        c1_new = local_verts[curr_CLV[1]] - local_verts[0]
        n_new  = np.cross(c0_new, c1_new)
 
        # Eq. A4 (ii): c^0_{n+1} = -c^{iout}_n
        R1 = self._get_rot_from_vecs(-c0_new, ck)
        # Eq. A4 (iii): n^0_{n+1} =     n^{iout-1}_n
        R2 = self._get_rot_from_vecs(R1 @ n_new, n_prev)
 
        R = R2 @ R1
        return local_verts @ R.T + origin

    # =========================================================================
    #  Main Calculation
    # =========================================================================

    def compute_strip_kinematics(self, initial_rho0):
        """
        N-periodic folding angle propagation + 3D geometry assembly
        Return the number of units that successfully calculated
        """
        self.rhos            = np.zeros((self.total_units, 4))
        self.global_faces    = []
        self.global_vertices = []
        self.unit_iouts      = []
 
        rho_in   = float(initial_rho0)
        prev_out = None # previous output folding angle (for continuity tracking)
        prev_gv  = None # previous global vertex list
        valid    = 0
        
        for t in range(self.total_units):
            n    = t % self.N
            cell = self.cells[n]
 
            solver = cell["solver"]
            A      = cell["A"]
            B      = cell["B"]
            sigma  = cell["sigma"]
            iout   = cell["iout"]
            osi    = cell["output_solver_idx"]
            CLV    = cell["CLV"]
        
            prev_cell = self.cells[(t - 1) % self.N] if t > 0 else None
            
            # ── 1. fold angle oracle ─────────────────────────────────────────
            if iout == 2 and A is not None:
                # Linear propagation (Imada Eq. 1)
                cos_next = A * np.cos(rho_in) + B
                if abs(cos_next) > 1.0:     # self-blocking
                    break
                predicted = (sigma
                             * (1.0 if rho_in >= 0 else -1.0)
                             * np.arccos(np.clip(cos_next, -1.0, 1.0)))
            else:
                predicted = None            #  Nonlinear: select branch using continuity tracking
        
            # ── 2. single-vertex solve ───────────────────────────────────────
            rho1_sols = solver._solve_quadratic_rho1(rho_in)
            if not rho1_sols:
                break
            
            # ── 3. branch selection ──────────────────────────────────────────
            best_fs = None
            min_err = float('inf')
            
            for r1 in rho1_sols:   
                fs = solver.solve_full_rhos_from_drive_and_rho1(rho_in, r1)
                if fs is None: continue
                out = fs[osi]
                
                if predicted is not None:
                    # iout=2
                    err = (abs(abs(out) - abs(predicted))
                           + (0.0 if np.sign(out) == np.sign(predicted) else 1.0))
                elif prev_out is None:
                    # first step: sigma sign standard
                    exp_sign = sigma * (1.0 if rho_in >= 0 else -1.0)
                    err = 0.0 if np.sign(out) == exp_sign else 2.0
                else:
                    err = abs(out - prev_out)
                    
                if err < min_err:
                    min_err = err
                    best_fs = fs
                    
            if best_fs is None:
                break
            
            out_rho = best_fs[osi]
            if abs(out_rho) > np.pi + 1e-6:
                break
                
            self.rhos[t] = best_fs
            valid += 1
            self.unit_iouts.append(iout)
            
            # ── 4. 3D geometry ────────────────────────────────────────────────
            _, lv_raw = solver.get_3d_geometry(self.rhos[t])
            lv = self._apply_lengths_and_scale(lv_raw, t, cell["lengths"], CLV)
 
            prev_iout = prev_cell["iout"] if prev_cell is not None else None
            prev_CLV  = prev_cell["CLV"]  if prev_cell is not None else None
 
            gv = self._place_unit_global(lv, prev_gv, CLV, prev_iout, prev_CLV)
 
            self.global_vertices.append(gv)
            prev_gv = gv
 
            # Face assembly: CLV regardless of geom
            O  = gv[0]
            e0 = gv[CLV[0]]
            e1 = gv[CLV[1]]
            e2 = gv[CLV[2]]
            e3 = gv[CLV[3]]
            self.global_faces.append([
                [O, e0, e1],    # face 0: e0~e1
                [O, e1, e2],    # face 1: e1~e2
                [O, e2, e3],    # face 2: e2~e3
                [O, e3, e0],    # face 3: e3~e0
            ])
 
            prev_out = out_rho
            rho_in   = out_rho
 
        return valid

    def validate_with_algebraic_solver(self, unit_index=0, verbose=True):
        """
        single vertex validation using spherical cosines (regardless of iout).
        rhos[t] 배열 구조: [rho_e1, rho_e2, rho_e3, rho_e0] (tessellator idx)
        """
        n = unit_index % self.N
        t0, t1, t2, t3 = self.cells[n]["alphas"]
        r_e1, r_e2, r_e3, r_e0 = self.rhos[unit_index]
 
        val1 = np.cos(t0) * np.cos(t1) - np.sin(t0) * np.sin(t1) * np.cos(r_e2)
        val2 = np.cos(t2) * np.cos(t3) - np.sin(t2) * np.sin(t3) * np.cos(r_e0)
        ok   = np.isclose(val1, val2, atol=1e-3)
 
        if verbose:
            print(f"[Unit {unit_index} / Cell {n}] Spherical check: {'PASS' if ok else 'FAIL'}")
            print(f"  rho_e0={np.rad2deg(r_e0):.2f}°  rho_e1={np.rad2deg(r_e1):.2f}°"
                  f"  rho_e2={np.rad2deg(r_e2):.2f}°  rho_e3={np.rad2deg(r_e3):.2f}°")
        return ok
    
    # =========================================================================
    #  Visualization
    # =========================================================================
 
    _CELL_PALETTES = [
        ['#CCCCCC', '#A8C4E0', '#A8D4A8', '#E0B8B0'],
        ['#F0E0A0', '#D8C878', '#E8D8A8', '#C8B880'],
        ['#D0C0F0', '#C0A8E0', '#D8B8F0', '#B0A0D8'],
        ['#A8E4E4', '#88CCCC', '#A0DCDC', '#78C4C4'],
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
            last_t    = len(self.global_vertices) - 1
            last_CLV  = self.cells[last_t % self.N]["CLV"]
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
                cx += [O[0], ep[0], np.nan] # to make each spoke as independent line
                cy += [O[1], ep[1], np.nan]
                cz += [O[2], ep[2], np.nan]
 
        return backbone, e1_pts, e3_pts, (cx, cy, cz)
    
    def _render_faces_and_lines(self, ax, valid,
                                poly_col, line_bb, line_e1, line_e3, line_cr):
        """Renew Poly3DCollection + Line3D object of current result."""
        
        # faces
        if self.global_faces and valid > 0:
            flat   = [f for unit in self.global_faces for f in unit]
            colors = []
            for t in range(valid):
                pal = self._CELL_PALETTES[t % self.N % len(self._CELL_PALETTES)]
                colors.extend(pal)
            poly_col.set_verts(flat)
            poly_col.set_facecolors(colors)
            
        # crease lines
        backbone, e1_pts, e3_pts, (cx, cy, cz) = self._build_crease_lines()
        
        def _set(line, pts):
            line.set_data(pts[:, 0], pts[:, 1])
            line.set_3d_properties(pts[:, 2])
            
        if backbone is not None:
            _set(line_bb, backbone)
            _set(line_e1, e1_pts)
            _set(line_e3, e3_pts)
            line_cr.set_data(cx, cy)
            line_cr.set_3d_properties(cz)
            
        # axis range
        rng = max(valid * 1.6, 2.0)
        ax.set_xlim(-rng, rng)
        ax.set_ylim(-rng, rng)
        ax.set_zlim(-rng, rng)
        
    def plot_3d(self, rho0_deg, figsize=(11, 8)):
        """
        Plot static strip 3D shape for given input angle.
 
        Parameters
        ----------
        rho0_deg : float   input fold angle [deg]
        figsize  : tuple
        """
        valid = self.compute_strip_kinematics(np.deg2rad(rho0_deg))
 
        fig = plt.figure(figsize=figsize)
        ax  = fig.add_subplot(111, projection='3d')
 
        if valid == 0:
            ax.set_title(f"No valid units at rho0 = {rho0_deg} deg")
            plt.show()
            return
 
        poly    = Poly3DCollection([], edgecolors='#555555', linewidths=0.5, alpha=0.85)
        ax.add_collection3d(poly)
        line_bb, = ax.plot([], [], [], color='crimson',   lw=2.5, label='Backbone')
        line_e1, = ax.plot([], [], [], color='navy',      lw=1.5, ls='--', label='e1 tips')
        line_e3, = ax.plot([], [], [], color='darkgreen', lw=1.5, ls='--', label='e3 tips')
        line_cr, = ax.plot([], [], [], color='black',     lw=0.6, alpha=0.35)
 
        self._render_faces_and_lines(ax, valid, poly,
                                     line_bb, line_e1, line_e3, line_cr)
 
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.legend(fontsize=8, loc='upper left')
 
        cell_desc = "  |  ".join(
            f"cell{i} iout={c['iout']} sigma={c['sigma']:+d}"
            for i, c in enumerate(self.cells)
        )
        ax.set_title(
            f"rho0 = {rho0_deg} deg  |  {valid}/{self.total_units} units\n{cell_desc}",
            fontsize=9
        )
        plt.tight_layout()
        plt.show()
        
    def setup_interactive_viewer(self):
        """
        Interactive 3D viewer with rho0 slider.
        """
        fig = plt.figure(figsize=(12, 8))
        ax  = fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(bottom=0.18)
 
        ax_sl  = plt.axes([0.20, 0.06, 0.60, 0.03])
        slider = Slider(ax_sl, 'rho0 (deg)', -175, 175, valinit=45)
 
        # Initial Rendering object generation (no afterimage)
        poly    = Poly3DCollection([], edgecolors='#555555', linewidths=0.5, alpha=0.85)
        ax.add_collection3d(poly)
        line_bb, = ax.plot([], [], [], color='crimson',   lw=2.5, label='Backbone')
        line_e1, = ax.plot([], [], [], color='navy',      lw=1.5, ls='--', label='e1 tips')
        line_e3, = ax.plot([], [], [], color='darkgreen', lw=1.5, ls='--', label='e3 tips')
        line_cr, = ax.plot([], [], [], color='black',     lw=0.6, alpha=0.35)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.legend(fontsize=8, loc='upper left')
        
        def update(val=None):
            rho0_deg = slider.val
            valid    = self.compute_strip_kinematics(np.deg2rad(rho0_deg))
 
            self._render_faces_and_lines(ax, valid, poly,
                                         line_bb, line_e1, line_e3, line_cr)
 
            cell_desc = "  |  ".join(
                f"cell{i} iout={c['iout']} sigma={c['sigma']:+d}"
                for i, c in enumerate(self.cells)
            )
            ax.set_title(
                f"rho0 = {rho0_deg:.1f} deg  |  {valid}/{self.total_units} units\n{cell_desc}",
                fontsize=9
            )
            fig.canvas.draw_idle()
 
        slider.on_changed(update)
        update()
        plt.show()
        
    
    
    def plot_cobweb(self, rho0_deg, ax=None, color='orange'):
        """
        Track the Cobweb along tesellations, arrow & 'Blocked' mark when angle over 180 or path is shorted
            
        Branch selection criterion (assume universal for all degree-4 vertices):
        - sgn(ρ⁰)·sgn(ρ²) == sigma
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
            
        interp_funcs = []
        colors_bg = ['steelblue', 'seagreen', 'indianred', 'mediumpurple']
        
        _GEOM_SHORT = {"euclidean": "Euc", "elliptic": "Ell", "hyperbolic": "Hyp"}
        
        for idx, config in enumerate(self.cell_configs):
            parsed = self._init_cell(config, self.lengths)
            
            sim = parsed["solver"]
            sigma = parsed["sigma"]
            iout = parsed["iout"]
            CLV = parsed["CLV"]
            mapped_out = parsed["output_solver_idx"]
            is_flat_foldable = parsed["is_flat_foldable"]
            geom = parsed["geom"]
            
            if len(sim.branch1) == 0 and len(sim.branch2) == 0:
                sim.run_simulation(resolution=1000)

            mapped_in = CLV[0] - 1
            rho2_idx  = CLV[2] - 1
            
            valid_x, valid_y = [], []
            
            for branch in [sim.branch1, sim.branch2]:
                if len(branch) == 0:
                    continue
                for row in branch:
                    x      = row[mapped_in]   # input crease angle
                    y      = row[mapped_out]  # output crease angle
                    rho2   = row[rho2_idx]    # opposite crease (branch)

                    if abs(x) < 1e-4 or abs(rho2) < 1e-4:
                        valid_x.append(x); valid_y.append(y)
                    elif np.sign(x) * np.sign(rho2) == sigma:
                        valid_x.append(x); valid_y.append(y)

            valid_x = np.array(valid_x)
            valid_y = np.array(valid_y)

            if len(valid_x) > 0:
                sort_idx = np.argsort(valid_x)
                x_sorted, y_sorted = valid_x[sort_idx], valid_y[sort_idx]
                unique_x, unique_indices = np.unique(x_sorted, return_index=True)
                unique_y = y_sorted[unique_indices]
                f_interp = interp1d(unique_x, unique_y, bounds_error=False, fill_value=np.nan)
            else:
                f_interp  = lambda x: np.nan
                unique_x  = np.array([])
                unique_y  = np.array([])

            interp_funcs.append(f_interp)

            c_bg = colors_bg[idx % len(colors_bg)]
            geom_tag = _GEOM_SHORT.get(geom, "?")
            ff_label = '✓FF' if is_flat_foldable else 'nFF'
            ax.plot(
                unique_x, unique_y,
                color=c_bg, lw=2, alpha=0.7,
                label=r'Config %d  $\sigma=%+d$  $e_{out}=%d$  [%s/%s]'
                  % (idx, sigma, iout, geom_tag, ff_label)
            )

        ax.plot([-np.pi, np.pi], [-np.pi, np.pi], 'k--', lw=1.2, alpha=0.5, label='y = x')
        
        # --- 2. Tracking Cobweb path ---
        rho_curr = np.radians(rho0_deg)
        for i in range(self.total_units):
            config_idx = i % len(self.cell_configs)
            f_interp = interp_funcs[config_idx]

            rho_next = f_interp(rho_curr)

            # [조건 검사] 운동학적 단절(NaN) 또는 물리적 한계(180도) 도달 여부
            if np.isnan(rho_next) or abs(rho_next) >= np.pi - 1e-5:
                # 넘어갔다면 플롯 범위를 위해 +-pi 로 클리핑 처리
                if not np.isnan(rho_next):
                    rho_next_clipped = np.sign(rho_next) * np.pi
                else:
                    rho_next_clipped = rho_curr  # stop when NaN
                
                ax.plot([rho_curr, rho_curr], [rho_curr, rho_next_clipped], color=color, lw=1.5, alpha=0.9)
                ax.annotate('Blocked', 
                            xy=(rho_curr, rho_next_clipped), 
                            xytext=(rho_curr + 0.2, rho_next_clipped - np.sign(rho_next_clipped) * 0.3),
                            arrowprops=dict(facecolor='red', edgecolor='red', shrink=0.0, width=1.5, headwidth=7),
                            color='red', fontsize=11, fontweight='bold', zorder=10)
                break 

            # vertical -> horizontal
            ax.plot([rho_curr, rho_curr], [rho_curr, rho_next], color=color, lw=1.5, alpha=0.9)
            ax.plot([rho_curr, rho_next], [rho_next, rho_next], color=color, lw=1.5, alpha=0.9)
            rho_curr = float(rho_next)

        # --- 축 서식 ---
        ax.set_xlabel(r'$\rho_i$ [rad]', fontsize=10)
        ax.set_ylabel(r'$\rho_{i+1}$ [rad]', fontsize=10)
        ax.set_xlim([-np.pi, np.pi])
        ax.set_ylim([-np.pi, np.pi])
        ax.set_aspect('equal')
        
        pi_ticks = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
        pi_tick_labels = [r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$']
        ax.set_xticks(pi_ticks)
        ax.set_xticklabels(pi_tick_labels)
        ax.set_yticks(pi_ticks)
        ax.set_yticklabels(pi_tick_labels)

        ax.axhline(0, color='black', linewidth=1)
        ax.axvline(0, color='black', linewidth=1)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc='best', fontsize='small')

        return ax
        
    # =========================================================================
    #  File convert
    # =========================================================================
    def export_to_mat(self, filepath, rho0_deg):
        """
        current strip geometry to MATLAB patch .mat file.
        """
        import scipy.io

        valid = self.compute_strip_kinematics(np.deg2rad(rho0_deg))
        if valid == 0:
            print(f"[WARNING] No valid unit at rho0={rho0_deg}.")
            return

        # ── vertex coordinate ─────────────────────────────────────────
        # MATLAB patch: vertices (V×3), faces (F×3) — 1-indexed
        all_verts  = []   # (x, y, z) list
        all_faces  = []   # triangular face (v1, v2, v3) list, 1-indexed

        vert_offset = 0
        for unit_faces in self.global_faces:
            for tri in unit_faces:          # tri = [v0, v1, v2] each 3D vertices
                v_start = vert_offset + 1   # MATLAB: 1-indexed
                for v in tri:
                    all_verts.append(v)
                all_faces.append([v_start, v_start+1, v_start+2])
                vert_offset += 3

        vertices = np.array(all_verts, dtype=float)  # (N, 3)
        faces    = np.array(all_faces, dtype=int)     # (M, 3)

        # ── backbone coordinate ─────────────────────────────────────────────
        result = self._build_crease_lines()
        backbone = result[0] if result[0] is not None else np.zeros((0, 3))
        e1_pts   = result[1] if result[1] is not None else np.zeros((0, 3))
        e3_pts   = result[2] if result[2] is not None else np.zeros((0, 3))

        # ── meta information ─────────────────────────────────────────────────
        meta = {
            "rho0_deg":    float(rho0_deg),
            "valid_units": float(valid),
            "total_units": float(self.total_units),
            "N_period":    float(self.N),
        }

        scipy.io.savemat(filepath, {
            "vertices": vertices,
            "faces":    faces,
            "backbone": backbone,
            "e1_pts":   e1_pts,
            "e3_pts":   e3_pts,
            "meta":     meta,
        })
        print(f"[SAVE] {filepath}  "
            f"({len(vertices)} verts, {len(faces)} faces, {valid} units)")
        
    # =========================================================================
    #  Debugging
    # =========================================================================
        
    def debug_kinematics(self, rho0_deg, stop_on_flip=True):
        """
        [Part 1] Per-cell admissible range analysis  (branch filter = same as plot_cobweb, opposite crease sign)
        [Part 2] Step-by-step kinematic trace

        Parameters
        ----------
        rho0_deg     : float  input angle (degrees)
        stop_on_flip : bool   flip 감지 시 즉시 중단
        
        Returns
        -------
        rho_sequence : list[(t, rho_in_deg, rho_out_deg)]
        """
        _GEOM_SHORT = {"euclidean": "Euc", "elliptic": "Ell", "hyperbolic": "Hyp"}
        
        # =========================================================================
        # Part 1: Per-cell Admissible Range Analysis
        # =========================================================================
        
        print(f"\n  {'─'*68}")
        print(f"  [Part 1]  Admissible Range per Cell Config")
        print(f"  {'─'*68}")

        for idx, cell in enumerate(self.cells):
            sim              = cell["solver"]
            sigma            = cell["sigma"]
            iout             = cell["iout"]
            CLV              = cell["CLV"]
            mapped_out       = cell["output_solver_idx"]
            is_flat_foldable = cell["is_flat_foldable"]
            geom             = cell["geom"]
            alphas_deg       = cell["alphas_deg"]
            alphas           = cell["alphas"]          # radians

            # simulation data 없을 때 생성
            if len(sim.branch1) == 0 and len(sim.branch2) == 0:
                sim.run_simulation(resolution=1000)

            mapped_in = CLV[0] - 1   # rho0 (input crease)
            rho2_idx  = CLV[2] - 1   # rho2 (output crease, branch selection)

            # ── Branch filter: sgn(ρ⁰)·sgn(ρ²) == σ ──────
            valid_x, valid_y = [], []
            for branch in [sim.branch1, sim.branch2]:
                if len(branch) == 0:
                    continue
                for row in branch:
                    x    = row[mapped_in]
                    y    = row[mapped_out]
                    rho2 = row[rho2_idx]
                    if abs(x) < 1e-4 or abs(rho2) < 1e-4:
                        valid_x.append(x); valid_y.append(y)
                    elif np.sign(x) * np.sign(rho2) == sigma:
                        valid_x.append(x); valid_y.append(y)

            valid_x = np.array(valid_x)
            valid_y = np.array(valid_y)

            # ── Header ────────────────────────────────────────────────────────
            geom_tag = _GEOM_SHORT.get(geom, "?")
            ff_tag   = "FF" if is_flat_foldable else "nFF"
            print(f"\n  ┌─ Cell {idx}  [{geom_tag}/{ff_tag}]  σ={sigma:+d}  iout={iout}")
            print(f"  │  alphas = {alphas_deg.astype(int).tolist()} deg")
            print(f"  │  Branch pts : branch1={len(sim.branch1)},  branch2={len(sim.branch2)}")
            print(f"  │  Valid (σ-filter) : {len(valid_x)}")

            if len(valid_x) == 0:
                print(f"  │ No admissible points")
                print(f"  └{'─'*64}")
                continue

            x_min, x_max = float(valid_x.min()), float(valid_x.max())
            y_min, y_max = float(valid_y.min()), float(valid_y.max())

            # ── Overall range ─────────────────────────────────────────────────
            print(f"  │")
            print(f"  │  ρ⁰  (input)  ∈ [{x_min:+.4f}, {x_max:+.4f}] rad"
                f"  = [{np.degrees(x_min):+7.2f}°, {np.degrees(x_max):+7.2f}°]")
            print(f"  │  ρ{iout}  (output) ∈ [{y_min:+.4f}, {y_max:+.4f}] rad"
                f"  = [{np.degrees(y_min):+7.2f}°, {np.degrees(y_max):+7.2f}°]")

            # ── Positive / Negative branch ─────────────────────────────────────
            pos_mask = valid_x >  1e-4
            neg_mask = valid_x < -1e-4
            print(f"  │")
            for mask, label in [(pos_mask, f"ρ⁰ > 0"), (neg_mask, f"ρ⁰ < 0")]:
                if mask.sum() > 0:
                    xb, yb = valid_x[mask], valid_y[mask]
                    print(f"  │  ┃ {label}  ({mask.sum():4d} pts)  "
                        f"ρ⁰ ∈ [{np.degrees(xb.min()):+6.1f}°, {np.degrees(xb.max()):+6.1f}°]  "
                        f"ρ{iout} ∈ [{np.degrees(yb.min()):+6.1f}°, {np.degrees(yb.max()):+6.1f}°]")
                else:
                    print(f"  │  ┃ {label}: ∅")

            # ── Endpoint reachability ──────────────────────────────────────────
            eps            = 0.05   # ≈ 2.9°
            reach_dev      = bool(np.any(np.abs(valid_x) < eps))
            reach_flat_in  = bool(np.any(np.abs(np.abs(valid_x) - np.pi) < eps))
            reach_flat_out = bool(np.any(np.abs(np.abs(valid_y) - np.pi) < eps))
            print(f"  │")
            print(f"  │  Endpoints (tol ≈ {np.degrees(eps):.0f}°) :"
                f"  ρ⁰=0 {'✓' if reach_dev else '✗'}"
                f"   ρ⁰=±π {'✓' if reach_flat_in else '✗'}"
                f"   ρ{iout}=±π {'✓' if reach_flat_out else '✗'}")

            # ── Configuration space gap ────────────────────────────────────────
            if len(valid_x) > 2:
                xs   = np.sort(valid_x)
                dxs  = np.diff(xs)
                maxg = float(dxs.max())
                if maxg > 0.3:
                    gi = int(dxs.argmax())
                    print(f"  │  ⚠ Disconnected: gap at ρ⁰ ∈ ({xs[gi]:+.3f}, {xs[gi+1]:+.3f})"
                        f"  width = {np.degrees(maxg):.2f}°")

            # ── Slope at ρ⁰=0  (numerical + theoretical for FF adjacent) ────────
            nz_mask = (np.abs(valid_x) < 0.3) & (np.abs(valid_x) > 1e-4)
            if nz_mask.sum() >= 2:
                xz, yz = valid_x[nz_mask], valid_y[nz_mask]
                so     = np.argsort(xz)
                slope  = float(np.polyfit(xz[so], yz[so], 1)[0])
                stab   = "stable  (|f'|<1)" if abs(slope) < 1.0 else "unstable(|f'|≥1)"
                print(f"  │")
                print(f"  │  f'(0) numerical  = {slope:+.4f}  [{stab}]")

                # Imada 2026 Eq.(2): p = -sgn(cosθ⁰+σcosθ¹)·√((A-B)/(A+B))
                if is_flat_foldable and iout in (1, 3):
                    th0, th1 = alphas[0], alphas[1]
                    A_c = np.cos(th0) * np.cos(th1) + sigma
                    B_c = np.sin(th0) * np.sin(th1)
                    ratio = (A_c - B_c) / (A_c + B_c + 1e-300)
                    if ratio >= 0 and (A_c + B_c) > 1e-9:
                        p_abs    = np.sqrt(ratio)
                        sign_fac = np.sign(np.cos(th0) + sigma * np.cos(th1))
                        p_theory = -sign_fac * p_abs
                        width    = abs(2 * np.log(np.tan(np.pi / 20))
                                    / np.log(abs(p_theory) + 1e-300))
                        print(f"  │  f'(0) theory     = {p_theory:+.4f}"
                            f"  [p, Imada2026 Eq.2  A={A_c:.4f} B={B_c:.4f}]")
                        print(f"  │  Transition width ≈ {width:.1f} cells  (10–90% sigmoid)")
                    else:
                        print(f"  │  f'(0) theory  : (A-B)/(A+B) < 0 → complex p (non-sigmoid)")

            print(f"  └{'─'*64}")

        # =========================================================================
        # Part 2: Step-by-step kinematic trace  (original logic unchanged)
        # =========================================================================
        print(f"\n  {'─'*68}")
        print(f"  [Part 2]  Step-by-step Kinematic Trace  (rho0 = {rho0_deg}°)")
        print(f"  {'─'*68}")
        print(f"  {'t':>3}  {'cell':>4}  {'iout':>4}  "
            f"{'rho_in(°)':>10}  {'rho_out(°)':>11}  "
            f"{'branch_err':>10}  {'status'}")
        print(f"  {'─'*3}  {'─'*4}  {'─'*4}  "
            f"{'─'*10}  {'─'*11}  {'─'*10}  {'─'*20}")

        rho_in       = np.deg2rad(rho0_deg)
        prev_out     = None
        rho_sequence = []

        for t in range(self.total_units):
            n    = t % self.N
            cell = self.cells[n]

            solver = cell["solver"]
            A, B   = cell["A"], cell["B"]
            sigma  = cell["sigma"]
            iout   = cell["iout"]
            osi    = cell["output_solver_idx"]

            # ── oracle ────────────────────────────────────────────────────────
            if iout == 2 and A is not None:
                cos_next = A * np.cos(rho_in) + B
                if abs(cos_next) > 1.0:
                    print(f"  {t:3d}  {n:4d}  {iout:4d}  "
                        f"{np.rad2deg(rho_in):10.4f}  "
                        f"{'N/A':>11}  {'N/A':>10}  self-blocking")
                    break
                predicted = (sigma * (1. if rho_in >= 0 else -1.)
                            * np.arccos(np.clip(cos_next, -1., 1.)))
            else:
                predicted = None

            # ── quadratic solve ───────────────────────────────────────────────
            rho1_sols = solver._solve_quadratic_rho1(rho_in)
            if not rho1_sols:
                print(f"  {t:3d}  {n:4d}  {iout:4d}  "
                    f"{np.rad2deg(rho_in):10.4f}  "
                    f"{'N/A':>11}  {'N/A':>10}  no solution")
                break

            # ── branch selection ──────────────────────────────────────────────
            candidates = []
            for r1 in rho1_sols:
                fs = solver.solve_full_rhos_from_drive_and_rho1(rho_in, r1)
                if fs is None:
                    continue
                out = fs[osi]
                if predicted is not None:
                    err = (abs(abs(out) - abs(predicted))
                        + (0. if np.sign(out) == np.sign(predicted) else 1.))
                elif prev_out is None:
                    exp_sign = sigma * (1. if rho_in >= 0 else -1.)
                    err = 0. if np.sign(out) == exp_sign else 2.
                else:
                    err = abs(out - prev_out)
                candidates.append((err, out, fs))

            if not candidates:
                print(f"  {t:3d}  {n:4d}  no valid candidate"); break

            candidates.sort(key=lambda x: x[0])
            best_err, out_rho, best_fs = candidates[0]

            # ── status flags ──────────────────────────────────────────────────
            status = "✓"
            if abs(out_rho) > np.pi + 1e-6:
                status = "|rho|>π"
            if prev_out is not None and abs(out_rho - prev_out) > np.pi / 2:
                status = f"  FLIP Δ={np.rad2deg(abs(out_rho - prev_out)):.1f}°"
            if len(candidates) >= 2:
                margin = candidates[1][0] - candidates[0][0]
                if margin < 0.05:
                    status += f"  margin={margin:.4f}"

            print(f"  {t:3d}  {n:4d}  {iout:4d}  "
                f"{np.rad2deg(rho_in):10.4f}  "
                f"{np.rad2deg(out_rho):11.4f}  "
                f"{best_err:10.5f}  {status}")

            rho_sequence.append((t, np.rad2deg(rho_in), np.rad2deg(out_rho)))

            if abs(out_rho) > np.pi + 1e-6:
                break
            if stop_on_flip and "FLIP" in status:
                print(f"\n  → FLIP 감지: t={t}에서 중단")
                break

            self.rhos[t] = best_fs
            prev_out = out_rho
            rho_in   = out_rho

        print(f"\n{'='*72}\n")
        return rho_sequence
    
    def _get_rho_for_crease(self, t, k):
        """
        At unit t, return tessellator crease k fold angle.
 
        index mapping in rhos[t]:
            Euclidean/Hyperbolic : rhos_idx = (k-1) % 4
            Elliptic             : rhos_idx = k
        """
        geom = self.cells[t % self.N]["geom"]
        idx  = k % 4 if geom == "elliptic" else (k - 1) % 4
        return self.rhos[t][idx]
 
    def get_mv_pattern(self, t, tol=1e-4):
        """
        unit t 의 M/V/F 패턴을 (e0, e1, e2, e3) 순서 문자열로 반환.
        ex: "MVVM"
        """
        out = ""
        for k in range(4):
            rho = self._get_rho_for_crease(t, k)
            out += "V" if rho > tol else "M" if rho < -tol else "F"
        return out
    
        
import scipy.io
 
 
def export_debug_placement_mat(tessellator, filepath, rho0_deg, t_flip=10):
    """
    t_flip-1 (prev) 과 t_flip (curr) 에서의 placement 핵심 벡터를
    strip 전체 geometry와 함께 .mat 파일로 내보냅니다.
 
    MATLAB quiver3 로 시각화할 수 있도록 설계됨.
 
    내보내는 구조
    ─────────────
    strip geometry:
        vertices, faces, backbone, e1_pts, e3_pts, all_origins, meta
 
    prev unit (t_flip-1, global 좌표):
        O_prev   : 원점 (1×3)
        ck       : c^{iout}      방향 벡터 (1×3)
        ckm1     : c^{iout-1}    방향 벡터 (1×3)
        n_prev   : n^{iout-1}    단위 법선 (1×3)
 
    curr unit (t_flip):
        O_curr        : 배치 후 원점     (1×3)   global
        c0_global     : c0 (배치 후)     (1×3)   global
        c1_global     : c1 (배치 후)     (1×3)   global
        n_new_global  : n0 (배치 후)     (1×3)   global, 단위벡터
 
    R1 정보 (회전 진단):
        R1_from  : _get_rot_from_vecs 의 u = -c0_new (단위벡터)
        R1_to    : _get_rot_from_vecs 의 v =  ck     (단위벡터)
        R1_axis  : cross(u, v) / |cross|
        R1_theta_deg : 회전 각도 (degrees)
    """
    valid = tessellator.compute_strip_kinematics(np.deg2rad(rho0_deg))
    if valid <= t_flip:
        raise ValueError(f"valid units({valid}) ≤ t_flip({t_flip}). "
                         f"num_periods를 늘려주세요.")
 
    t_prev = t_flip - 1
 
    # ── strip geometry ───────────────────────────────────────────────
    all_verts, all_faces = [], []
    for unit_faces in tessellator.global_faces:
        for tri in unit_faces:
            base = len(all_verts) + 1           # MATLAB 1-indexed
            for v in tri:
                all_verts.append(v)
            all_faces.append([base, base+1, base+2])
 
    vertices    = np.array(all_verts, dtype=float)
    faces       = np.array(all_faces, dtype=int)
    all_origins = np.array([gv[0] for gv in tessellator.global_vertices])
 
    res      = tessellator._build_crease_lines()
    backbone = res[0] if res[0] is not None else np.zeros((0, 3))
    e1_pts   = res[1] if res[1] is not None else np.zeros((0, 3))
    e3_pts   = res[2] if res[2] is not None else np.zeros((0, 3))
 
    # ── prev unit 벡터 (global) ──────────────────────────────────────
    prev_gv   = tessellator.global_vertices[t_prev]
    prev_cell = tessellator.cells[t_prev % tessellator.N]
    prev_iout = prev_cell["iout"]
    prev_CLV  = prev_cell["CLV"]
 
    k_idx   = prev_CLV[prev_iout]
    km1_idx = prev_CLV[(prev_iout - 1) % 4]
 
    O_prev = prev_gv[0].copy()
    ck     = prev_gv[k_idx]   - prev_gv[0]
    ckm1   = prev_gv[km1_idx] - prev_gv[0]
    n_prev = np.cross(ckm1, ck)
    if np.linalg.norm(n_prev) > 1e-9:
        n_prev = n_prev / np.linalg.norm(n_prev)
 
    # ── curr unit 벡터 (local → apply_lengths → place_unit_global) ──
    curr_cell   = tessellator.cells[t_flip % tessellator.N]
    curr_CLV    = curr_cell["CLV"]
    curr_solver = curr_cell["solver"]
 
    _, lv_raw = curr_solver.get_3d_geometry(tessellator.rhos[t_flip])
    lv = tessellator._apply_lengths_and_scale(
        lv_raw, t_flip, curr_cell["lengths"], curr_CLV
    )
 
    # local 기준 벡터 (배치 전, 원점=[0,0,0])
    c0_local = lv[curr_CLV[0]] - lv[0]
    c1_local = lv[curr_CLV[1]] - lv[0]
    n_local  = np.cross(c0_local, c1_local)
    if np.linalg.norm(n_local) > 1e-9:
        n_local = n_local / np.linalg.norm(n_local)
 
    # global 기준 벡터 (배치 후)
    curr_gv  = tessellator.global_vertices[t_flip]
    O_curr   = curr_gv[0].copy()
    c0_global = curr_gv[curr_CLV[0]] - curr_gv[0]
    c1_global = curr_gv[curr_CLV[1]] - curr_gv[0]
    n_global  = np.cross(c0_global, c1_global)
    if np.linalg.norm(n_global) > 1e-9:
        n_global = n_global / np.linalg.norm(n_global)
 
    # ── R1 진단 ─────────────────────────────────────────────────────
    # R1 = _get_rot_from_vecs(-c0_local, ck)
    u = -c0_local / (np.linalg.norm(c0_local) + 1e-300)
    v =  ck       / (np.linalg.norm(ck)       + 1e-300)
 
    cross_uv = np.cross(u, v)
    dot_uv   = float(np.dot(u, v))
    nc       = np.linalg.norm(cross_uv)
    R1_axis  = cross_uv / nc if nc > 1e-9 else np.array([0., 0., 1.])
    R1_theta = float(np.degrees(np.arccos(np.clip(dot_uv, -1., 1.))))
 
    # ── 콘솔 출력 ───────────────────────────────────────────────────
    def _fmt(v): return np.array2string(v.round(4), separator=', ')
 
    print(f"\n{'='*60}")
    print(f"  t={t_prev} (prev)  →  t={t_flip} (curr)   [rho0={rho0_deg}°]")
    print(f"{'─'*60}")
    print(f"  O_prev  = {_fmt(O_prev)}")
    print(f"  ck      = {_fmt(ck)}  ← c^{prev_iout} (output crease 방향)")
    print(f"  ckm1    = {_fmt(ckm1)}  ← c^{prev_iout-1}")
    print(f"  n_prev  = {_fmt(n_prev)}  ← n^{prev_iout-1} (단위)")
    print(f"{'─'*60}")
    print(f"  O_curr  = {_fmt(O_curr)}")
    print(f"  c0_glb  = {_fmt(c0_global)}  ← c0 (배치 후)")
    print(f"  c1_glb  = {_fmt(c1_global)}  ← c1 (배치 후)")
    print(f"  n_glb   = {_fmt(n_global)}   ← n0 (단위, 배치 후)")
    print(f"{'─'*60}")
    print(f"  R1: {_fmt(u)}  →  {_fmt(v)}")
    print(f"  R1 axis  = {_fmt(R1_axis)}")
    print(f"  R1 theta = {R1_theta:.3f}°")
    dot_n = float(np.dot(n_prev, n_global))
    print(f"  n_prev · n_global = {dot_n:.6f}  "
          f"({'✓ 일치' if dot_n > 0.999 else '⚠ 불일치!' if dot_n < -0.999 else '△ 부분 불일치'})")
    print(f"{'='*60}\n")
 
    # ── .mat 저장 ────────────────────────────────────────────────────
    scipy.io.savemat(filepath, {
        # strip geometry
        "vertices":    vertices,
        "faces":       faces,
        "backbone":    backbone,
        "e1_pts":      e1_pts,
        "e3_pts":      e3_pts,
        "all_origins": all_origins,
        "meta": {
            "rho0_deg":    float(rho0_deg),
            "valid_units": float(valid),
            "total_units": float(tessellator.total_units),
            "N_period":    float(tessellator.N),
            "t_flip":      float(t_flip),
            "t_prev":      float(t_prev),
        },
        # prev unit vectors (global)
        "O_prev":   O_prev.reshape(1, 3),
        "ck":       ck.reshape(1, 3),
        "ckm1":     ckm1.reshape(1, 3),
        "n_prev":   n_prev.reshape(1, 3),
        # curr unit vectors (global, after placement)
        "O_curr":       O_curr.reshape(1, 3),
        "c0_global":    c0_global.reshape(1, 3),
        "c1_global":    c1_global.reshape(1, 3),
        "n_new_global": n_global.reshape(1, 3),
        # R1 diagnostics
        "R1_from":      u.reshape(1, 3),
        "R1_to":        v.reshape(1, 3),
        "R1_axis":      R1_axis.reshape(1, 3),
        "R1_theta_deg": np.array([[R1_theta]]),
    })
    print(f"[저장 완료] {filepath}\n")

# =============================================================================
#  MAIN IMPLEMENTATION
# =============================================================================
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
    num_periods=2, lengths=(1.0, 1.0, 1.0),
    )
    
    flat_foldable_helix = OrigamiTesellator1D(
    cell_configs=[
        {"alphas": [70, 60, 110, 120], "sigma": 1, "iout": 2},
    ],
    num_periods=1, lengths=(0.8, 1.5, 0.8),
    )
    
    C_left.plot_cobweb(rho0_deg=-20)
    plt.title("1D Strip Tessellation Propagation")
    plt.show()
    
    seq = C_left.debug_kinematics(rho0_deg=-20)
    #C_left.plot_3d(rho0_deg=-30)
    #flat_foldable_helix.plot_3d(rho0_deg=20)
    #flat_foldable_helix.setup_interactive_viewer()
    #C_left.setup_interactive_viewer()
    #C_left.export_to_mat("strip_rho45.mat", rho0_deg=120)
    export_debug_placement_mat(C_left, "strip_debug.mat", rho0_deg=-20, t_flip=1)