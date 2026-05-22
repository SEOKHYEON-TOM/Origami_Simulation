import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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
        self.num_periods  = int(num_periods)
        self.total_units  = self.N * self.num_periods
        self.num_units    = self.total_units
        self.scale_factor = float(scale_factor)
        self.contact      = contact
        self.verbose      = verbose
        
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
        """ calculate rotation matrix R for rotating from u to v """
        u = u / np.linalg.norm(u)
        v = v / np.linalg.norm(v)
        c = np.cross(u, v)
        d = np.dot(u, v)
        nc = np.linalg.norm(c)
        
        if nc < 1e-9:
            return np.eye(3) if d > 0 else -np.eye(3)
        else:
            k = c / nc
            theta = np.arccos(np.clip(d, -1.0, 1.0))
            return self._rodrigues(k, theta)

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
        # Eq. A4 (iii): n^0_{n+1} = n^{iout-1}_n
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
        각 step의 rho값, branch selection 오차, geometry 이상 여부를 출력.

        Parameters
        ----------
        rho0_deg     : float  입력각 (degrees)
        stop_on_flip : bool   flip 감지 시 즉시 중단
        """
        rho_in   = np.deg2rad(rho0_deg)
        prev_out = None
        prev_gv  = None
        valid    = 0

        print(f"\n{'='*72}")
        print(f"  debug_kinematics  |  rho0={rho0_deg}°  |  "
            f"N={self.N}  periods={self.num_periods}")
        print(f"{'─'*72}")
        print(f"  {'t':>3}  {'cell':>4}  {'iout':>4}  "
            f"{'rho_in(°)':>10}  {'rho_out(°)':>11}  "
            f"{'branch_err':>10}  {'status'}")
        print(f"  {'─'*3}  {'─'*4}  {'─'*4}  "
            f"{'─'*10}  {'─'*11}  {'─'*10}  {'─'*20}")

        rho_sequence = []   # (t, rho_in_deg, rho_out_deg) 기록

        for t in range(self.total_units):
            n    = t % self.N
            cell = self.cells[n]

            solver = cell["solver"]
            A, B   = cell["A"], cell["B"]
            sigma  = cell["sigma"]
            iout   = cell["iout"]
            osi    = cell["output_solver_idx"]
            CLV    = cell["CLV"]
            prev_cell = self.cells[(t-1) % self.N] if t > 0 else None

            # ── oracle 계산 ──────────────────────────────────────────────
            if iout == 2 and A is not None:
                cos_next = A * np.cos(rho_in) + B
                if abs(cos_next) > 1.0:
                    print(f"  {t:3d}  {n:4d}  {iout:4d}  "
                        f"{np.rad2deg(rho_in):10.4f}  "
                        f"{'N/A':>11}  {'N/A':>10}  ⛔ self-blocking")
                    break
                predicted = (sigma * (1. if rho_in >= 0 else -1.)
                            * np.arccos(np.clip(cos_next, -1., 1.)))
            else:
                predicted = None

            # ── quadratic solve ──────────────────────────────────────────
            rho1_sols = solver._solve_quadratic_rho1(rho_in)
            if not rho1_sols:
                print(f"  {t:3d}  {n:4d}  {iout:4d}  "
                    f"{np.rad2deg(rho_in):10.4f}  "
                    f"{'N/A':>11}  {'N/A':>10}  ⛔ no solution")
                break

            # ── branch selection ─────────────────────────────────────────
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
                print(f"  {t:3d}  {n:4d}  ⛔ no valid candidate"); break

            candidates.sort(key=lambda x: x[0])
            best_err, out_rho, best_fs = candidates[0]

            # ── flip 감지 ────────────────────────────────────────────────
            status = "✓"
            if abs(out_rho) > np.pi + 1e-6:
                status = "⛔ |rho|>π"

            # continuity 급변 감지 (이전 대비 변화량이 클 때)
            if prev_out is not None:
                delta = abs(out_rho - prev_out)
                if delta > np.pi / 2:
                    status = f"⚠️  FLIP Δ={np.rad2deg(delta):.1f}°"

            # branch 경쟁 (두 후보의 오차 차이가 작을 때 → 불안정 구간)
            if len(candidates) >= 2:
                margin = candidates[1][0] - candidates[0][0]
                if margin < 0.05:
                    status += f"  ⚡margin={margin:.4f}"

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
            valid += 1
            prev_out = out_rho
            rho_in   = out_rho
            prev_gv  = None  # geometry는 생략 (속도)

        print(f"{'='*72}\n")
        return rho_sequence
    
    def _get_rho_for_crease(self, t, k):
        """
        unit t 의 tessellator crease k 에 해당하는 fold angle 반환.
 
        rhos[t] 배열 내 인덱스 매핑:
            Euclidean/Hyperbolic : rhos_idx = (k-1) % 4
            Elliptic             : rhos_idx = k
        """
        geom = self.cells[t % self.N]["geom"]
        idx  = k % 4 if geom == "elliptic" else (k - 1) % 4
        return self.rhos[t][idx]
 
    def get_mv_pattern(self, t, tol=1e-4):
        """
        unit t 의 M/V/F 패턴을 (e0, e1, e2, e3) 순서 문자열로 반환.
        예: "MVVM"
        """
        out = ""
        for k in range(4):
            rho = self._get_rho_for_crease(t, k)
            out += "V" if rho > tol else "M" if rho < -tol else "F"
        return out
    
        
        

# =============================================================================
#  MAIN IMPLEMENTATION
# =============================================================================
if __name__ == "__main__":

    # Flat-foldable & helical case
    cell_configs = [
        {"alphas": [95, 60, 85, 120], "sigma": 1, "iout": 2},
    ]
 
    tessellator = OrigamiTesellator1D(
        cell_configs=cell_configs,
        num_periods=8,
        lengths=(1.5, 1.0, 1.5),
        scale_factor=1.0,
    )
    
    B_middle = OrigamiTesellator1D(
    cell_configs=[
        {"alphas": [90, 90, 105, 105], "sigma": -1, "iout": 2},
        {"alphas": [80, 80,  90,  90], "sigma": -1, "iout": 2},
    ],
    num_periods=6, lengths=(1.0, 1.0, 1.0),
    )
    
    C_left = OrigamiTesellator1D(
    cell_configs=[
        {"alphas": [70, 20, 110, 160], "sigma": 1, "iout": 3},
    ],
    num_periods=15, lengths=(1.0, 1.0, 1.0),
    )
    
    C_right = OrigamiTesellator1D(
    cell_configs=[
        {"alphas": [125, 40, 65, 150], "sigma": -1, "iout": 3},
        {"alphas": [125, 40, 65, 150], "sigma":  1, "iout": 1},
    ],
    num_periods=1, lengths=(1.0, 1.0, 1.0),
)
    
    seq = C_left.debug_kinematics(rho0_deg=-10)
    #tessellator.plot_3d(rho0_deg=45)
    C_right.setup_interactive_viewer()
    C_left.export_to_mat("strip_rho45.mat", rho0_deg=-10) # due to branch sigularity flip occurs when t = 10
    #C_right.export_to_mat("strip_rho45.mat", rho0_deg=-30)