import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, CheckButtons
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class OrigamiDegree4Simulator:
    def __init__(self, alphas_deg, contact=True, verbose=True):
        """
        alphas_deg: [alpha1, alpha2, alpha3, alpha4] [deg]
        contact: physical bound (-pi~pi) between panels 
        """
        self.original_alphas_deg = alphas_deg
        self.original_alphas = np.deg2rad(self.original_alphas_deg)
        self.contact = contact
        self.verbose = verbose
       
        sum_alpha = np.sum(self.original_alphas)
        self.is_euclidean = np.isclose(sum_alpha, 2*np.pi)

        a1, a2, a3, a4 = self.original_alphas
        self.is_flat_foldable = self.is_euclidean and np.isclose(a1 + a3, np.pi, atol=1e-5)
        
        if self.is_euclidean:
            self.driving_crease = 4
            self.geom_type = "Euclidean (Flat)"
        elif sum_alpha < 2*np.pi:
            self.driving_crease = 1
            self.geom_type = "Elliptic (Cone-like)"
        else:
            self.driving_crease = 4
            self.geom_type = "Hyperbolic (Saddle-like)"
            
        self.shift_amount = 4 - self.driving_crease
        self.alphas = np.roll(self.original_alphas, self.shift_amount)
        
        self.full_math_data = [] # total solution space (Faded lines)
        self.branch1 = []        # path 1 (M/V pattern 1)
        self.branch2 = []        # path 2 (M/V pattern 2)
        
        if self.verbose:
            print(f"[{self.geom_type}] Sum: {np.rad2deg(sum_alpha):.2f} deg / Driving Crease: e{self.driving_crease}")
        
        # Hard coding parameters
        self.TOLERANCE = 1e-9 # treated as zero 
        self.SINGULARITY_THRESHOLD = 1e-15 # wanna avoid

    # --- [Core Solver] ---
    def _solve_quadratic_rho1(self, rho_drive):
        """Given rho 4, find rho 1 solution using eq6 (Appendix A6) in Foschi(2022)."""
        a1, a2, a3, a4 = self.alphas
        x = np.tan(rho_drive / 2)
        # define coefficients of Ay^2 + By + C = 0 (y = tan(rho1/2))
        A = x**2 * np.cos(a1+a3-a4) + np.cos(a1-a3-a4) - (1+x**2)*np.cos(a2)
        B = 4 * x * np.sin(a1) * np.sin(a3)
        C = np.cos(a1+a3+a4) + x**2 * np.cos(a1-a3+a4) - (1+x**2)*np.cos(a2)
        
        if np.abs(A) < self.TOLERANCE:
            if np.abs(B) < self.TOLERANCE: return []
            return [2 * np.arctan(-C / B)]
            
        discriminant = B**2 - 4*A*C
        if discriminant < -self.TOLERANCE: return [] # Kinematically infeasible
        discriminant = max(0, discriminant)
        y1 = (-B + np.sqrt(discriminant)) / (2*A)
        y2 = (-B - np.sqrt(discriminant)) / (2*A)
        return [2 * np.arctan(y1), 2 * np.arctan(y2)]

    def solve_full_rhos_from_drive_and_rho1(self, rho_drive, rho1):
        """Recover all four folding angles from the driving angle and rho1."""

        a1, a2, a3, a4 = self.alphas
        x4, x1 = np.tan(rho_drive / 2), np.tan(rho1 / 2)

        N2 = (1+x4**2)*np.cos(a1+a2) - x4**2*np.cos(a3-a4) - np.cos(a3+a4)
        D2 = (1+x4**2)*np.cos(a1-a2) - x4**2*np.cos(a3-a4) - np.cos(a3+a4)
        N3 = (1+x1**2)*np.cos(a2+a3) - x1**2*np.cos(a4-a1) - np.cos(a4+a1)
        D3 = (1+x1**2)*np.cos(a2-a3) - x1**2*np.cos(a4-a1) - np.cos(a4+a1)

        if abs(D2) < self.SINGULARITY_THRESHOLD or abs(D3) < self.SINGULARITY_THRESHOLD:
            return None

        x2_sq, x3_sq = -N2 / D2, -N3 / D3
        if x2_sq < -1e-5 or x3_sq < -1e-5:
            return None

        mag_x2, mag_x3 = np.sqrt(max(0, x2_sq)), np.sqrt(max(0, x3_sq))

        LHS34 = np.cos(a1) * (1+x3_sq) * (1+x4**2)
        T1_34 = np.cos(a4-a3-a2) * x4**2
        T2_34 = np.cos(a4+a3-a2) * x3_sq
        T3_34 = np.cos(a4-a3+a2) * x3_sq * x4**2
        T4_34 = np.cos(a4+a3+a2)
        P34 = (LHS34 - (T1_34 + T2_34 + T3_34 + T4_34)) / (4 * np.sin(a4) * np.sin(a2))

        sign_x3 = np.sign(P34 * x4) if abs(x4) > self.TOLERANCE else np.sign(P34)
        if sign_x3 == 0:
            sign_x3 = 1
        x3 = sign_x3 * mag_x3
        rho3 = 2 * np.arctan(x3)

        LHS12 = np.cos(a3) * (1+x1**2) * (1+x2_sq)
        T1_12 = np.cos(a2-a1-a4) * x2_sq
        T2_12 = np.cos(a2+a1-a4) * x1**2
        T3_12 = np.cos(a2-a1+a4) * x1**2 * x2_sq
        T4_12 = np.cos(a2+a1+a4)
        P12 = (LHS12 - (T1_12 + T2_12 + T3_12 + T4_12)) / (4 * np.sin(a2) * np.sin(a4))

        if abs(x1) > self.TOLERANCE:
            sign_x2 = np.sign(P12 * x1)
        else:
            LHS23 = np.cos(a4) * (1+x2_sq) * (1+x3_sq)
            T1_23 = np.cos(a3-a2-a1) * x3_sq
            T2_23 = np.cos(a3+a2-a1) * x2_sq
            T3_23 = np.cos(a3-a2+a1) * x2_sq * x3_sq
            T4_23 = np.cos(a3+a2+a1)
            P23 = (LHS23 - (T1_23 + T2_23 + T3_23 + T4_23)) / (4 * np.sin(a3) * np.sin(a1))
            sign_x2 = np.sign(P23 * x3) 

        if sign_x2 == 0:
            sign_x2 = 1
        x2 = sign_x2 * mag_x2
        rho2 = 2 * np.arctan(x2)

        return np.roll([rho1, rho2, rho3, rho_drive], -self.shift_amount)

    def _unwrap_filter(self, path, step_size):
        """Unwrap engine: cutoff the path over folding angle pi"""
        if len(path) == 0: return np.empty((0, 4))
        path = np.array(path)
        
        d_idx = self.driving_crease - 1
        diffs = np.abs(np.diff(path[:, d_idx]))
        split_indices = np.where(diffs > 2.0 * step_size)[0] + 1
        blocks = np.split(path, split_indices)
        
        valid_blocks = []
        for blk in blocks:
            if len(blk) == 0: continue
            unwrapped = np.unwrap(blk, axis=0)
            anchor_idx = np.argmin(np.sum(np.abs(blk), axis=1))
            for col in range(4):
                shift = np.round(unwrapped[anchor_idx, col] / (2*np.pi)) * (2*np.pi)
                unwrapped[:, col] -= shift
            valid_mask = np.all(np.abs(unwrapped) <= np.pi + 1e-4, axis=1)
            if np.any(valid_mask): valid_blocks.append(blk[valid_mask])
            
        return np.vstack(valid_blocks) if valid_blocks else np.empty((0, 4))

    def get_mv_pattern(self, rhos):
        """Get rhos, change into M/V/F string (ex 'MVVV') for Euclidean case"""
        return "".join(["V" if r > 1e-4 else "M" if r < -1e-4 else "F" for r in rhos])
    
    # --- [Computation Methods] ---
    def run_simulation(self, resolution=1000):
        """Path calculation & find physical path"""
        a1, a2, a3, a4 = self.alphas
        rho_drive_range = np.linspace(-np.pi + 1e-4, np.pi - 1e-4, resolution)
        step_size = (2 * np.pi) / resolution
        
        raw_m1, raw_m2 = [], []
        prev_r1_m1, prev_r1_m2 = None, None
        
        for rho_drive in rho_drive_range:
            rho1_sols = self._solve_quadratic_rho1(rho_drive)
            if not rho1_sols:
                prev_r1_m1, prev_r1_m2 = None, None
                continue
                
            # Nearest Neighbor Tracking
            if len(rho1_sols) == 2:
                if prev_r1_m1 is None: r1_m1, r1_m2 = rho1_sols[0], rho1_sols[1]
                else:
                    d_stay = abs(rho1_sols[0]-prev_r1_m1) + abs(rho1_sols[1]-prev_r1_m2)
                    d_swap = abs(rho1_sols[1]-prev_r1_m1) + abs(rho1_sols[0]-prev_r1_m2)
                    r1_m1, r1_m2 = (rho1_sols[0], rho1_sols[1]) if d_stay <= d_swap else (rho1_sols[1], rho1_sols[0])
                prev_r1_m1, prev_r1_m2 = r1_m1, r1_m2
                tracked = [(0, r1_m1), (1, r1_m2)]
            else:
                tracked = [(0, rho1_sols[0])]
                prev_r1_m1 = rho1_sols[0]
                
            for m_idx, r1 in tracked:
                res_rhos = self.solve_full_rhos_from_drive_and_rho1(rho_drive, r1)
                if res_rhos is None:
                    continue
                if m_idx == 0: raw_m1.append(res_rhos)
                else: raw_m2.append(res_rhos)
        
        # Kinematic path without contact  
        self.full_math_data = np.vstack([raw_m1, raw_m2]) if raw_m1 or raw_m2 else []
        d_idx = self.driving_crease - 1
        
        # ---Euclidean case: M/V pattern classification (two branch clustering)---
        if self.is_euclidean:
            all_rhos = raw_m1 + raw_m2
            branch_dict = {}
            leftovers = []
            
            for rhos in all_rhos:
                pat = self.get_mv_pattern(rhos)
                if 'F' in pat: continue 
                
                # Find minority index with different M/V pattern'
                m_count, v_count = pat.count('M'), pat.count('V')
                if m_count == 1:
                    idx = pat.find('M')
                    if idx not in branch_dict: branch_dict[idx] = []
                    branch_dict[idx].append(rhos)
                elif v_count == 1:
                    idx = pat.find('V')
                    if idx not in branch_dict: branch_dict[idx] = []
                    branch_dict[idx].append(rhos)
                else:
                    leftovers.append(rhos)
                
            # Extract two branch data with the most collected data
            sorted_patterns = sorted(branch_dict.keys(), key=lambda k: len(branch_dict[k]), reverse=True)
            raw_m1 = branch_dict[sorted_patterns[0]] if len(sorted_patterns) >= 1 else []
            raw_m2 = branch_dict[sorted_patterns[1]] if len(sorted_patterns) >= 2 else []
            
            # Collect singular cases
            self.singular_states = sorted(leftovers, key=lambda x: x[d_idx]) if leftovers else []
            
            # Console print
            if len(sorted_patterns) >= 1:
                sample_pat_1 = self.get_mv_pattern(raw_m1[len(raw_m1)//2])
                print(f"  -> Branch 1 assigned to Minority Index [{sorted_patterns[0]}] (e.g. {sample_pat_1})")
            if len(sorted_patterns) >= 2:
                sample_pat_2 = self.get_mv_pattern(raw_m2[len(raw_m2)//2])
                print(f"  -> Branch 2 assigned to Minority Index [{sorted_patterns[1]}] (e.g. {sample_pat_2})")
            if len(self.singular_states) > 0:
                print(f"  -> Separated {len(self.singular_states)} transition/singular states (stored in self.singular_states)")
            
        # --- Data save & unwarp ---
        # Align original data so that lines do not get twisted while drawing
        
        if len(raw_m1) > 0: raw_m1 = sorted(raw_m1, key=lambda x: x[d_idx])
        if len(raw_m2) > 0: raw_m2 = sorted(raw_m2, key=lambda x: x[d_idx])
        
        self.branch1 = self._unwrap_filter(raw_m1, step_size)
        self.branch2 = self._unwrap_filter(raw_m2, step_size)
        
        self.math_branch1 = np.array(raw_m1) if len(raw_m1) > 0 else [] # for 3d interactive
        self.math_branch2 = np.array(raw_m2) if len(raw_m2) > 0 else []
        
    # --- [Visualization Methods] ---
    def plot_2d_static(self):
        """Static configuration plot (FIG.3. in paper, x axis - rho1)"""
        fig, ax = plt.subplots(figsize=(8,8))
        
        d_idx = self.driving_crease - 1
        other_indices = [i for i in range(4) if i != d_idx]
        labels = [r'$\rho_2$', r'$\rho_3$', r'$\rho_4$']
        colors = ['blue', 'green', 'red'] # (rho2: ?뚮옉, rho3: 珥덈줉, rho4: 鍮④컯)
        
        if self.contact:
            # --- [Case 1] Contact = True ---
            if len(self.full_math_data) > 0:
                for j in range(3):
                    ax.scatter(self.full_math_data[:,0], self.full_math_data[:,j+1], color=colors[j], s=2, alpha = 0.05) # alpha < 1 to make it blurry
                    
            for data in [self.branch1, self.branch2]:
                if len(data) == 0: continue
                for j in range(3):
                    ax.scatter(data[:,0], data[:,j+1], color=colors[j], label=f"{labels[j]}", s=8, alpha = 1.0)
            ax.set_title(f"2D Trajectory (Contact = True)\nSolid: self-interation O / Faded: self-interation X")
        
        else:
            # --- [Case 2] Contact = False ---
            if len(self.full_math_data) > 0:
                for j in range(3):
                    ax.scatter(self.full_math_data[:,0], self.full_math_data[:,j+1], color=colors[j], label=f"{labels[j]}", s=8, alpha = 1.0)
            ax.set_title(f"2D Trajectory (Contact = False)\nFull Kinematic Space (Ignored Physical Bounds)")                
        ax.set_xlabel(r"$\rho_1$ [rad]")
        ax.set_ylabel(r"Folding Angles $\rho_2, \rho_3, \rho_4$ [rad]")     
        ax.set_xlim([-np.pi, np.pi])
        ax.set_ylim([-np.pi, np.pi])
        ax.set_aspect('equal')
        ax.axhline(0, color='black', linewidth=1)
        ax.axvline(0, color='black', linewidth=1)
        ax.grid(True, linestyle='--', alpha = 0.6)
        
        handles, labels_leg = ax.get_legend_handles_labels()
        by_label = dict(zip(labels_leg, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize='small')

        plt.tight_layout()
        plt.show()

    def show_animated_2d_plot(self, interval=0.001, skip_step=10): ## Can use when (contact = True)
        """
        Check dynamic orbit with real-time dot pointing
        interval: time between pointing
        skip_step: # of steps to skip for enhanced animation speed
        """
        # total animation time = (resolution / skip_step) * interval
        
        # --- Contact validation---
        if not self.contact:
            print(" [Notice] Animation is a tool for checking the folding until contact ")
            print("        When contact=False, use plot_2d_static() ")
            return
        # -----------------------------------
    
        if len(self.branch1) == 0 and len(self.branch2) == 0:
            print("No data for visualization. First, run run_simulation()")
            return
        
        # Activate interactive mode (real-time renewal)
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 8))
        
        colors = ['blue', 'green', 'red']
        labels = [r'$\rho_2$', r'$\rho_3$', r'$\rho_4$']
        
        ax.set_xlim([-np.pi, np.pi])
        ax.set_ylim([-np.pi, np.pi])
        ax.axhline(0, color='black', lw=1)
        ax.axvline(0, color='black', lw=1)
        ax.grid(True, ls='--', alpha=0.5)
        ax.set_title(f"2D Kinematic Propagation (Driving e{self.driving_crease})")
        ax.set_xlabel(r"$\rho_1$ (rad)")
        ax.set_ylabel(r"Folding Angles (rad)")

        # --- Animate each branch sequentially---
        for data_idx, data in enumerate([self.branch1, self.branch2]):
            if len(data) == 0: continue
            
            for i in range(0, len(data), skip_step):
                for j in range(3):
                    # rho1 vs (rho2, rho3, rho4)
                    ax.scatter(data[i, 0], data[i, j+1], color=colors[j], s=5, alpha=0.8)
                
                plt.pause(interval)
        
        print("End animation.")
        plt.ioff()
        plt.show()
    
    def _rodrigues_rotation_matrix(self, axis, angle):
        """[utility] rotation matrix amount angle with axis"""
        norm = np.linalg.norm(axis)
        if norm < 1e-9: return np.eye(3)
        axis = axis / norm
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    
    def get_3d_geometry(self, rhos_rad):
        """Calculate 4 planes coordinates with present folding angles"""
        a1, a2, a3, a4 = self.original_alphas 
        r1, r2, r3, r4 = rhos_rad
        origin = np.array([0.0, 0.0, 0.0])
        
        v4 = np.array([1.0, 0.0, 0.0]) 
        v1 = np.array([np.cos(a4), np.sin(a4), 0.0])
        
        v2_flat = np.array([np.cos(a4 + a1), np.sin(a4 + a1), 0.0])
        v2 = self._rodrigues_rotation_matrix(v1, r1) @ v2_flat
        
        v3_flat = np.array([np.cos(-a3), np.sin(-a3), 0.0])
        v3_test_pos = self._rodrigues_rotation_matrix(v4, r4) @ v3_flat
        v3_test_neg = self._rodrigues_rotation_matrix(v4, -r4) @ v3_flat
        
        target_cos = np.cos(a2)
        if abs(np.dot(v2, v3_test_pos) - target_cos) < abs(np.dot(v2, v3_test_neg) - target_cos):
            v3 = v3_test_pos
        else:
            v3 = v3_test_neg
        faces = [
            [origin, v4, v1], # Face 4
            [origin, v1, v2], # Face 1
            [origin, v2, v3], # Face 2
            [origin, v3, v4]  # Face 3
        ]
        vertices = np.array([origin, v1, v2, v3, v4])
        return faces, vertices
    
    
    def show_3d_interactive(self):

        from matplotlib.widgets import Slider, RadioButtons, CheckButtons
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        available_branches = []
        if len(self.branch1) > 0 or len(self.math_branch1) > 0: available_branches.append('Branch 1')
        if len(self.branch2) > 0 or len(self.math_branch2) > 0: available_branches.append('Branch 2')
        
        if not available_branches:
            print("No data for visualization. First, run run_simulation()")
            return

        fig = plt.figure(figsize=(15, 7))
        
        ax_3d = fig.add_axes([0.02, 0.25, 0.45, 0.65], projection='3d')
        ax_3d.set_xlim([-1, 1]); ax_3d.set_ylim([-1, 1]); ax_3d.set_zlim([-1, 1])
        ax_3d.set_xlabel('X'); ax_3d.set_ylabel('Y'); ax_3d.set_zlabel('Z')
        
        ax_2d = fig.add_axes([0.55, 0.25, 0.4, 0.65])
        ax_2d.set_xlim([-np.pi, np.pi]); ax_2d.set_ylim([-np.pi, np.pi])
        ax_2d.axhline(0, color='black', lw=1); ax_2d.axvline(0, color='black', lw=1)
        ax_2d.grid(True, ls='--', alpha=0.5)
        ax_2d.set_xlabel(f"Driving Angle (e{self.driving_crease}) [rad]")
        ax_2d.set_ylabel("Folding Angles [rad]")

        best_idx = self.driving_crease - 1
        other_indices = [i for i in range(4) if i != best_idx]
        labels = [f"$e_{i+1}$" for i in range(4) if i != best_idx]
        colors = ['blue', 'green', 'red']
        
        ui_state = {'branch': available_branches[0], 'contact': self.contact}
        def get_active_data():
            if ui_state['branch'] == 'Branch 1': return self.branch1 if ui_state['contact'] else self.math_branch1
            else: return self.branch2 if ui_state['contact'] else self.math_branch2

        init_data = get_active_data()
        init_rhos = init_data[len(init_data) // 2]

        # 3D reset
        faces, vertices = self.get_3d_geometry(init_rhos)
        face_colors = ['lightgray', 'lightblue', 'lightgreen', 'lightcoral']
        poly3d = Poly3DCollection(faces, facecolors=face_colors, edgecolors='black', linewidths=2, alpha=0.8)
        ax_3d.add_collection3d(poly3d)
        scatter_3d = ax_3d.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color='black', s=50)

        # 2D reset
        colors = ['blue', 'green', 'red']
        bg_lines = [ax_2d.plot([], [], color=colors[j], label = labels[j], lw=2, alpha=0.4)[0] for j in range(3)]
        tracker_pts = [ax_2d.scatter([], [], color=c, s=100, edgecolors='black', zorder=5) for c in colors]
        tracker_vline = ax_2d.axvline(x=init_rhos[best_idx], color='gray', linestyle='--', alpha=0.7, zorder=1)
        sing_lines = [ax_2d.plot([], [], color=c, lw=2.0, linestyle='--', alpha=0.35, zorder=3)[0] for c in colors] # Singular lines

        def redraw_2d_background():
            data = get_active_data()
            if len(data) == 0: return
            sorted_data = data[np.argsort(data[:, best_idx])]
            
            if ui_state['contact']:
                l_width, l_alpha = 3.5, 0.95  # Contact=True: bold and dark
            else:
                l_width, l_alpha = 2.0, 0.4   # Contact=False: thin and light
                
            for j in range(3):
                y_idx = other_indices[j]
                bg_lines[j].set_data(sorted_data[:, best_idx], sorted_data[:, y_idx])
                bg_lines[j].set_linewidth(l_width); bg_lines[j].set_alpha(l_alpha)
            
            # --- Display conditional singularities ---
            data_rhos = data[:, best_idx]
            # Whether path covers (-pi ~ pi) (if over 6.0 rad, treated as fully defined)
            is_fully_defined = (data_rhos.max() - data_rhos.min()) >= 6.0
            
            show_singular = (not ui_state['contact']) and (not is_fully_defined) and hasattr(self, 'singular_states') and len(self.singular_states) > 0
            
            if show_singular:
                sing_data = np.array(self.singular_states)
                s_sorted = sing_data[np.argsort(sing_data[:, best_idx])]
                # add 'NaN' if there's long horizontal gap for clear plot
                x_vals = s_sorted[:, best_idx]
                gap_indices = np.where(np.diff(x_vals) > 1.0)[0] + 1
                
                for j in range(3):
                    y_idx = other_indices[j]
                    y_vals = s_sorted[:, y_idx]
                    
                    x_plot = np.insert(x_vals, gap_indices, np.nan)
                    y_plot = np.insert(y_vals, gap_indices, np.nan)
                    
                    sing_lines[j].set_data(x_plot, y_plot)
                    sing_lines[j].set_visible(True)
            else:
                for j in range(3):
                    sing_lines[j].set_visible(False)
            # ----------------------------------------

            title_c = "Physical Bound" if ui_state['contact'] else "Full Kinematic Space"
            ax_2d.set_title(f"Configuration Space ({ui_state['branch']} / {title_c})")
            ax_2d.legend(loc='upper right', fontsize='small', framealpha=0.7)
            fig.canvas.draw_idle()

        redraw_2d_background()

        ax_radio = fig.add_axes([0.02, 0.02, 0.12, 0.15]) # radio for branch selecting
        radio = RadioButtons(ax_radio, available_branches, active=0)
        ax_slider = fig.add_axes([0.2, 0.1, 0.6, 0.03]) # main slider
        slider = Slider(ax_slider, f'Fold (e{self.driving_crease})', -np.pi, np.pi, valinit=init_rhos[best_idx])
        ax_check = fig.add_axes([0.85, 0.05, 0.11, 0.09]) # contact checkbox
        check = CheckButtons(ax_check, ['Contact Bound'], [ui_state['contact']])

        def update_all(val=None):
            data = get_active_data()
            if len(data) == 0: return
            
            target_val = slider.val
            data_rhos = data[:, best_idx]
            global_min, global_max = data_rhos.min(), data_rhos.max()
            is_fully_defined = (global_max - global_min) >= 6.0 
            
            # --- Contact=False, combining singular states, set the global limit ---
            if not ui_state['contact'] and hasattr(self, 'singular_states') and len(self.singular_states) > 0:
                sing_rhos = np.array(self.singular_states)[:, best_idx]
                global_min = min(global_min, sing_rhos.min())
                global_max = max(global_max, sing_rhos.max())
            # ---------------------------------------------------------------------------------
            
            is_clamped = False
            if target_val < global_min or target_val > global_max:
                slider.eventson = False 
                target_val = np.clip(target_val, global_min, global_max)
                slider.set_val(target_val)
                slider.eventson = True
                is_clamped = True

            idx = np.abs(data_rhos - target_val).argmin()
            current_rhos = data[idx]
            display_x = current_rhos[best_idx]
            is_singular = False

            show_singular = (not ui_state['contact']) and (not is_fully_defined) and hasattr(self, 'singular_states') and len(self.singular_states) > 0
            
            # --- Through distance match, switch regular path and singular path ---
            if show_singular:
                sing_data = np.array(self.singular_states)
                s_idx = np.abs(sing_data[:, best_idx] - target_val).argmin()
                
                dist_to_data = np.abs(display_x - target_val)
                dist_to_sing = np.abs(sing_data[s_idx, best_idx] - target_val)
                
                # Snap when target_val closer to singular state than regular path
                if np.abs(sing_data[s_idx, best_idx] - target_val) < 0.05:
                    current_rhos = sing_data[s_idx]
                    display_x = current_rhos[best_idx]
                    is_singular = True
            # ----------------------------------------------------------------------------

            # title update
            if is_singular:
                ax_3d.set_title(f"??SINGULAR STATE (Transition) ??ne{self.driving_crease} = {np.rad2deg(display_x):.1f} deg", color='darkmagenta', fontweight='bold')
            elif is_clamped:
                bound_msg = "Contact Bound" if ui_state['contact'] else "Math Limit"
                ax_3d.set_title(f"LIMIT REACHED! ({bound_msg})\ne{self.driving_crease} = {np.rad2deg(target_val):.1f} deg", color='red', fontweight='bold')
            else:
                # Contact off
                ax_3d.set_title(f"Interactive 3D Fold\ne{self.driving_crease} = {np.rad2deg(display_x):.1f} deg", color='black')

            # Renew 3D Geometry
            new_faces, new_vertices = self.get_3d_geometry(current_rhos)
            poly3d.set_verts(new_faces)
            scatter_3d._offsets3d = (new_vertices[:, 0], new_vertices[:, 1], new_vertices[:, 2])
            
            # Renew 2D marker on plot
            tracker_vline.set_xdata([display_x, display_x])
            for j in range(3):
                y_idx = other_indices[j]
                tracker_pts[j].set_offsets([target_val, current_rhos[y_idx]])
            fig.canvas.draw_idle()

        def on_radio_click(label):
            ui_state['branch'] = label
            redraw_2d_background()
            mid_val = get_active_data()[len(get_active_data()) // 2, best_idx]
            slider.eventson = False; slider.set_val(mid_val); slider.eventson = True
            update_all()

        def on_check_click(label):
            ui_state['contact'] = not ui_state['contact']
            redraw_2d_background(); update_all()

        slider.on_changed(update_all)
        radio.on_clicked(on_radio_click)
        check.on_clicked(on_check_click)

        print("Starting Ultimate 3D Interactive Viewer...")
        plt.show()
                       
        
# --- Example ---
if __name__ == "__main__":
    # Elliptic Case
    #sector_angles = [60-0.5, 90-0.5, 135-0.5, 75-0.5]
    # Hyperbolic Case
    #sector_angles = [60+0.5, 90+0.5, 135+0.5, 75+0.5]
    # Developable Case
    #sector_angles = [60, 90, 135, 75]
    # Folding table
    #sector_angles = [112.5,112.5,90,135]
    # Flat-foldable
    sector_angles = [80, 120, 100, 60]
    sim = OrigamiDegree4Simulator(sector_angles, contact=True)
    sim.run_simulation(resolution=1000) 
    sim.plot_2d_static()
    #sim.show_animated_2d_plot()
    sim.show_3d_interactive()
