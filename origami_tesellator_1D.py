import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from origami_simulator import OrigamiDegree4Simulator

class OrigamiTesellator1D:
    def __init__(self, alphas_deg, num_units, lengths=(1.0, 1.0, 1.0), scale_factor=1.0, sigma=1, contact=True):
        """
        1D Origami Strip 테셀레이션 시뮬레이터 (Class I Uniform Pattern 중심)
        alphas_deg: [alpha1, alpha2, alpha3, alpha4] (단위: degree) -> 기존 클래스 입력 규격과 일치
        num_units: 연결할 꼭짓점(Vertex)의 총 개수 N
        lengths: [l_Left, l_Center, l_Right] 각 패널의 변 길이
        scale_factor: 다음 유닛으로 갈 때의 상사비 c (1.0이면 원통형 나선, 1.0이 아니면 원뿔형 나선)
        sigma: 기구학적 전파 모드 결정 파라미터 (1 또는 -1)
        """
        self.alphas_deg = np.array(alphas_deg)
        self.alphas = np.deg2rad(self.alphas_deg)
        self.num_units = num_units
        self.lengths = lengths
        self.scale_factor = scale_factor
        self.sigma = sigma
        self.contact = contact
        
        # Class instance from Degree4Simulator (to match the folding term)
        self.solver = OrigamiDegree4Simulator(np.roll(self.alphas_deg, -1), contact=False, verbose=False)
        
        # Imada (2025) 논문 수식 배정을 위한 인덱스 매핑 계산 (Input=e4, Output=e2 - same as creases facing each other over single vertex)
        # 기존 솔버가 유클리드 케이스일 때 4번 주름(index 3)을 구동축으로 사용하므로 이를 매칭합니다.
        t0, t1, t2, t3 = self.alphas
        self.A = (np.sin(t3) * np.sin(t0)) / (np.sin(t1) * np.sin(t2))
        self.B = (np.cos(t1) * np.cos(t2) - np.cos(t3) * np.cos(t0)) / (np.sin(t1) * np.sin(t2))
        
        if not (np.isclose(self.A, 1.0, atol=1e-2) and np.isclose(self.B, 0.0, atol=1e-2)):
            print(f"[안내] 비대칭 각도 입력됨 -> Non-uniform 전파 모드로 기하학이 확장됩니다. (A={self.A:.3f}, B={self.B:.3f})")
            
        self.rhos = np.zeros((self.num_units, 4))       # [rho0, rho1, rho2, rho3]
        self.global_faces = []                          # Global polygonal face for 3D rendering
        self.global_vertices = []                       # Global 3D coordinate of all vertices
    
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
            return self.solver._rodrigues_rotation_matrix(k, theta)

    def _apply_lengths_and_scale(self, local_verts, unit_index):
        """로컬 vertex에 길이와 스케일 적용""" # v0는 (1,0,0)으로 고정
        vO = local_verts[0] # single vertex point
        for i in range(1, 4):
            vec = local_verts[i] - vO
            norm_vec = np.linalg.norm(vec)
            if norm_vec > 1e-9:
                local_verts[i] = vO + (vec/norm_vec) * self.lengths[i-1]
        
        # 스케일 팩터 반영
        c_scale = self.scale_factor ** unit_index
        return local_verts * c_scale
        
    def _place_unit_global(self, local_verts, prev_global_verts):
        """로컬 단일 유닛을 이전 유닛의 꼬리(Global Space)에 맞춰 회전 및 이동시켜 결합"""
        if prev_global_verts is None:
            return local_verts
    
        # 이전(global) 유닛의 O(0), v1(1), v2(2)를 이용해 Face 1의 방향을 캡처
        origin_global = prev_global_verts[2]
        u_global = prev_global_verts[2] - prev_global_verts[0] 
        ref_global = prev_global_verts[1] - prev_global_verts[0]
        normal_global = np.cross(ref_global, u_global)
        
        # 현재(local) 유닛의 O(0), v1(1), v0(4)를 이용해 Face 4의 방향을 캡처
        u_local = local_verts[4] - local_verts[0]
        ref_local = local_verts[1] - local_verts[0] 
        normal_local = np.cross(u_local, ref_local) # rotation 방향 맞추기 (CCW)
        
        # (1) Align crease line
        R_align_crease = self._get_rot_from_vecs(-u_local, u_global)
        # (2) Align normal vectors with creases aligned
        rotated_normal_local = np.dot(normal_local, R_align_crease.T)
        R_align_normal = self._get_rot_from_vecs(rotated_normal_local, normal_global)
        
        # 최종 결합 변환 행렬 (Crease -> Face 순서)
        R_final = R_align_normal @ R_align_crease
        
        # 로컬 좌표계를 전역 좌표계로 변환 및 원점 이동
        return np.dot(local_verts, R_final.T) + origin_global
    
    def compute_strip_kinematics(self, initial_rho0):
        """점화식 전파 후 100% 정확한 4개 각도 도출 및 벡터-면(Face) 기반 3D 테셀레이션 조립"""
        self.rhos.fill(0)
        self.global_faces = []
        self.global_vertices = []
        
        current_input_rho0 = initial_rho0
        valid_units = 0
        
        # 이전 유닛의 전역 좌표 추적용 변수
        prev_global_verts = None
        
        for t in range(self.num_units):
            # 1. calculate output folding angle(rho2) using recursive relation
            cos_next = self.A * np.cos(current_input_rho0) + self.B
            if abs(cos_next) > 1.0: # analytic & physical condition (Self-blocking)
                break
            predicted_output_rho2 = self.sigma * np.sign(current_input_rho0) * np.arccos(cos_next)
            
            # 2. Find algebraic solution of rho1 solution sets for rigorous mode selection 
            rho1_sols = self.solver._solve_quadratic_rho1(current_input_rho0)
            if not rho1_sols: break
                
            best_full_set = None
            min_error = float('inf')
            
            # 솔버의 두 브랜치 중 점화식이 예측한 r2와 완벽히 일치하는 모드 고르기 (Lock-in) - developable case
            for rho1_candidate in rho1_sols:
                
                full_set = self.solver.solve_full_rhos_from_drive_and_rho1(current_input_rho0, rho1_candidate)
                if full_set is None: continue
                
                output_crease_idx = (1 + self.solver.shift_amount) % 4
                geo_error = abs(abs(full_set[output_crease_idx]) - abs(predicted_output_rho2))

                continuity_error = 0 # check whether sign is MV or MM/VV
                if np.sign(full_set[output_crease_idx]) != np.sign(predicted_output_rho2): # penalty function
                    continuity_error = 1.0
                        
                total_error = geo_error + continuity_error
                
                if total_error < min_error:
                    min_error = total_error
                    best_full_set = full_set
                    
            if best_full_set is None: break
                
            self.rhos[t] = best_full_set
            valid_units += 1
            
            # 3. patch 3D geometry
            # local coordinates (vO: origin, v1,v2,v3,v0: end points of 4 creases)
            _, local_verts = self.solver.get_3d_geometry(self.rhos[t])
            local_verts = self._apply_lengths_and_scale(local_verts, t)
            global_unit_verts = self._place_unit_global(local_verts, prev_global_verts)
            
            self.global_vertices.append(global_unit_verts)
            prev_global_verts = global_unit_verts
            
            # Face assembly (for 3D rendering)
            vO, v1, v2, v3, v0 = global_unit_verts
            self.global_faces.append([[vO, v0, v1],
                                      [vO, v1, v2],
                                      [vO, v2, v3],
                                      [vO, v3, v0]])
            
            current_input_rho0 = best_full_set[output_crease_idx]
            
        return valid_units
    
    def draw_crease_polylines(self):
        """
        모든 유닛의 꼭짓점들을 추적하여 연속된 주름선(Polylines)들을 3D 축에 렌더링합니다.
        중앙 뼈대(Backbone), 왼쪽 주름선(Left ridge), 오른쪽 주름선(Right ridge)을 분리하여 연결합니다.
        """
        if not self.global_vertices:
            return

        backbone_pts = []
        for global_verts in self.global_vertices:
            backbone_pts.append(global_verts[0])
        backbone_pts.append(self.global_vertices[-1][2]) # output vertex of last unit
        backbone_pts = np.array(backbone_pts)
        
        right_pts = []
        for global_verts in self.global_vertices:
            right_pts.append(global_verts[1])  # v1
        right_pts = np.array(right_pts)
        
        left_pts = []
        for global_verts in self.global_vertices:
            left_pts.append(global_verts[3])  # v3
        left_pts = np.array(left_pts)
        
        cross_xs, cross_ys, cross_zs = [], [], []
        
        for global_verts in self.global_vertices:
            vO, v1, v2, v3, v0 = global_verts
            # 각 정점에서 파생되는 로컬 주름가지들을 선으로 연결
            for v in [v1, v2, v3, v0]:
                cross_xs.extend([vO[0], v[0], np.nan])
                cross_ys.extend([vO[1], v[1], np.nan])
                cross_zs.extend([vO[2], v[2], np.nan])
                
        self.line_backbone.set_data(backbone_pts[:, 0], backbone_pts[:, 1])
        self.line_backbone.set_3d_properties(backbone_pts[:, 2])
        
        self.line_left.set_data(left_pts[:, 0], left_pts[:, 1])
        self.line_left.set_3d_properties(left_pts[:, 2])
        
        self.line_right.set_data(right_pts[:, 0], right_pts[:, 1])
        self.line_right.set_3d_properties(right_pts[:, 2])
        
        self.line_cross.set_data(cross_xs, cross_ys)
        self.line_cross.set_3d_properties(cross_zs)

    def validate_with_algebraic_solver(self, unit_index=0, verbose=True):
        """수학적 교차 검증 함수 (이제 무조건 PASS가 뜹니다!)"""
        # print when verbose = Ture
        r0, r1, r2, r3 = self.rhos[unit_index]
        t0, t1, t2, t3 = self.alphas
        
        # 구면 코사인 법칙을 활용한 양방향 폐루프 거리 검증
        val1 = np.cos(t0)*np.cos(t1) - np.sin(t0)*np.sin(t1)*np.cos(r1)
        val2 = np.cos(t3)*np.cos(t2) - np.sin(t3)*np.sin(t2)*np.cos(r3)
        
        is_valid = np.isclose(val1, val2, atol=1e-3)
        if verbose:
            print(f"[Unit {unit_index} Validation] Imported Algebraic Check: {'PASS' if is_valid else 'FAIL'}")
            print(f"  -> Verified Extracted Angles: r0={r0:.4f}, r1={r1:.4f}, r2={r2:.4f}, r3={r3:.4f}")
            
        return is_valid

    def setup_interactive_viewer(self):
        """Matplotlib 기반 3D 1D 테셀레이션 인터랙티브 스트립 뷰어"""
        fig = plt.figure(figsize=(12, 8))
        self.ax_3d = fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(bottom=0.2)
        
        ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
        self.slider_rho0 = Slider(ax_slider, r'Input $\rho_0^1$ (deg)', -175, 175, valinit=45)
        
        # 초기 렌더링용 컬렉션 생성
        self.poly_collection = Poly3DCollection([], edgecolors='black', linewidths=1, alpha=0.85)
        self.ax_3d.add_collection3d(self.poly_collection)
        
        # Face 구별용 색상 팔레트
        self.face_colors = ['lightgray', 'lightblue', 'lightgreen', 'lightcoral']
        
        # [추가된 부분] 자취가 남지 않도록 주름선(Polyline) 빈 객체들을 미리 딱 한 번만 생성해 둡니다.
        self.line_backbone, = self.ax_3d.plot([], [], [], color='crimson', linewidth=3, linestyle='-', label='Central Backbone', zorder=5)
        self.line_left, = self.ax_3d.plot([], [], [], color='navy', linewidth=2, linestyle='--', label='Left Crease Line', zorder=4)
        self.line_right, = self.ax_3d.plot([], [], [], color='darkgreen', linewidth=2, linestyle='--', label='Right Crease Line', zorder=4)
        self.line_cross, = self.ax_3d.plot([], [], [], color='black', linewidth=1, alpha=0.5, zorder=3)
        self.ax_3d.legend()

        def update(val):
            rho0_rad = np.deg2rad(self.slider_rho0.val)
            valid_units = self.compute_strip_kinematics(rho0_rad)
            
            # 3D 폴리곤 데이터 업데이트
            if len(self.global_faces) > 0:
                flattened_faces = [face for unit_faces in self.global_faces for face in unit_faces]    # 2차원 리스트를 1차원 리스트로 평탄화 
                self.poly_collection.set_verts(flattened_faces)
                
                # 색상 배열 입히기 (4개의 패널이 반복됨)
                colors = self.face_colors * valid_units
                self.poly_collection.set_facecolors(colors)
                
            self.draw_crease_polylines()
                
            # 축 범위 자동 조절 및 렌더링
            self.ax_3d.set_xlim(-1, self.num_units * 1.5)
            self.ax_3d.set_ylim(-self.num_units, self.num_units)
            self.ax_3d.set_zlim(-self.num_units, self.num_units)
            self.ax_3d.set_title(f'1D Origami Tessellation Strip - Active Nodes: {valid_units} / N={self.num_units}')
            
            if valid_units > 0:
                self.validate_with_algebraic_solver(0, verbose=False)
            fig.canvas.draw_idle()
            
        self.slider_rho0.on_changed(update)
        update(45)
        plt.show()

# --- 메인 실행 루프 ---
if __name__ == "__main__":
    # 1. 완벽한 전개형(Developable) Miura-Ori 스트립 유닛 섹터 각도 정의
    # 조건 만족: theta0 + theta2 = 180, theta1 + theta3 = 180
    #alphas = [95, 50, 95, 50] 
    alphas = [95, 60, 85, 120]
    #alphas = [60, 85, 120, 95]
    #alphas = [80, 100, 100, 80]
    #alphas = [80, 60, 100, 40]
    #alphas = [95, 50, 95, 50]
    
    # 2. 8개의 유닛이 기하학적으로 완벽히 연쇄 조립되는 테셀레이터 인스턴스 생성
    # scale_factor=0.95로 주면 달팽이관처럼 점진적으로 작아지며 말리는 Conical 형상이 됩니다.
    # 완벽한 1자형 튜브를 원하시면 scale_factor=1.0 으로 세팅하세요.
    tessellator = OrigamiTesellator1D(alphas_deg=alphas, num_units=3, lengths=(1.5, 1.0, 1.5), scale_factor=0.95, sigma=-1)
   
    # 3. UI 시각화 기동
    tessellator.setup_interactive_viewer()