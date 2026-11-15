import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 같은 폴더 내에 있는 사용자 테셀레이터 클래스를 불러옵니다.
from origami_tesellator_1D import OrigamiTesellator1D

def get_se3_frame(origin, c_main, c_adj, rho_val, is_base=True):
    """
    주어진 원점과 주름 벡터들을 바탕으로 로컬 좌표계를 형성한 뒤,
    주름 자체를 대칭 중심축으로 삼기 위해 X축(주름축) 기준으로 rho/2 만큼 회전 보정된 R을 반환합니다.
    """
    # 1. 기본 주름 방향을 로컬 X축으로 설정
    X = c_main / (np.linalg.norm(c_main) + 1e-9)
    
    # 2. 이웃 주름과의 외적을 통해 임시 패널 법선 렌더링
    if is_base:
        n_panel = np.cross(c_main, c_adj)  # c0 x c1
    else:
        n_panel = np.cross(c_adj, X)       # ckm1 x ck
        
    Z_panel = n_panel / (np.linalg.norm(n_panel) + 1e-9)
    Y_panel = np.cross(Z_panel, X)
    
    R_panel = np.column_stack((X, Y_panel, Z_panel))
    
    # 3. 주름축(로컬 X축) 기준으로 rho/2 만큼 회전하여 양 패널의 정중앙(각이등분선)으로 보정
    rho_half = rho_val / 2.0
    c_h = np.cos(rho_half)
    s_h = np.sin(rho_half)
    
    R_flip_x = np.array([
        [1,   0,    0],
        [0, c_h, -s_h],
        [0, s_h,  c_h]
    ])
    
    R_corrected = R_panel @ R_flip_x
    return R_corrected, origin

def extract_relative_se3(tessellator, rho0_deg):
    """
    주름축 기준으로 rho/2씩 보정된 Base Frame 대비 End-effector Frame의 상대 SE(3) 변환 정보와
    함께 물리적인 입력 및 출력 주름의 양 끝점(정점) 좌표들을 상대 좌표계 기준으로 변환하여 반환합니다.
    """
    # 1D 스트립 기구학 계산 실행 (성공한 유닛 개수가 반환됨)
    valid = tessellator.compute_strip_kinematics(np.deg2rad(rho0_deg))
    if valid != tessellator.total_units:
        return None

    # --- 1. Base Frame (입력 주름, 0번째 유닛) 추출 ---
    gv_first = tessellator.global_vertices[0]
    CLV_first = tessellator.cells[0]["CLV"]
    
    O_base = gv_first[0]
    V_input_end = gv_first[CLV_first[0]]
    
    c0_base = V_input_end - O_base
    c1_base = gv_first[CLV_first[1]] - O_base
    rho_in = tessellator.rhos[0][0]
    
    R_base, P_base = get_se3_frame(O_base, c0_base, c1_base, rho_in, is_base=True)

    # --- 2. End-effector Frame (출력 주름, 마지막 유닛) 추출 ---
    last_idx = valid - 1
    gv_last = tessellator.global_vertices[last_idx]
    cell_last = tessellator.cells[last_idx % tessellator.N]
    CLV_last = cell_last["CLV"]
    iout_last = cell_last["iout"]

    # O_last(말단 유닛 정점) 및 V_output_end(출력 주름 끝 정점)
    O_last = gv_last[0]
    V_output_end = gv_last[CLV_last[iout_last]]
    
    ck_last = V_output_end - O_last
    ckm1_last = gv_last[CLV_last[(iout_last - 1) % 4]] - O_last
    rho_out = tessellator.rhos[last_idx][iout_last]

    R_last, P_last = get_se3_frame(O_last, ck_last, ckm1_last, rho_out, is_base=False)

    # --- 3. Base Frame 기준 상대 좌표계로 변환 (상대 SE(3)) ---
    R_rel = R_base.T @ R_last
    P_rel = R_base.T @ (P_last - P_base)

    # 물리 주름 선분들을 Base Frame 상대 좌표계로 변환
    input_p0 = np.zeros(3)  # 원점 (0,0,0)
    input_p1 = R_base.T @ (V_input_end - O_base)

    output_p0 = R_base.T @ (O_last - O_base)
    output_p1 = R_base.T @ (V_output_end - O_base)

    return {
        "P_rel": P_rel,
        "R_rel": R_rel,
        "input_line": (input_p0, input_p1),
        "output_line": (output_p0, output_p1)
    }

def check_planarity(positions, tolerance=1e-5):
    """ SVD(특이값 분해)를 이용한 3차원 궤적 점들의 평면성 정량 검증 루틴 """
    if len(positions) < 3:
        return True, np.array([0, 0, 1]), 0.0
    centroid = np.mean(positions, axis=0)
    centered_positions = positions - centroid
    U, S, Vt = np.linalg.svd(centered_positions)
    normal_vector = Vt[2, :]
    distances = np.abs(np.dot(centered_positions, normal_vector))
    max_residual = np.max(distances)
    is_planar = max_residual < tolerance
    return is_planar, normal_vector, max_residual

def plot_sector_angle_configurations(case_alphas_history, base_alphas, colors):
    """ 콘솔 창에 각 케이스별 10개 유닛의 Sector Angle 구성과 기하학적 타입을 판정해 출력합니다. """
    num_cases = len(case_alphas_history)
    num_units = len(case_alphas_history[0])
    
    print("\n==========================================================================================")
    print("📋 [케이스별 Sector Angle 분포 및 기하학적 성질 판정 리포트]")
    print("==========================================================================================")
    print(f"💡 기준 오리지널 Alphas (Euclidean 레퍼런스): {base_alphas} | 합계: {sum(base_alphas):.2f}°\n")
    
    tol = 1e-7 
    for c_idx in range(num_cases):
        print(f"▶ [Case {c_idx+1}] 구성 내역:")
        print("-" * 90)
        for u_idx in range(num_units):
            alphas = case_alphas_history[c_idx][u_idx]
            sum_alphas = sum(alphas)
            if abs(sum_alphas - 360.0) < tol:
                geom_type = "📐 Euclidean"
            elif sum_alphas < 360.0:
                geom_type = "🔴 Elliptic (Conical)"
            else:
                geom_type = "🔵 Hyperbolic (Saddle)"
            alphas_str = ", ".join([f"{a:6.2f}" for a in alphas])
            print(f"  · Unit {u_idx+1:02d} | Alphas: [{alphas_str}] | 합계: {sum_alphas:6.2f}° | 기하 성질: {geom_type}")
        print("-" * 90 + "\n")
    print("==========================================================================================")

def run_trajectory_sweep(tessellator, rho_start, rho_end, steps=100):
    """ 입력각 범위를 구동하며 상대 위치 궤적 및 시각화용 풀 딕셔너리 리스트를 반환합니다. """
    rho_vals = np.linspace(rho_start, rho_end, steps)
    sweep_results = []
    for rho in rho_vals:
        res = extract_relative_se3(tessellator, rho)
        if res is not None:
            sweep_results.append(res)
    return sweep_results

# ------------------------------------------------------------------
# 🚀 메인 스크립트 실행 제어 루틴
# ------------------------------------------------------------------
if __name__ == "__main__":
    base_alphas = np.array([70, 20, 110, 160])
    num_periods = 2
    num_cases = 5
    rho_start, rho_end = -150, -20
    steps = 100
    q_len = 0.8
    
    np.random.seed(24)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']   
    
    case_alphas_history = []
    for c_idx in range(num_cases):
        case_units = []
        for u in range(num_periods):
            epsilon = np.random.uniform(-1.0, 1.0, size=4)
            perturbed_alphas = (np.array(base_alphas) + epsilon).tolist()
            case_units.append(perturbed_alphas)
        case_alphas_history.append(case_units)
    
    plot_sector_angle_configurations(case_alphas_history, base_alphas, colors)
    
    fig = plt.figure(figsize=(13, 11))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.quiver(0, 0, 0, 1, 0, 0, color='gray', linestyle=':', length=q_len, arrow_length_ratio=0.1, alpha=0.4)
    ax.quiver(0, 0, 0, 0, 1, 0, color='gray', linestyle=':', length=q_len, arrow_length_ratio=0.1, alpha=0.4)
    ax.quiver(0, 0, 0, 0, 0, 1, color='gray', linestyle=':', length=q_len, arrow_length_ratio=0.1, alpha=0.4)
    ax.text(q_len + 0.05, 0, 0, 'Ref X', color='gray', fontsize=9)
    ax.text(0, q_len + 0.05, 0, 'Ref Y', color='gray', fontsize=9)
    ax.text(0, 0, q_len + 0.05, 'Ref Z', color='gray', fontsize=9)
    ax.scatter(0, 0, 0, color='purple', s=60, zorder=5)

    all_positions = []
    input_crease_drawn = False

    # ------------------------------------------------------------------
    # 🎯 [Original Case] 연산 및 주름 선분 플로팅
    # ------------------------------------------------------------------
    print("==================================================================")
    print("🎯 [Original Case] (Epsilon = 0) 기구학 궤적 연산 기동")
    print("==================================================================")
    original_configs = [{"alphas": base_alphas.tolist(), "sigma": 1, "iout": 3} for _ in range(num_periods)]
    original_strip = OrigamiTesellator1D(cell_configs=original_configs, num_periods=1, lengths=(1.0, 1.0, 1.0), verbose=False)
    
    orig_res_list = run_trajectory_sweep(original_strip, rho_start, rho_end, steps=steps)
    
    if orig_res_list:
        orig_pts = np.array([res["P_rel"] for res in orig_res_list])
        all_positions.append(orig_pts)
        
        is_planar, normal_vec, max_err = check_planarity(orig_pts, tolerance=1e-4)
        print(f"✅ Original  | 평면성판정: {is_planar} | 최대평면이탈: {max_err:.6e}")
        
        ax.plot(orig_pts[:, 0], orig_pts[:, 1], orig_pts[:, 2], color='black', linestyle='--', linewidth=3, zorder=10, label='Original Reference Trajectory')
        
        for res in orig_res_list:
            if not input_crease_drawn:
                inp_p0, inp_p1 = res["input_line"]
                ax.plot([inp_p0[0], inp_p1[0]], [inp_p0[1], inp_p1[1]], [inp_p0[2], inp_p1[2]],
                        color='crimson', linewidth=4.5, zorder=12, label='Physical Input Crease')
                input_crease_drawn = True
                
            out_p0, out_p1 = res["output_line"]
            ax.plot([out_p0[0], out_p1[0]], [out_p0[1], out_p1[1]], [out_p0[2], out_p1[2]], color='gray', linewidth=0.5, alpha=0.08)
            
        f_out_p0, f_out_p1 = orig_res_list[-1]["output_line"]
        ax.plot([f_out_p0[0], f_out_p1[0]], [f_out_p0[1], f_out_p1[1]], [f_out_p0[2], f_out_p1[2]],
                color='black', linewidth=3.5, zorder=11, label='Final Output Crease (Original)')

    # ------------------------------------------------------------------
    # 🔄 [Perturbed Cases] 5개 변동 스트립 연산 및 .mat 데이터 추출 연동
    # ------------------------------------------------------------------
    print("\n==================================================================")
    print(f"🔄 불균일 무작위 Perturbed 케이스 ({num_cases}개) 연산 및 .mat 익스포트")
    print("==================================================================")
    
    for c_idx in range(num_cases):
        cell_configs = [{"alphas": case_alphas_history[c_idx][u], "sigma": 1, "iout": 3} for u in range(num_periods)]
        perturbed_strip = OrigamiTesellator1D(cell_configs=cell_configs, num_periods=1, lengths=(1.0, 1.0, 1.0), verbose=False)
        
        case_res_list = run_trajectory_sweep(perturbed_strip, rho_start, rho_end, steps=steps)
        if not case_res_list:
            continue
            
        case_pts = np.array([res["P_rel"] for res in case_res_list])
        all_positions.append(case_pts)
        
        is_planar, normal_vec, max_err = check_planarity(case_pts, tolerance=1e-4)
        print(f"📊 Case {c_idx+1} 판정 | 평면성판정: {is_planar} | 최대평면이탈: {max_err:.6f}")
        
        # [NumPy 차원 에러 버그 수정완료] 
        case_traj = case_pts
        ax.plot(case_traj[:, 0], case_traj[:, 1], case_traj[:, 2], color=colors[c_idx], lw=2.0, alpha=0.8, label=f'Case {c_idx+1} Trajectory')
        
        for res in case_res_list:
            op0, op1 = res["output_line"]
            ax.plot([op0[0], op1[0]], [op0[1], op1[1]], [op0[2], op1[2]], color=colors[c_idx], linewidth=0.5, alpha=0.05)
            
        f_p0, f_p1 = case_res_list[-1]["output_line"]
        ax.plot([f_p0[0], f_p1[0]], [f_p0[1], f_p1[1]], [f_p0[2], f_p1[2]], color=colors[c_idx], lw=2.0, zorder=8)
        
        P_end = case_res_list[-1]["P_rel"]
        R_end = case_res_list[-1]["R_rel"]
        f_len = 0.4
        
        ax.quiver(P_end[0], P_end[1], P_end[2], R_end[0, 0], R_end[1, 0], R_end[2, 0], color='r', length=f_len, arrow_length_ratio=0.2, lw=2, zorder=15)
        ax.quiver(P_end[0], P_end[1], P_end[2], R_end[0, 1], R_end[1, 1], R_end[2, 1], color='g', length=f_len, arrow_length_ratio=0.2, lw=2, zorder=15)
        ax.quiver(P_end[0], P_end[1], P_end[2], R_end[0, 2], R_end[1, 2], R_end[2, 2], color='b', length=f_len, arrow_length_ratio=0.2, lw=2, zorder=15)

        # ------------------------------------------------------------------
        # 💾 [사용자 요청 기능 추가] 테셀레이터 내장 export_to_mat 함수 연동
        # ------------------------------------------------------------------
        # 각 케이스별로 스윕 최종 연산 상태를 담은 독립 .mat 파일을 기록합니다.
        mat_filename = f"case_{c_idx+1}_kinematics.mat"
        try:
            # origami_tesellator_1D 내장 익스포트 함수 자동 호출
            perturbed_strip.export_to_mat(mat_filename, rho0_deg=45)
            print(f"  └ 💾 MATLAB 연동용 파일 백업 완료: {mat_filename}")
        except AttributeError:
            # 혹시 모를 모듈 가변성을 대비해 scipy.io.savemat 형태의 Fallback 구조 마련
            import scipy.io as sio
            # 최종 스냅샷의 물리적 형상 정보 및 접힘각 역추적 구조를 사전 구조화하여 백업
            sio.savemat(mat_filename, {
                'global_vertices': perturbed_strip.global_vertices,
                'rhos': perturbed_strip.rhos,
                'cell_alphas': case_alphas_history[c_idx],
                'end_effector_P': P_end,
                'end_effector_R': R_end
            })
            print(f"  └ 💾 Backup-Savemat 파일 백업 완료: {mat_filename}")

    # --- 3. 3D 시각화 레이아웃 종횡비 최적화 설정 ---
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title("End-Tip Trajectories with Input/Output Crease Segments & Local SE(3) Frames", fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    
    combined_pts = np.vstack(all_positions)
    max_range = np.array([combined_pts[:,0].max() - combined_pts[:,0].min(), 
                          combined_pts[:,1].max() - combined_pts[:,1].min(), 
                          combined_pts[:,2].max() - combined_pts[:,2].min(), q_len]).max() / 2.0
    mid_x, mid_y, mid_z = np.mean(combined_pts, axis=0)
    
    ax.set_xlim(mid_x - max_range - 0.2, mid_x + max_range + 0.2)
    ax.set_ylim(mid_y - max_range - 0.2, mid_y + max_range + 0.2)
    ax.set_zlim(mid_z - max_range - 0.2, mid_z + max_range + 0.2)
    
    print("\n==================================================================")
    print("🎉 모든 데이터 추출(.mat) 및 3D 그래프 통합 출력 스윕 완료!")
    print("==================================================================")
    plt.tight_layout()
    plt.show()