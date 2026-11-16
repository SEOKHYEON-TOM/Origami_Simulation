import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 같은 폴더 내에 있는 사용자 테셀레이터 클래스를 불러옵니다.
from origami_tesellator_1D import OrigamiTesellator1D, crease_frame
from origami_kinematics import classify_geometry


def get_se3_frame(origin, c_main, c_adj, rho_val, is_base=True):
    """
    주어진 원점과 주름 벡터들을 바탕으로 로컬 좌표계를 형성한 뒤,
    주름 자체를 대칭 중심축으로 삼기 위해 X축(주름축) 기준으로 rho/2 만큼 회전 보정된 R을 반환합니다.
    """
    # 1. 기본 주름 방향을 로컬 X축으로 설정
    c_main = np.asarray(c_main, float)
    c_adj = np.asarray(c_adj, float)
    origin = np.asarray(origin, float)
    if (
        origin.shape != (3,)
        or not np.isfinite(origin).all()
        or not np.isfinite(rho_val)
    ):
        raise ValueError("finite frame origin and fold required")

    # 2. 이웃 주름과의 외적을 통해 임시 패널 법선 렌더링
    if is_base:
        n_panel = np.cross(c_main, c_adj)  # c0 x c1
    else:
        n_panel = np.cross(c_adj, c_main)  # ckm1 x ck

    R_panel = crease_frame(c_main, n_panel)

    # 3. 주름축(로컬 X축) 기준으로 rho/2 만큼 회전하여 양 패널의 정중앙(각이등분선)으로 보정
    # rho rotates the before-panel normal into the after-panel normal.
    # Base starts from AFTER c0; output starts from BEFORE cout.
    rho_half = (-rho_val if is_base else rho_val) / 2.0
    c_h = np.cos(rho_half)
    s_h = np.sin(rho_half)

    R_flip_x = np.array([[1, 0, 0], [0, c_h, -s_h], [0, s_h, c_h]])

    R_corrected = R_panel @ R_flip_x
    return R_corrected, origin


def extract_relative_se3(tessellator, rho0_deg, *, compute=True, track=False):
    """
    주름축 기준으로 rho/2씩 보정된 Base Frame 대비 End-effector Frame의 상대 SE(3) 변환 정보와
    함께 물리적인 입력 및 출력 주름의 양 끝점(정점) 좌표들을 상대 좌표계 기준으로 변환하여 반환합니다.
    """
    # 1D 스트립 기구학 계산 실행 (성공한 유닛 개수가 반환됨)
    if compute:
        valid = tessellator.compute_strip_kinematics(np.deg2rad(rho0_deg), track=track)
        result = tessellator.last_attempt
    else:
        result = tessellator.last_result
        valid = len(tessellator.global_vertices)
        if result is not None and abs(result.input_angle - np.deg2rad(rho0_deg)) > 1e-8:
            raise ValueError("requested angle differs from the accepted strip state")
    if valid != tessellator.total_units or result is None or not result.complete:
        return None

    # --- 1. Base Frame (입력 주름, 0번째 유닛) 추출 ---
    gv_first = tessellator.global_vertices[0]
    CLV_first = tessellator.cells[0]["CLV"]

    O_base = gv_first[0]
    V_input_end = gv_first[CLV_first[0]]

    c0_base = V_input_end - O_base
    c1_base = gv_first[CLV_first[1]] - O_base
    rho_in = tessellator.get_fold_angles(0)[0]

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
    rho_out = tessellator.get_fold_angles(last_idx)[iout_last]

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
        "output_line": (output_p0, output_p1),
        "rho0_deg": float(rho0_deg),
        "status": "OK",
        "valid": True,
        "contact_scope": result.contact_scope,
        "global_contact_checked": False,
    }


def check_planarity(positions, tolerance=1e-5):
    """SVD(특이값 분해)를 이용한 3차원 궤적 점들의 평면성 정량 검증 루틴"""
    positions = np.asarray(positions, float).reshape(-1, 3)
    positions = positions[np.isfinite(positions).all(axis=1)]
    if len(positions) < 3:
        return False, np.full(3, np.nan), np.nan
    centroid = np.mean(positions, axis=0)
    centered_positions = positions - centroid
    U, S, Vt = np.linalg.svd(centered_positions)
    normal_vector = Vt[2, :]
    distances = np.abs(np.dot(centered_positions, normal_vector))
    max_residual = np.max(distances)
    is_planar = max_residual < tolerance
    return is_planar, normal_vector, max_residual


def plot_sector_angle_configurations(case_alphas_history, base_alphas, colors):
    """콘솔 창에 각 케이스별 10개 유닛의 Sector Angle 구성과 기하학적 타입을 판정해 출력합니다."""
    num_cases = len(case_alphas_history)
    num_units = len(case_alphas_history[0])

    print(
        "\n=========================================================================================="
    )
    print("📋 [케이스별 Sector Angle 분포 및 기하학적 성질 판정 리포트]")
    print(
        "=========================================================================================="
    )
    print(
        f"💡 기준 오리지널 Alphas (Euclidean 레퍼런스): {base_alphas} | 합계: {sum(base_alphas):.2f}°\n"
    )

    tol = 1e-7
    for c_idx in range(num_cases):
        print(f"▶ [Case {c_idx+1}] 구성 내역:")
        print("-" * 90)
        for u_idx in range(num_units):
            alphas = case_alphas_history[c_idx][u_idx]
            sum_alphas = sum(alphas)
            case = classify_geometry(alphas)["case"]
            if case == "developable":
                geom_type = "📐 Euclidean"
            elif case == "elliptic":
                geom_type = "🔴 Elliptic (Conical)"
            else:
                geom_type = "🔵 Hyperbolic (Saddle)"
            alphas_str = ", ".join([f"{a:6.2f}" for a in alphas])
            print(
                f"  · Unit {u_idx+1:02d} | Alphas: [{alphas_str}] | 합계: {sum_alphas:6.2f}° | 기하 성질: {geom_type}"
            )
        print("-" * 90 + "\n")
    print(
        "=========================================================================================="
    )


def run_trajectory_sweep(
    tessellator,
    rho_start,
    rho_end,
    steps=100,
    *,
    track=True,
    reset=True,
    include_failures=True,
):
    """입력각 범위를 구동하며 상대 위치 궤적 및 시각화용 풀 딕셔너리 리스트를 반환합니다."""
    if (
        isinstance(steps, bool)
        or not isinstance(steps, (int, np.integer))
        or steps < 2
        or not np.isfinite([rho_start, rho_end]).all()
    ):
        raise ValueError("finite endpoints and integer steps>=2 required")
    if reset:
        tessellator.reset()
    rho_vals = np.linspace(rho_start, rho_end, steps)
    sweep_results = []
    segment = 0
    after_gap = True
    for index, rho in enumerate(rho_vals):
        frame_error = None
        try:
            res = extract_relative_se3(tessellator, rho, track=track)
        except ValueError as error:
            res = None
            frame_error = str(error)
        if res is None:
            attempt = tessellator.last_attempt
            res = dict(
                P_rel=np.full(3, np.nan),
                R_rel=np.full((3, 3), np.nan),
                input_line=(np.full(3, np.nan), np.full(3, np.nan)),
                output_line=(np.full(3, np.nan), np.full(3, np.nan)),
                rho0_deg=float(rho),
                status=attempt.status if frame_error is None else "NUMERICAL_FAILURE",
                valid=False,
                termination=(
                    attempt.termination
                    if frame_error is None
                    else dict(reason="NUMERICAL_FAILURE", detail=frame_error)
                ),
                global_contact_checked=False,
                contact_scope=attempt.contact_scope,
            )
            after_gap = True
        else:
            if after_gap:
                segment += 1
            after_gap = False
        res.update(sample_index=index, segment_id=segment if res["valid"] else None)
        if include_failures or res["valid"]:
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
    np.random.seed(24)
    histories = [
        [
            (base_alphas + np.random.uniform(-1, 1, 4)).tolist()
            for _ in range(num_periods)
        ]
        for _ in range(num_cases)
    ]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    plot_sector_angle_configurations(histories, base_alphas, colors)
    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(111, projection="3d")
    all_finite = []
    for index, angles in enumerate([[base_alphas.tolist()] * num_periods] + histories):
        cfg = [dict(alphas=a, sigma=1, iout=3) for a in angles]
        strip = OrigamiTesellator1D(cfg, 1, verbose=False)
        sweep = run_trajectory_sweep(strip, rho_start, rho_end, steps)
        valid = [r for r in sweep if r["valid"]]
        label = "Original" if index == 0 else f"Case {index}"
        if not valid:
            print(
                label, "no complete admissible strip; first reason:", sweep[0]["status"]
            )
            continue
        positions = np.array(
            [r["P_rel"] for r in sweep]
        )  # NaNs break failed intervals.
        all_finite.extend(r["P_rel"] for r in valid)
        planar, _, residual = check_planarity(positions, tolerance=1e-4)
        print(
            label,
            "valid frames",
            len(valid),
            "/",
            steps,
            "planar",
            planar,
            "max residual",
            residual,
        )
        color = "black" if index == 0 else colors[index - 1]
        ax.plot(*positions.T, c=color, ls="--" if index == 0 else "-", label=label)
        first, last = valid[0], valid[-1]
        for key, width in [("input_line", 3), ("output_line", 2)]:
            line = first[key] if key == "input_line" else last[key]
            ax.plot(*np.asarray(line).T, c=color, lw=width)
        for j, c in enumerate(("r", "g", "b")):
            ax.quiver(*last["P_rel"], *last["R_rel"][:, j], color=c, length=0.3)
        if index:
            try:
                strip.export_to_mat(f"case_{index}_kinematics.mat", rho0_deg=45)
            except ValueError as error:
                print(label, "snapshot at 45 degrees not exported:", error)
    if all_finite:
        points = np.array(all_finite)
        center = (points.max(axis=0) + points.min(axis=0)) / 2
        radius = max(float(np.max(np.ptp(points, axis=0))) / 2, 0.4)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.legend()
    ax.set(
        xlabel="X",
        ylabel="Y",
        zlabel="Z",
        title="Relative strip trajectories (gaps retained)",
    )
    fig.tight_layout()
    plt.show()
