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
    centers_world = np.array([v[0] for v in tessellator.global_vertices])

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
        # Copies in the assembly frame allow a sweep to use one fixed reference.
        "unit_centers_world": centers_world,
        "backbone_world": np.vstack((centers_world, V_output_end)),
        "P_base_world": P_base.copy(),
        "R_base_world": R_base.copy(),
        "R_end_world": R_last.copy(),
        "input_line_world": np.array([O_base, V_input_end]),
        "output_line_world": np.array([O_last, V_output_end]),
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
    return _sample_poses(tessellator, rho_vals, track=track,
                         reset=False, include_failures=include_failures)


def _sample_poses(tessellator, rho_vals, *, track, reset, include_failures=True):
    if reset:
        tessellator.reset()
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


def sample_backbone_motion(tessellator, rho_values_deg, *, reference="initial",
                           track=True, reset=True):
    """Sample all centers and the output tip in a declared coordinate frame.

    initial: first valid sample's input frame, held fixed for the whole series.
    current: each sample's own input frame (legacy relative-SE(3) convention).
    world: tessellator assembly coordinates. Input order is preserved; failed
    samples retain status, NaN geometry and segment boundaries. Angles are degrees.
    This executes a sweep and changes the strip's accepted state when successful.
    """
    angles = np.asarray(rho_values_deg, float)
    if angles.ndim != 1 or not angles.size or not np.isfinite(angles).all():
        raise ValueError("rho_values_deg must be a nonempty finite 1D sequence")
    if reference not in ("initial", "current", "world"):
        raise ValueError("reference must be initial, current or world")
    rows = _sample_poses(tessellator, angles, track=track, reset=reset)
    first = next((r for r in rows if r["valid"]), None)
    n = tessellator.total_units
    for row in rows:
        row.update(reference=reference,
                   reference_rho0_deg=(first["rho0_deg"] if first is not None
                                       and reference == "initial" else None))
        if not row["valid"]:
            row.update(unit_centers=np.full((n, 3), np.nan),
                       backbone=np.full((n + 1, 3), np.nan),
                       P_input=np.full(3, np.nan), R_input=np.full((3, 3), np.nan))
            continue
        ref = first if reference == "initial" else row
        R = np.eye(3) if reference == "world" else ref["R_base_world"]
        P = np.zeros(3) if reference == "world" else ref["P_base_world"]
        row.update(
            unit_centers=(row["unit_centers_world"] - P) @ R,
            backbone=(row["backbone_world"] - P) @ R,
            P_rel=R.T @ (row["unit_centers_world"][-1] - P),
            R_rel=R.T @ row["R_end_world"],
            P_input=R.T @ (row["P_base_world"] - P),
            R_input=R.T @ row["R_base_world"],
            input_line=(row["input_line_world"] - P) @ R,
            output_line=(row["output_line_world"] - P) @ R,
        )
    return rows


def draw_coordinate_frame(ax, origin, rotation, *, length=0.3, label=""):
    """Draw RGB X/Y/Z arrows at an actual pose; return their artists and tips."""
    P, R = np.asarray(origin, float), np.asarray(rotation, float)
    if (P.shape != (3,) or R.shape != (3, 3) or
            not np.isfinite(P).all() or not np.isfinite(R).all() or
            not np.isfinite(length) or length <= 0 or
            not np.allclose(R.T @ R, np.eye(3), atol=1e-7) or
            not np.isclose(np.linalg.det(R), 1, atol=1e-7)):
        raise ValueError("finite origin, proper rotation and positive length required")
    arrows, texts = [], []
    tips = P + length * R.T
    for j, (color, name) in enumerate(zip(("r", "g", "b"), ("X", "Y", "Z"))):
        arrows.append(ax.quiver(*P, *(length * R[:, j]), color=color,
                                linewidth=1.8, arrow_length_ratio=0.22))
        texts.append(ax.text(*tips[j], f"{label}:{name}" if label else name,
                             color=color, fontsize=8))
    return dict(arrows=arrows, texts=texts, tips=tips, origin=P.copy(),
                rotation=R.copy(), label=label)


def _frame_indices(rows, count, indices=None):
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 0:
        raise ValueError("frame_count must be a nonnegative integer")
    valid = [i for i, r in enumerate(rows) if r["valid"]]
    if indices is not None:
        chosen = list(indices)
        if any(isinstance(i, bool) or not isinstance(i, (int, np.integer))
               or i not in valid for i in chosen):
            raise ValueError("frame_indices must refer to valid samples")
        return list(dict.fromkeys(chosen))
    if not valid or count == 0:
        return []
    return [valid[i] for i in np.unique(np.rint(
        np.linspace(0, len(valid) - 1, min(count, len(valid)))).astype(int))]


def _broken_path(rows, getter):
    """Keep physical/numerical gaps even for a compacted successful sweep."""
    points, previous = [], None
    for row in rows:
        if not row["valid"]:
            if points:
                points.append(np.full(3, np.nan))
            previous = None
            continue
        if previous is not None and (
            row.get("segment_id", 0) != previous.get("segment_id", 0) or
            row.get("sample_index", 0) != previous.get("sample_index", -1) + 1
        ):
            points.append(np.full(3, np.nan))
        points.append(np.asarray(getter(row), float))
        previous = row
    return np.asarray(points).reshape(-1, 3)


def _finish_motion_plot(ax, rows, bounds, frames, title, *, failure_label=None):
    # Accumulate bounds for overlaid cases, including arrow tips, so frames
    # cannot be clipped by limits computed from the end trajectory alone.
    if not hasattr(ax, "origami_motion_bounds"):
        ax.origami_motion_bounds = []
        ax.origami_frames = []
        ax.origami_failures = []
    ax.origami_frames.extend(frames)
    ax.origami_failures.extend(r for r in rows if not r["valid"])
    ax.origami_motion_bounds.extend(bounds + [f["tips"] for f in frames])
    finite = [np.asarray(p).reshape(-1, 3) for p in ax.origami_motion_bounds]
    if finite:
        points = np.vstack(finite)
        points = points[np.isfinite(points).all(axis=1)]
        if len(points):
            center = (points.max(axis=0) + points.min(axis=0)) / 2
            radius = max(float(np.max(np.ptp(points, axis=0))) * 0.6, 0.1)
            for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
                setter(c - radius, c + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set(xlabel="X", ylabel="Y", zlabel="Z", title=title)
    failures = [r for r in rows if not r["valid"]]
    if failures:
        text = "; ".join(f"{r['rho0_deg']:g} deg: {r['status']}" for r in failures[:4])
        prefix = "Unavailable" + (f" ({failure_label})" if failure_label else "")
        ax.text2D(0.02, 0.02, prefix + ": " + text +
                  (f"; +{len(failures)-4} more" if len(failures) > 4 else ""),
                  transform=ax.transAxes, fontsize=8, color="firebrick")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)
    ax.origami_motion = rows
    return ax


def _motion_axes(ax):
    if ax is None:
        ax = plt.figure(figsize=(11, 8)).add_subplot(111, projection="3d")
    return ax


def _frame_length(points, requested):
    if requested is not None:
        if not np.isfinite(requested) or requested <= 0:
            raise ValueError("frame_length must be finite and positive")
        return float(requested)
    finite = np.asarray(points).reshape(-1, 3)
    finite = finite[np.isfinite(finite).all(axis=1)]
    return max(float(np.max(np.ptp(finite, axis=0))) * 0.07, 0.15) if len(finite) else 0.3


def plot_end_trajectory(sweep, *, ax=None, frame_count=4, frame_indices=None,
                        frame_length=None, show_base_frame=True,
                        label="End trajectory", color=None, show=True):
    """Plot end-center paths and labeled coordinate frames at selected samples.

    Accepts legacy run_trajectory_sweep (current input reference) or
    sample_backbone_motion output. P_rel is the last unit center, not the tip.
    """
    rows = list(sweep)
    chosen = _frame_indices(rows, frame_count, frame_indices)
    points = _broken_path(rows, lambda r: r["P_rel"])
    length = _frame_length(points, frame_length)
    ax = _motion_axes(ax)
    ax.plot(*points.T, label=label, color=color)
    frames, bounds = [], [points]
    valid = [r for r in rows if r["valid"]]
    if valid and show_base_frame:
        first = valid[0]
        frames.append(draw_coordinate_frame(
            ax, first.get("P_input", np.zeros(3)), first.get("R_input", np.eye(3)),
            length=length, label="B"))
    for i in chosen:
        row = rows[i]
        frames.append(draw_coordinate_frame(ax, row["P_rel"], row["R_rel"],
                                            length=length, label=f"E{i}"))
        ax.scatter(*row["P_rel"], color="black", s=12)
        ax.text(*row["P_rel"], f" {row['rho0_deg']:g} deg", fontsize=8)
    if valid:
        for row, key in ((valid[0], "input_line"), (valid[-1], "output_line")):
            line = np.asarray(row[key])
            ax.plot(*line.T, color=color, linewidth=2)
            bounds.append(line)
    ref = rows[0].get("reference", "current") if rows else "current"
    _finish_motion_plot(ax, rows, bounds, frames, f"End-center trajectory | reference={ref}",
                        failure_label=label)
    if show:
        plt.show()
    return ax


def plot_backbone_motion(motion, *, ax=None, show_unit_paths=True,
                         frame_count=3, frame_indices=None, frame_length=None,
                         unit_label_step=None, show=True):
    """Overlay one backbone per input angle and dotted per-unit center paths.

    Centers are unit-indexed; the final backbone point is the output tip.
    B/E arrows show input/end frames at selected poses; RGB means X/Y/Z.
    """
    rows = list(motion)
    chosen = _frame_indices(rows, frame_count, frame_indices)
    if len({r["reference"] for r in rows}) > 1:
        raise ValueError("all poses must use the same reference")
    valid = [r for r in rows if r["valid"]]
    points = np.vstack([r["backbone"] for r in valid]) if valid else np.empty((0, 3))
    length = _frame_length(points, frame_length)
    ax = _motion_axes(ax)
    frames = []
    n = len(valid[0]["unit_centers"]) if valid else 0
    step = max(1, int(np.ceil(n / 8))) if unit_label_step is None else unit_label_step
    if isinstance(step, bool) or not isinstance(step, (int, np.integer)) or step < 1:
        raise ValueError("unit_label_step must be a positive integer")
    for i, row in enumerate(rows):
        if not row["valid"]:
            continue
        color = plt.get_cmap("viridis")(i / max(len(rows) - 1, 1))
        ax.plot(*row["backbone"].T, "-o", color=color, linewidth=1.8, markersize=3,
                label=f"rho0={row['rho0_deg']:g} deg")
        if row is valid[0]:
            for unit in sorted(set(range(0, n, step)) | {n - 1}):
                ax.text(*row["unit_centers"][unit], f" u{unit}", fontsize=8)
        if i in chosen:
            for prefix, P, R in (("B", row["P_input"], row["R_input"]),
                                  ("E", row["P_rel"], row["R_rel"])):
                frames.append(draw_coordinate_frame(ax, P, R, length=length,
                                                    label=f"{prefix}{i}"))
    if show_unit_paths:
        for unit in range(n):
            path = _broken_path(rows, lambda r: r["unit_centers"][unit])
            ax.plot(*path.T, ":", color="gray", alpha=0.45, linewidth=0.8)
    ref = rows[0]["reference"] if rows else "initial"
    reference_angle = rows[0].get("reference_rho0_deg") if rows else None
    suffix = f" ({reference_angle:g} deg)" if reference_angle is not None else ""
    _finish_motion_plot(ax, rows, [points], frames,
                        f"Backbone poses | reference={ref}{suffix}")
    if show:
        plt.show()
    return ax


# ------------------------------------------------------------------
# 🚀 메인 스크립트 실행 제어 루틴
# ------------------------------------------------------------------
def _run_comparison_demo():
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
    reference_strip = None
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
        # These six paths nearly coincide. Keep one labeled end frame on the
        # comparison overview; callers can request more frames on each case.
        frame_indices = [max(i for i, r in enumerate(sweep) if r["valid"])] if index == 0 else []
        plot_end_trajectory(sweep, ax=ax, frame_indices=frame_indices, frame_length=0.3,
                            show_base_frame=index == 0, label=label,
                            color=color, show=False)
        if index == 0:
            reference_strip = strip
        if index:
            try:
                strip.export_to_mat(f"case_{index}_kinematics.mat", rho0_deg=45)
            except ValueError as error:
                print(label, "snapshot at 45 degrees not exported:", error)
    ax.set(
        xlabel="X",
        ylabel="Y",
        zlabel="Z",
        title="End-center trajectories | current input reference | RGB = X/Y/Z",
    )
    # Explicit margins keep 3D labels/legend on canvas; tight_layout can expand
    # the axes beyond the figure when many projected text artists are present.
    fig.subplots_adjust(left=0.04, right=0.93, bottom=0.10, top=0.90)
    if reference_strip is not None:
        motion = sample_backbone_motion(reference_strip, [-150, -120, -90, -60, -20])
        plot_backbone_motion(motion, show=False)
    plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Strip end trajectories and backbone poses")
    parser.add_argument("--demo", choices=("comparison", "backbone"), default="comparison")
    parser.add_argument("--periods", type=int, default=5,
                        help="four-vertex Fig. 3F cell repeats in backbone demo")
    parser.add_argument("--angles", type=float, nargs="+", default=[15, 45, 90, 135, 170],
                        help="ordered drive angles in degrees for backbone demo")
    args = parser.parse_args()
    if args.demo == "comparison":
        _run_comparison_demo()
        return
    # Imada--Tachi Fig. 3F angular/connectivity data; unit lengths are a
    # visualization choice, not a reconstruction of the paper's finite panels.
    configs = [
        dict(alphas=[120, 60, 60, 120], iout=1, sigma=-1),
        dict(alphas=[60, 60, 120, 120], iout=2, sigma=1),
        dict(alphas=[120, 60, 60, 120], iout=3, sigma=-1),
        dict(alphas=[120, 120, 60, 60], iout=2, sigma=1),
    ]
    strip = OrigamiTesellator1D(configs, args.periods, contact=True, verbose=False)
    motion = sample_backbone_motion(strip, args.angles)
    plot_backbone_motion(motion, show=False)
    plot_end_trajectory(motion, show=False)
    for row in motion:
        if not row["valid"]:
            print(f"{row['rho0_deg']:g} deg: {row['status']}", row["termination"])
    plt.show()


if __name__ == "__main__":
    main()
