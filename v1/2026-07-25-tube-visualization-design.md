# Tube Visualization 추가 설계

Status: Approved — 2026-07-25

## 목적

`find_ff_tube_design`/`find_ne_tube_design`가 찾은 설계로 실제 N각형 단면
tube를 3D로 렌더링하고, N각형이 실제로 닫히는지(ring closure) 눈으로 확인할
수 있게 한다. 지금까지는 사용자가 직접 `OrigamiTesellator1D`를 만들어
`plot_3d()`를 호출해야 했는데, 이걸 `origami_tube_design.py` 안의 전용
함수로 감싼다.

## 범위

- **포함**: `origami_tube_design.py`에 함수 2개(+공용 내부 헬퍼 1개, +ring
  좌표 계산 헬퍼 1개) 추가. 정적 3D plot + 인터랙티브(슬라이더) 뷰어 둘 다.
  Ring(N개 유닛마다 한 바퀴) 경계를 강조하는 전용 렌더링.
- **제외**: 기존 4개 파일(`origami_simulator.py`, `origami_tesellator_1D.py`,
  `origami_forward_kinematics.py`, `origami_helix_inverse_design.py`) 수정
  없음 — `OrigamiTesellator1D`의 기존 public/private 메서드만 재사용.

## API

```python
def visualize_ff_tube(result, num_rings=10, interactive=False, save_path=None)
def visualize_ne_tube(result, num_rings=10, interactive=False, save_path=None)
```

- `result`: `find_ff_tube_design`/`find_ne_tube_design`가 반환한 dict.
- `num_rings`: N각형 고리를 몇 개 쌓을지 (`num_periods = result["target_N"] * num_rings`로 tessellator 생성).
- `interactive`: `False`(기본) → 정적 3D plot. `True` → fold 상태(rho0) 슬라이더 포함 뷰어.
- `save_path`: `interactive=False`일 때만 유효 (`plt.savefig`, 없으면 `plt.show()`). `interactive=True`인데 `save_path`가 주어지면 "interactive 모드에서는 무시됨" 경고 출력 후 무시.
- 전제조건 실패 시 (`result.get("alphas_deg") is None` 또는 FF는 `rho0_deg`/NE는 `rho_star_deg`가 `None`): `ValueError`로 명확한 메시지와 함께 즉시 실패 (그릴 유효한 설계가 없으므로, 다른 diagnostic 플롯들과 달리 조용히 NaN 처리하지 않는다).

두 함수는 각자 필요한 rho0 소스만 고르고, 실제 렌더링은 공용 내부 헬퍼에
위임한다:

```python
def _visualize_tube(result, rho0_deg, num_rings, interactive, save_path)
```

## Ring 경계 계산

```python
def _ring_marker_points(tess, target_N):
    """매 target_N번째 유닛의 origin 좌표 (ring 경계 지점들)."""
    origins = [tess.global_vertices[t][0]
               for t in range(0, len(tess.global_vertices), target_N)]
    return np.array(origins)
```
N각형이 정확히 닫히면(ω == 2π/N), 이 점들이 하나의 매끈한 나선(또는 원)을
그리며, 인접한 두 ring 지점 사이의 간격이 균일해야 한다 — 이게 바로
"N각형이 실제로 닫혔다"는 시각적 증거가 된다.

## 렌더링 방식

`OrigamiTesellator1D.plot_3d()`/`setup_interactive_viewer()`는 내부적으로
자체 fig/ax를 만들고 끝에 `plt.show()`를 호출해버려서, 우리가 ring 마커를
추가로 얹을 틈이 없다. 대신 그 두 메서드가 공통으로 쓰는 내부 헬퍼
`tess._render_faces_and_lines(ax, valid, poly, line_bb, line_e1, line_e3, line_cr)`
를 직접 재사용해서 면/backbone/crease를 그리고, 그 위에 우리가 만든 ring
마커 라인(굵은 마젠타색, 점 표시)을 추가로 그린다.

- **정적**: `tess.compute_strip_kinematics(np.deg2rad(rho0_deg))` 한 번 →
  `_render_faces_and_lines` → ring 마커 plot → 제목에 `target_N`,
  `omega_achieved_deg`, `r_achieved`, `g_achieved` 표시 → `savefig`/`show`.
- **인터랙티브**: `setup_interactive_viewer`와 동일한 슬라이더 구조
  (`rho0 (deg)`, 범위 -175~175)를 직접 구성하고, 슬라이더 `on_changed`
  콜백 안에서 `compute_strip_kinematics` 재계산 → `_render_faces_and_lines`
  갱신 → ring 마커 라인의 `set_data`/`set_3d_properties`도 함께 갱신.

## 리스크

- `num_rings`가 크면 (`num_periods = target_N * num_rings`) 계산/렌더링이
  느려질 수 있음 — 기본값 10으로 실전에서 감당 가능한 크기 확인 필요.
- NE variant는 이미 문서화된 대로 `g_achieved` ≈ 0이라, `visualize_ne_tube`
  결과는 나선이 아니라 평평한 원형 고리 스택처럼 보일 것 — 버그 아님, 이미
  아는 한계.
