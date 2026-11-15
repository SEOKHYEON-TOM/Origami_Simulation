# Origami N각형 Tube Cross-Section 설계 — Screw-Closure 탐색 도구

Status: Approved (design) — 2026-07-24

## 1. 배경 및 목적

목표: `OrigamiTesellator1D`로 만든 strip이, N각형 단면을 가지는 tube를 이루면서 folding이
진행됨에 따라 동적으로 helical shape을 갖도록 하는 sector-angle 구성을 찾는다.

핵심 기하학적 조건 (이전 대화에서 도출):

- Strip의 한 period(반복 단위)를 지나는 rigid transform은 screw motion
  `S(ρ) = (ω(ρ), r(ρ), g(ρ))` (회전각, 회전축까지 반경, 축방향 이동량)로 분해된다.
- N각형 단면이 "닫히려면"(polygon closure) 이 screw의 회전각이 정확히
  `ω = 2π/N` 이어야 한다. (N=4 → ω=90°)
- Flat-foldability(Kawasaki 조건)는 필요하지 않음 — 있어도 되고 없어도 됨.
  대신 mirror(reflection) 기반 구성 없이, **단일 strip 타입을 tube 축 둘레로
  2π/N 회전 배치**하는 것으로 N개 strip 간 screw 일치 조건
  `S1=S2=...=S_N=S(t)`을 구성상 자동으로 만족시킨다 (회전은 같은 축의
  screw와 항상 commute하므로 mirror 구성의 "2 reflection = 순수 회전" 퇴화가
  발생하지 않음).

두 가지 "dynamic" 시나리오를 모두 다룬다:

- **FF (Flat-foldable) variant**: Kawasaki 조건을 만족하는 단일 cell
  (`θ0,θ1,180-θ0,180-θ1`)에서는 strip 전체 길이에서 folding angle ρ가
  일정(Class I, A=1,B=0)하다. 외부 actuation 파라미터 ρ0을 0→π로 스윕하는
  "시간축" 관점에서, 양 끝(ρ0→0, ρ0→π)은 평면(펼쳐진 시트/완전히 접힌
  스택)이라 torsion이 없고, 중간 어느 ρ0에서 ω=2π/N을 정확히 지나가는 것을
  목표로 한다. → "펼쳐짐 ↔ 접힘 사이 중간 상태가 helical tube."
- **NE (Non-Euclidean) variant**: bird's-foot 대칭(`α1=α2,α3,α4`)의
  Kawasaki 조건을 깨는 strip에서, 안정 고정점 ρ*(Class II, |A|<1)를
  ω(ρ*)=2π/N이 되도록 설계한다. Strip의 "공간축"(cell index t)을 따라
  ρ_t가 ρ*로 수렴하므로, tube를 한쪽 끝에서부터 펼쳐나갈수록(cell이
  늘어날수록) 단면이 점점 정확한 N각형 helix에 가까워진다.
  → "펼치는 과정 자체가 비-helix에서 완벽한 helix로 수렴."

## 2. 범위

**포함**:
- N=4 (사각형 tube)로 먼저 검증. 함수 시그니처는 임의 N을 파라미터로 받되,
  검증/데모는 N=4로 진행.
- FF/NE 각 variant에 대한 탐색(최적화) 함수 1개씩.
- 각 variant에 대한 검증용 matplotlib 플롯 1개씩 (기본 플롯 — 대시보드 수준
  아님).
- 기존 3개 파일(`origami_simulator.py`, `origami_tesellator_1D.py`,
  `origami_helix_inverse_design.py`, `origami_forward_kinematics.py`)은
  **일절 수정하지 않는다** — 새 모듈에서 import해서 재사용만 한다.

**제외 (이번 스펙 범위 아님)**:
- 실제 N개 strip이 옆으로 나란히 배치된 진짜 다각형 단면 3D tube 렌더링
  (지금은 backbone curve/screw 파라미터 검증까지만).
- 임의 N에 대한 자동 스캔/최적 N 탐색.
- Mirror(reflection) 기반 4-cell period 구성 (flat-foldability 불필요 확인
  후 폐기된 접근).
- Actuator나 물리 시뮬레이션(실 시간 folding 동작 애니메이션).

## 3. 아키텍처

새 파일 하나만 추가: **`origami_tube_design.py`** (프로젝트 루트,
기존 4개 파일과 동일 위치).

```
origami_tube_design.py
 ├─ import: OrigamiTesellator1D           (origami_tesellator_1D.py)
 ├─ import: get_se3_frame                 (origami_forward_kinematics.py)
 ├─ import: build_and_analyze,            (origami_helix_inverse_design.py)
 │          compute_AB, compute_fixed_point, classify_AB,
 │          _extract_helix_params
 └─ import: differential_evolution, minimize (scipy.optimize)
```

기존 파일에 대한 수정은 없음 — `get_se3_frame`은 이미 순수 함수
(origin, c_main, c_adj, rho_val, is_base)로 어떤 unit index에도 적용
가능하므로 재사용만 하면 된다.

## 4. 컴포넌트 상세

### 4.1 공통 유틸

```python
def rotation_angle_from_matrix(R: np.ndarray) -> float:
    """R의 회전각 ω [rad] = arccos(clip((trace(R)-1)/2, -1, 1))"""
```

```python
def period_omega(tessellator, idx0: int = 0, idx1: int = 1) -> float | None:
    """
    이미 compute_strip_kinematics()가 실행된 tessellator에서,
    unit idx0 -> unit idx1 사이의 screw 회전각 ω를 계산.
    기본값(idx0=0, idx1=1)은 self.N=1(단일 cell type) 가정 하의 "한 period".
    NE 수렴 검증(4.3)에서는 임의의 (t-1, t) 쌍으로 그대로 호출한다.

    unit t의 프레임 = get_se3_frame(origin, c0, c1, rho_val, is_base=True)
        origin  = global_vertices[t][0]
        c0      = global_vertices[t][CLV[0]] - origin
        c1      = global_vertices[t][CLV[1]] - origin
        rho_val = tessellator.rhos[t][0]
        (origami_forward_kinematics.extract_relative_se3의 is_base=True
         분기와 정확히 동일한 컨벤션 — CLV/rho 인덱스 매핑을 새로 해석하지
         않고 기존 검증된 코드의 패턴을 모든 unit에 균일하게 재사용한다.)

    R_rel = R_idx0.T @ R_idx1  ->  rotation_angle_from_matrix(R_rel)
    valid <= max(idx0,idx1) 이면 None 반환.
    """
```

### 4.2 FF variant

```python
def find_ff_tube_design(
    target_N: int,
    r_target: float,
    g_target: float,
    w_omega: float = 3.0,
    sigma=1, iout=2, num_periods=8, lengths=(1.0,1.0,1.0),
) -> dict:
    """
    미지수 3개: (theta0, theta1, rho0)  [deg]
    alphas = [theta0, theta1, 180-theta0, 180-theta1]  (Kawasaki 자동 만족)

    목적함수:
        res = build_and_analyze(alphas, rho0, sigma, iout, num_periods, lengths)
        (res is None 또는 res["helix"] is None 또는 period_omega(...) is None
         이면 큰 페널티(1e6) 반환 — 기구학적으로 성립하지 않는 구성)
        r, g = res["helix"]["r"], res["helix"]["g"]
        omega = period_omega(res["tessellator"])
        target_omega = 2*pi/target_N
        err = (r-r_target)^2/(r_target^2+eps)
            + (g-g_target)^2
            + w_omega * (omega-target_omega)^2/(target_omega^2+eps)

    탐색: differential_evolution (bounds: theta0,theta1 in (5,85), rho0 in (5,175))
          -> Nelder-Mead 정밀화 (기존 HelixInverseDesigner와 동일 2단계 패턴)

    반환: {"theta0_deg","theta1_deg","rho0_deg","alphas_deg",
           "omega_achieved_deg","r_achieved","g_achieved","residual",
           "tessellator"}
    """
```

검증 플롯:

```python
def plot_ff_tube_transition(result, save_path=None):
    """
    result["alphas_deg"] 고정, rho0를 5deg~175deg로 스윕하며
    각 rho0에서 build_and_analyze + period_omega로 (omega, r, g) 계산.

    3-subplot: omega(rho0) [with target_omega, target_N 표시],
               r(rho0), g(rho0).
    기대: 양 끝(rho0->5, rho0->175)에서 **g -> 0** 에 가까워지고 (torsion 소실,
          평면 상태 — FF strip은 Class I이라 rho_t == rho0 for all t 이므로
          rho0->0/pi는 모든 crease의 fold angle이 0/pi가 되어 전체가 평면이
          되는 것이 대수적으로 보장됨), 찾은 rho0*에서 omega == target_omega
          를 실제로 지나가는지 확인. (omega 자체는 양 끝에서 0으로 갈 필요가
          없음 — 평면 상태에서도 zigzag의 in-plane 회전각은 0이 아닐 수 있음;
          "torsion 없음"은 g=0을 의미하지 omega=0을 의미하지 않는다.)
    """
```

### 4.3 NE variant

```python
def find_ne_tube_design(
    target_N: int,
    r_target: float,
    g_target: float,
    w_omega: float = 3.0,
    w_stable: float = 10.0,
    geom_type="hyperbolic",
    sigma=1, iout=2, num_periods=8, lengths=(1.0,1.0,1.0),
) -> dict:
    """
    미지수 3개: (alpha1(=alpha2), alpha3, alpha4)  [deg]

    목적함수:
        A,B = compute_AB([a1,a1,a3,a4])
        cos_star, rho_star = compute_fixed_point(A,B)
        (rho_star 없으면 큰 페널티)
        res = build_and_analyze(alphas, rho_star, sigma, iout, num_periods, lengths)
        (res is None 또는 res["helix"] is None 또는 period_omega(...) is None
         이면 큰 페널티(1e6) 반환)
        r, g = res["helix"]["r"], res["helix"]["g"]
        omega = period_omega(res["tessellator"])   # rho_star에서의 omega
        target_omega = 2*pi/target_N
        err = (r-r_target)^2/(...) + (g-g_target)^2
            + w_omega*(omega-target_omega)^2/(...)
            + w_stable * max(0, abs(A)-1)^2   # Class II(수렴) 선호, soft penalty

        (기존 inverse_non_euclidean과 달리 rho_star_target을 직접 지정하지
         않는다 — rho*는 alphas로부터 유도되는 결과이고, 대신 omega 조건이
         그 자리를 대체한다.)

    탐색: 기존과 동일 2단계 (differential_evolution -> Nelder-Mead),
          geom_type에 따른 bounds는 기존 inverse_non_euclidean과 동일하게 재사용.

    반환: {"alpha1_deg","alpha3_deg","alpha4_deg","alphas_deg",
           "rho_star_deg","A","B","omega_achieved_deg",
           "r_achieved","g_achieved","residual","tessellator"}
    """
```

검증 플롯:

```python
def plot_ne_tube_convergence(result, rho0_start_deg, num_cells=40, window=8, save_path=None):
    """
    result["alphas_deg"]로 alphas 고정, num_periods=num_cells인 긴 strip을
    rho0_start_deg (result["rho_star_deg"]와 다른 값, target rho*로부터 먼 값)
    에서 시작해 compute_strip_kinematics.

    각 cell t = 1..valid-1:
        omega_t = period_omega(tess, idx0=t-1, idx1=t)  (exact)
        backbone = tess._build_crease_lines()[0]
        r_t, g_t = _extract_helix_params(backbone[max(0,t-window):t+1])  (근사, sliding window)

    3-subplot: omega_t, r_t, g_t 를 cell index t에 대해 플롯하고
    target_omega, r_target, g_target을 수평선으로 표시.
    기대: t가 커질수록 (omega_t, r_t, g_t) -> (target_omega, r_target, g_target)
    으로 수렴 (rho_t가 rho*로 수렴하는 것의 기하학적 결과).
    """
```

## 5. 테스트/검증 계획

코드 자체 테스트(pytest 등)는 만들지 않는다 — 이 프로젝트는 스크립트/노트북
스타일 시뮬레이션 도구이고, 기존 4개 파일도 동일 스타일(각 파일 `__main__`
데모 블록으로 직접 확인). 대신:

- `find_ff_tube_design(target_N=4, r_target=..., g_target=...)` 실행 후
  `plot_ff_tube_transition`으로 육안 확인 — omega가 목표 rho0에서
  90°를 지나는지, residual이 충분히 작은지(예: <1e-3) 콘솔 출력.
- `find_ne_tube_design(target_N=4, ...)` 실행 후 `plot_ne_tube_convergence`로
  수렴 추세 육안 확인 — Class II(|A|<1)인지 콘솔 출력.
- 두 variant 모두 `__main__` 블록에 N=4, 임의 r_target/g_target으로 데모
  실행 예제 포함 (기존 파일들의 `__main__` 패턴과 동일).

## 6. 리스크 / 미해결 질문

- **해 존재성**: 3개 미지수로 3개 목표(r,g,omega=2π/N)를 동시에 맞추는 것이
  항상 가능하다는 보장은 없다 (특히 FF는 r,g 자체가 rho0/alphas에 크게
  의존하므로 목표값에 따라 residual이 크게 남을 수 있음). 이 경우
  `r_target,g_target`을 재조정하며 재탐색하는 것을 1차 대응으로 한다.
- **FF 양 끝단 평면성**: rho0=0,π에서 g->0(torsion 소실)은 Class I(A=1,B=0)
  strip 전체에서 rho_t == rho0 for all t 라는 사실로부터 대수적으로 보장됨
  (rho0->0/pi면 모든 crease가 flat/flat-folded). 반대로 그 구간에서 omega가
  어떤 값으로 수렴하는지는 알파값에 의존하며 보장이 없음 — 목표 omega=2π/N을
  "구간 중간 어딘가에서" 실제로 지나가는지는 `plot_ff_tube_transition`으로
  사후 검증한다 (탐색이 항상 그런 rho0를 찾아낸다는 보장은 없음).
- **NE per-cell r,g 근사**: sliding-window PCA 기반 r_t,g_t는 근사치이며
  window 크기에 따라 흔들릴 수 있음 — 정성적 수렴 추세 확인용이지 정밀한
  수치는 아님.
- **`_extract_helix_params`, `compute_AB` 등 언더스코어(private) 함수를
  모듈 간 import**: Python 관례상 "내부용" 표시이지만 기술적으로는 문제
  없이 동작. 이번 스펙에서는 재구현 대신 그대로 import해 재사용한다.
