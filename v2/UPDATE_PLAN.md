# v2 업데이트 계획과 코드 검토

## 문서 목적과 현재 상태

각 모듈의 기능, 재현 가능한 문제, 개선 방향과 완료 기준을 누적한다.
**1–9절은 변경 전 코드에 대한 검토 이력**이다. 당시의 결함, 행 번호,
미완료 체크리스트는 그 기준 커밋의 상태를 나타내며 현재 구현 상태와 구분한다.
진행 중인 구현과 최종 검증 결과는 문서 끝의 구현 기록에서 정리한다.

- 검토 기준: 39628b5d9ab1aafdd20dec95305d44dc5edf2c51
- 대상: [origami_simulator.py](./origami_simulator.py), 총 694개 내용 행과 마지막 개행
- 재현 환경: Python 3.12.14, NumPy 2.3.5, Matplotlib 3.10.8, Agg backend
- 범위: 전체 소스 정적 검토, 기존 solver/branch 배열의 수치 재현, 별도의 회전 폐합 검증
- 논문의 모든 식을 원문과 대조한 검증이나 실제 GUI 조작·접촉 실험은 수행하지 않았다.
- 아래 행 번호는 검토 기준 커밋의 파일 기준이다.

## 1. origami_simulator.py의 현재 역할

### 1.1 계산 흐름

1. 섹터각 네 개를 받고 각도 합으로 Euclidean / Elliptic / Hyperbolic을 분류한다.
2. 구동 주름을 선택한다. 현재 Elliptic은 원래 e1, 그 외는 원래 e4다.
3. 내부적으로 섹터각 인덱스를 순환 이동해 구동 주름을 내부 e4로 맞춘다.
4. 반각 변수 x=tan(rho_drive/2), y=tan(rho1_internal/2)로 이차방정식을 푼다.
5. 다른 접힘각의 크기를 제곱 관계식으로 구하고, 추가 관계식으로 부호를 복원한다.
6. 구동각을 일정 간격으로 샘플링하며 내부 rho1의 최근접 값으로 두 해를 추적한다.
7. Euclidean은 M/V 패턴으로 다시 분류하고, unwrap 기반 각도 범위 필터를 적용한다.
8. Kinematic map과 샘플 기반 3D slider viewer를 제공한다.

여기서 rho1_internal은 내부 인덱스 기준이다. 특히 Elliptic의 경우
원래 섹터각·접힘각 번호와 혼동하지 않아야 한다.

### 1.2 주요 API

| API | 역할 |
| --- | --- |
| OrigamiDegree4Simulator(...) | 입력, 구동 주름, 허용오차 설정 |
| _solve_quadratic_rho1(...) | 고정 구동각에서 내부 rho1 후보 계산 |
| solve_full_rhos_from_drive_and_rho1(...) | 네 접힘각 복원 후 원래 인덱스로 반환 |
| run_simulation(...) | 전체 샘플링, 해 추적, M/V 분류, 필터링 |
| compute_folding_angles_from_input_output(...) | 특정 구동각과 입출력 부호 조건으로 해 조회 |
| get_3d_geometry(...) | 접힘각에서 네 삼각 패널의 좌표 생성 |
| plot_2d_static(), show_animated_2d_plot() | Kinematic map 표시 |
| show_3d_interactive() | branch/contact 선택과 구동각 slider |

## 2. 먼저 분리해야 하는 개념

현재 코드는 다음 네 가지를 같은 branch 번호로 표현하는 경향이 있다.

1. **대수적 해**: 고정한 구동각에서 나오는 이차방정식의 두 근.
2. **연속 해 경로**: 네 접힘각 공간에서 연속적으로 이어지는 곡선.
3. **M/V 패턴**: 한 상태에서 각 접힘각의 부호를 적은 지역적 라벨.
4. **접촉 제약을 적용한 경로**: 수학적 해 중 물리적 제약을 만족하는 구간.

두 근의 순서가 곧 연속 branch의 정체성은 아니다. 접힘각 하나가 0을
통과하면 같은 연속 경로에서도 M/V 패턴이 달라질 수 있다.
반대로 같은 입출력 부호를 갖는 서로 다른 해도 존재한다.

특이점도 다음을 구분해야 한다.

- **좌표 특이점**: tan(rho/2)가 rho=±pi에서 발산하는 문제.
- **구동각 투영의 특이점**: 전체 해 곡선은 매끄럽지만 선택한 구동각이
  극값을 가져, 다른 각들을 그 구동각의 단일값 함수로 쓸 수 없는 지점.
- **기구학적 특이점**: 전체 회전 폐합 제약의 Jacobian rank가 낮아지는 지점.
  평평한 상태의 branch 만남 등이 여기에 해당할 수 있다.
- **접촉 경계**: 비관통 등 물리적 제약이 바뀌는 경계.
- **수치 불안정**: 상쇄, 고정 절대오차, 잘못된 부호 복원에 따른 계산 실패.

판별식이 0이라는 이유만으로 실제 기구의 자유도가 증가했다고 판단하거나,
M/V 패턴이 2:2라는 이유만으로 singular state라고 판단하면 안 된다.

## 3. 재현 방법과 독립 검증 기준

각도 예시는 모두 원래 입력 순서의 degree 값이다. 함수 내부 계산과
반환 접힘각은 radian이므로 출력 시 변환했다.

회전 폐합을 다음과 같이 검사했다.

    M = Rx(rho1) Rz(alpha1) Rx(rho2) Rz(alpha2)
        Rx(rho3) Rz(alpha3) Rx(rho4) Rz(alpha4)
    closure_error = ||M - I||_F

이 convention은 일반 구동각에서 기존 solver가 반환한 유효 해들과
약 1e-15 수준으로 일치했다. 잘못된 부호 조합을 걸러내는 별도 검증이다.

폐합 Jacobian은 각 Rx 직전의 누적 좌표계에서 본 x축을 열로 갖는
3×4 행렬 J로 평가했다. 회전 폐합 상태에서 J의 rank는 국소 회전 제약의
독립성을 나타낸다. 구동각을 제외한 세 열의 가역성과 전체 J의 rank는
서로 다른 판정이다.

아래 수치는 현재 구현의 관찰값이며, 특히 epsilon·마지막 자릿수·경계점의
반환 개수는 NumPy/플랫폼에 따라 달라질 수 있다.

## 4. 확인된 문제와 케이스 특징

### S01. F가 포함된 정상 branch를 버림 — 재현 확인

- 관련: 146–148, 190–219, 240–241행
- 핵심: M/V 분류 중 패턴에 F가 하나라도 있으면 즉시 continue한다.
- F 판정은 |rho| ≤ 1e-4 rad, 약 0.00573°다.

재현:

    alphas = [60, 60, 120, 120]
    rho_drive = rho4 = 45°

유효 해 중 하나는 대략 다음과 같다.

    (rho1, rho2, rho3, rho4) = (0°, 45°, 0°, 45°)

일부 주름이 움직이지 않는 연속 접힘 모드지만, F가 포함되어 제거된다.
resolution=1000에서 full_math_data는 2000개 상태를 가지지만,
math_branch1은 1000개, math_branch2는 0개다.

또한 [70, 110, 70, 110]은 full_math_data에 2000개 상태가 있으면서도
math_branch1과 math_branch2가 모두 비고, 1000개 상태가 singular_states로
이동한다. 결과적으로 3D viewer는 선택 가능한 branch가 없다고 판단한다.

**케이스 특징:** 대칭 섹터각, 정지한 주름이 있는 모드, 0을 지나는 접힘각,
flat-foldable 구조의 평평한 상태 부근. 단, 모든 대칭 구조에서 같은
실패가 발생한다는 뜻은 아니다.

중요하게도 math_branch1/2는 M/V 재분류 후 만들어지므로,
contact=False로 바꿔도 이미 제거한 branch가 자동 복구되지 않는다.

### S02. ±pi 경계에서 최근접 추적이 다른 해로 넘어감 — 재현 확인

- 관련: 166–177행
- 핵심: 내부 rho1 하나의 일반 차이 abs(rho1_new-rho1_old)만 비교한다.
  원형 각도 거리, 다른 세 접힘각, 경로 접선은 사용하지 않는다.

재현:

    alphas = [60, 90, 120, 150]
    run_simulation(resolution=1000)

양의 구동각 90° 근처에서 math_branch1의 인접 상태가 다음처럼 바뀐다.

    직전: [179.89270, -119.94635,  89.99995, 89.90705]°
    직후: [-32.08558,  120.15450, -22.53766, 90.26740]°

같은 모드는 rho1이 +180°를 지나 -180°로 표시되어야 하지만,
현재 추적은 다른 근으로 갈아탄다. 네 각도의 원형 거리 norm은
약 3.86152 rad이다. 같은 다음 구동각에 존재하는 다른 후보를 연결하면
거리는 약 0.01027 rad에 불과하다.

**케이스 특징:** 출력각이 ±180° seam을 통과함, 두 근이 가까워짐,
선택한 한 각도만으로 두 경로를 구분하기 어려움, 샘플 간 변화가 큰 구간.

### S03. 이차식의 A=0에서 각도 해 하나가 사라짐 — 재현 확인

- 관련: 45–63행
- 반각 이차식의 계수를 이 절에서 A_q, B_q, C_q로 부른다.

현재 |A_q|<1e-9이면 선형식의 유한근 하나만 반환한다.
그러나 y=tan(rho1/2) 좌표의 무한대는 rho1=±180°라는 각도 상태다.

재현:

    alphas = [60, 90, 120, 150]
    rho_drive = 90°
    A_q ≈ 0, B_q ≈ 3, C_q ≈ 0.8660254

현재 반환되는 내부 rho1은 약 -32.20423° 하나다.
그러나 아래 상태도 회전 폐합 오차 약 2.8e-16으로 유효하다.

    [180°, -120°, 90°, 90°]

즉, 이 사례는 실수 유한근의 차수 감소를 각도 공간의 해 소멸로 취급한다.

A_q≈B_q≈0인 경우도 현재는 무조건 빈 리스트로 처리한다.
C_q도 0인 항등식과, C_q≠0인 불가능한 식을 구분하지 않는다.
특히 대칭 구조의 특정 구동각에서는 고정 구동각에 연속적인 해 집합이
남을 수 있어, 최대 두 개의 유한근이라는 표현 자체가 부족하다.

### S04. 구동각 0에서 부호 복원이 잘못됨 — 재현 확인

- 관련: 71–119행, 특히 90–95행

x3의 크기를 먼저 구한 후 P34와 x4의 부호로 x3 부호를 정한다.
x4=0이면 원래 곱 관계가 x3의 부호를 정하지 못하는데,
현재는 sign(P34)를 쓰고 0이면 +1을 강제한다.
x1=0일 때 x2 부호를 복원하는 대체식에도 비슷한 퇴화 가능성이 있다.

재현:

    alphas = [70, 70, 120, 110]
    rho_drive = 0°

현재 후보 하나는 다음과 같다.

    [-38.69026°, 30.63681°, +38.69026°, 0°]

이 조합의 회전 폐합 오차는 약 1.76808이다.
반면 구동각을 +0.000001°로 주면 대응 후보는 다음과 같다.

    [-38.69026°, 30.63681°, -38.69026°, 0.000001°]

이때 폐합 오차는 약 7.7e-16이다. rho3의 부호가 핵심 차이다.
다른 한 후보는 구동각 0에서도 유효하므로, 실패를 입력 전체의
기구학적 불가능성으로 처리해서는 안 된다.

**케이스 특징:** 구동각 또는 복원에 쓰는 인접 반각 변수가 0,
곱 관계가 부호 정보를 잃는 상태, 극소량의 부호·반올림에 의해
선택이 달라지는 상태.

기본 resolution=1000의 대칭 격자는 정확한 0을 포함하지 않아
이 문제를 놓치기 쉽다. 홀수 resolution 또는 직접 0을 조회하면 노출된다.

### S05. 평평한 상태 근처의 상쇄와 허용오차 혼용 — 재현 확인 + 코드 검토

- 관련: 41–42, 50–62, 71–83, 90–117, 146–148행

같은 크기의 cos 항을 빼서 매우 작은 계수를 계산하고, 보통의
이차방정식 공식을 사용한다. tan(rho/2)는 ±pi 근처에서 매우 커진다.
그 결과 작은 각도에서는 유효 숫자를 잃고, 큰 반각 변수에서는
큰 항 사이의 상쇄가 커질 수 있다.

재현:

    alphas = [70, 20, 110, 160]  # flat-foldable
    rho_drive = 0.000001°

이 계열에서 안정적인 삼각항등식으로 얻는 반각 기울기는 다음과 같다.

    k± = (sin(theta0) ± sin(theta1)) / sin(theta0+theta1)
    tan(rho1/2) = k± tan(rho_drive/2)

따라서 기대 rho1은 약 [0.000000597672, 0.000001281713]°다.
현재 반환값은 약 [0, 0.000001879385]°로 두 branch의 기울기가 왜곡된다.
절대각 오차는 작지만 branch 접선이나 수렴률을 판단할 때 상대오차는 크다.

허용오차도 다음처럼 역할별 단위·스케일이 다르다.

- 계수·판별식·반각변수 판정: 1e-9
- 복원식 분모: 1e-15
- 음수인 제곱값 허용: -1e-5
- M/V의 F 판정: 접힘각 1e-4 rad

이들은 동일한 오차 기준이 아니며, 식의 스케일에 따른 정규화도 없다.
작은 음수 제곱값은 크기 계산에서 0으로 잘리지만 부호 관계식에는
원래 음수값이 사용되는 점도 일관성 점검이 필요하다.

### S06. 판별식 0을 실제 기구학적 분기로 단정할 수 없음 — 재현 확인

- 관련: 58–63, 152–177행

재현:

    alphas = [40, 80, 50, 110]
    구동 주름 = 원래 e1
    rho_drive ≈ 50.9192574381°

경계에서 두 근은 같은 상태로 합쳐지지만, 현재는 중복 후보 두 개를 반환한다.

    [50.919257°, 107.782669°, 0°, 139.355926°]

구동각을 0.000001° 낮추면 해가 없고, 같은 양만큼 높이면 서로 다른 두 해가
나온다. 이 근처에서는 고정 간격 구동각 샘플링의 민감도가 커진다.

중요한 판정:

- 전체 3×4 폐합 Jacobian의 특이값은 약 [1.47824, 1.25801, 0.48188].
  즉 rank는 3이며, 전체 해 집합은 이 상태에서 국소적으로 매끄러운 1차원이다.
- 구동각을 제외한 3×3 부분 Jacobian의 determinant는 약 -2.85e-17.
  이 구동각으로 나머지 각도를 함수화하는 방식이 퇴화한다.

**케이스 특징:** 구동각의 극값, 같은 입력에 두 출력이 생기는 접힘 경계.
이 상태를 계속 따라가려면 구동각의 진행 방향을 되돌리거나,
다른 주름을 구동 좌표로 선택하거나, 호길이 continuation을 써야 한다.

반대로 전체 Jacobian rank가 낮아지는 진짜 branch 만남에서는
이전 경로·접선·별도 branch 선택 규칙이 필요하다.
단순히 resolution을 올리는 것으로 유일한 경로가 정해지지는 않는다.

### S07. singular_states가 실제 singularity를 나타내지 않음 — 재현 확인

- 관련: 190–219행

현재 singular_states에는 F가 없는 M/V 패턴 중 3:1 또는 1:3으로
분류되지 않은 상태가 들어간다. 판별식, 분모, Jacobian rank로 판정하지 않는다.
F가 있는 상태는 오히려 이 목록에도 들어가지 않고 제거된다.
Non-Euclidean에서는 이 목록을 생성하는 별도 처리도 없다.

재현:

    alphas = [60, 90, 135, 75]
    resolution = 1000

singular_states에 들어간 한 상태:

    [-128.278011°, 117.526847°, 156.328557°, -148.643917°]
    M/V = MVVM
    closure_error ≈ 4.7e-16
    판별식 ≈ 442.4084
    전체 Jacobian 최소 특이값 ≈ 0.6170

이 상태는 전체 폐합 제약에서 regular한 수학적 해다.
접촉 허용 여부는 별도 검사 대상이며, 이 수치만으로 물리적 비관통까지
입증한 것은 아니다. 다만 M/V 미분류를 수학적 특이점으로 부르는 것은 부정확하다.

### S08. 해 조회 함수가 배열 오류를 숨김 — 재현 확인

- 관련: 259–260, 298–306행

배열은 다음과 같이 만든다.

    output_angles_rad.shape == (n,)
    folding_angles_all.shape == (n, 4)

그러나 저장은 각각 [i,j], [i,j,:]로 시도한다. 해가 한 개만 있어도
첫 번째 대입에서 IndexError가 발생하며, bare except가 이를 숨긴다.

재현:

    alphas = [70, 20, 110, 160]
    input_angle_deg = 45
    output_index = 0
    sign = +1

실제 두 해의 rho1은 약 27.809617°, 55.927916°로 모두 입력과 같은 부호다.
그런데 API는 output_angles=NaN, folding_angles_all=NaN, num_solutions=0을
반환한다.

따라서 sign=±1은 일반적으로 두 branch를 고유하게 구분하는 키도 아니다.
또한 해당 API의 input_angle_deg는 임의로 지정한 입력 주름이 아니라,
생성자에서 선택한 구동 주름의 각도다.

### S09. 한 branch만 계산되면 run_simulation이 중단됨 — 재현 확인

- 관련: 54–56, 175–177, 187행

재현:

    OrigamiDegree4Simulator([90,90,90,90], verbose=False).run_simulation()

이 대칭 사례는 비영 구동각에서 주로 유한근 하나만 반환한다.
raw_m1에는 상태가 있지만 raw_m2는 비어 있다.
np.vstack([raw_m1,raw_m2])가 (N,4)와 빈 (0,) 배열을 결합하려다
ValueError를 발생시킨다.

추적 코드도 한 근만 남으면 무조건 m_idx=0으로 넣고,
prev_r1_m2를 갱신하지 않는다. 해가 다시 두 개가 되는 경계에서
이전 두 branch의 정체성을 보존할 보장이 없다.

## 5. 표시·접촉 처리에서 추가로 생길 수 있는 혼동

아래는 코드 검토로 확인한 처리 방식이며 실제 GUI 이벤트는 별도 검증 대상이다.

1. **단절 구간이 다시 연결된다.**
   _unwrap_filter는 블록을 구분하지만 마지막에 vstack으로 붙이고,
   viewer는 구동각으로 정렬한 뒤 한 선으로 그린다.
   [40,80,50,110]처럼 가운데 구동 불가능 구간이 있어도 이를 가로지르는 선이
   생길 수 있다. 구간 ID나 NaN 분리 표지가 필요하다.
2. **slider가 현재 상태의 연속성을 추적하지 않는다.**
   매번 구동각이 가장 가까운 샘플을 고른다. 유효 구간의 전체 min/max만
   검사하므로 내부 공백이나 여러 연결 성분을 구분하지 못한다.
3. **singular 표시가 휴리스틱이다.**
   구동각 범위 6 rad와 거리 0.05 rad 등 고정값을 사용한다.
   두 후보와의 거리를 계산하지만 실제 전환 조건은 singular 샘플과
   target 사이 거리가 0.05 미만인지 여부다.
4. **3D 그림이 각도 tuple의 오류를 가릴 수 있다.**
   get_3d_geometry는 rho1과 rho4로 좌표를 만들고 두 v3 후보의
   섹터각 오차를 비교한다. 입력 rho2, rho3와 최종 형상이 일치하는지
   검증하지 않는다. 두 후보 오차가 1e-6 이내면 rho1의 부호로 선택한다.
5. **contact는 완전한 비관통 검사가 아니다.**
   주로 unwrap 경로의 각도 범위를 제한한다. 패널-패널 교차 검사가 없으므로
   contact=True 결과를 일반적인 충돌 검증 완료 상태로 해석하면 안 된다.
6. **거의 Euclidean인 입력의 분류가 불명확하다.**
   np.isclose의 기본 상대오차 때문에 각도 합이 360.001°인 입력도 Euclidean으로
   분류된다. 이 허용오차와 solver의 식·M/V 분류 정책을 명시해야 한다.
7. **초기 상태와 입력 검증이 부족하다.**
   생성자에는 math_branch1/2가 초기화되어 있지 않아 run_simulation 전에
   viewer를 부르면 AttributeError가 날 수 있다. 섹터각 개수·유한성·허용범위,
   sin(alpha) 분모의 퇴화 입력도 명시적인 검증이 필요하다.

## 6. 업데이트 우선순위와 완료 기준

### 우선순위 1 — 기본 반환과 유효 해 보존

- [ ] 조회 API를 다중 해를 담을 수 있는 일관된 자료구조로 정의한다.
- [ ] bare except를 없애고 계산 불가능·좌표 퇴화·구현 오류를 구분한다.
- [ ] 0/1/2개 후보와 중복근, 비고립 해 집합을 별도 상태로 표현한다.
- [ ] M/V와 contact 처리가 원본 해와 경로를 삭제하지 않도록 분리한다.
- [ ] 복원한 네 접힘각에 독립적인 회전 폐합 잔차 검증을 적용한다.

완료 기준: S01, S04, S08, S09 사례에서 해가 조용히 사라지거나
오류 tuple이 정상 해로 반환되지 않는다.

### 우선순위 2 — solver의 좌표·수치 안정성

- [ ] 이차식 정규화, 안정적인 근 계산, 중복근 판정을 도입한다.
- [ ] ±pi에서 반각 무한대를 표현하거나 대체 좌표 chart로 전환한다.
- [ ] A_q=B_q=0에서 항등식과 불가능한 식을 분리한다.
- [ ] 제곱 관계의 0/0, 분모 0과 부호 정보 소실을 다른 식·극한·폐합 제약으로 처리한다.
- [ ] flat-foldable 특수형과 near-flat 삼각항등식을 검토한다.
- [ ] 계수, 판별식, 각도, 잔차의 허용오차를 역할별로 정의한다.

완료 기준: S03, S04, S05에서 유한 각도 상태를 보존하고,
epsilon 크기를 바꿔도 해와 branch 접선이 타당하게 수렴한다.

### 우선순위 3 — branch와 singularity의 정의·추적

- [ ] 네 각도의 원형 거리와 예측 접선을 사용해 경로를 추적한다.
- [ ] 구동각 투영 경계에서 adaptive sampling / 구동 주름 전환 /
      pseudo-arclength continuation 중 적합한 방법을 결정한다.
- [ ] 실제 Jacobian rank 저하, 투영 경계, ±pi seam, 접촉 경계를 별도 분류한다.
- [ ] M/V는 상태의 부가 라벨로 저장하며 branch ID와 분리한다.
- [ ] 실제 분기점에서 가능한 후속 경로와 사용자 선택 규칙을 명시한다.

완료 기준: S02의 수치 점프가 사라지고 S06과 S07을 서로 다른 종류로
판정한다. 평평한 branch 교차에서는 유일성을 임의로 가정하지 않는다.

### 우선순위 4 — viewer와 physical bound

- [ ] 연결 성분과 공백을 보존해서 그래프를 그린다.
- [ ] slider 이동 시 선택 branch와 허용 구간을 유지한다.
- [ ] 3D geometry가 네 접힘각 및 폐합 제약과 일치하는지 검증한다.
- [ ] contact의 정확한 범위를 정하고 각도 한계와 실제 충돌을 구분해 표시한다.
- [ ] 비어 있는 branch, 실행 전 호출, 잘못된 입력에 대한 동작을 정한다.

완료 기준: 계산된 해의 연결 관계와 표시된 선·3D 상태가 일치한다.

## 7. 최소 재현 예시

현재 파일을 수정하지 않고 v2 디렉토리에서 실행할 수 있다.

    import numpy as np
    from origami_simulator import OrigamiDegree4Simulator

    def states_at(sim, deg):
        drive = np.deg2rad(deg)
        return [
            sim.solve_full_rhos_from_drive_and_rho1(drive, r1)
            for r1 in sim._solve_quadratic_rho1(drive)
        ]

    # S01: 대칭 구조의 한 모드가 분류 중 사라짐
    sim = OrigamiDegree4Simulator([60,60,120,120], verbose=False)
    print([np.rad2deg(r) for r in states_at(sim,45) if r is not None])
    sim.run_simulation(1000)
    print(len(sim.full_math_data), len(sim.math_branch1), len(sim.math_branch2))
    # 관찰: 2000 1000 0

    # S04: 정확한 0과 극소 양의 구동각에서 rho3 부호 비교
    sim = OrigamiDegree4Simulator([70,70,120,110], verbose=False)
    for d in [0, 1e-6]:
        print(d, [np.rad2deg(r) for r in states_at(sim,d) if r is not None])

    # S08: 두 유효 해가 있어도 API는 NaN/0 반환
    sim = OrigamiDegree4Simulator([70,20,110,160], verbose=False)
    print(sim.compute_folding_angles_from_input_output(45,0,sign=1))

    # S09: ValueError 재현
    sim = OrigamiDegree4Simulator([90,90,90,90], verbose=False)
    sim.run_simulation()

폐합 오차 검사는 다음과 같이 독립적으로 재현할 수 있다.

    def rx(t):
        c, s = np.cos(t), np.sin(t)
        return np.array([[1,0,0],[0,c,-s],[0,s,c]])

    def rz(t):
        c, s = np.cos(t), np.sin(t)
        return np.array([[c,-s,0],[s,c,0],[0,0,1]])

    def closure_error(alphas_deg, rhos):
        M = np.eye(3)
        for a, r in zip(np.deg2rad(alphas_deg), rhos):
            M = M @ rx(r) @ rz(a)
        return np.linalg.norm(M - np.eye(3))

    sim = OrigamiDegree4Simulator([70,70,120,110], verbose=False)
    for rhos in states_at(sim,0):
        if rhos is not None:
            print(np.rad2deg(rhos), closure_error(sim.original_alphas_deg,rhos))
    # 관찰: 후보 하나는 약 1.76808, 다른 후보는 약 8.4e-16

## 8. 나머지 모듈

- origami_tesellator_1D.py: 상세 검토 예정.
- origami_forward_kinematics.py: 상세 검토 예정.

이 문서의 개선 항목은 제안이며 아직 구현 완료를 뜻하지 않는다.


## 9. 선행 문헌과 branch 분류 방법 비교

### 9.1 논의 범위와 결론

전체 kinematic space의 샘플이 이미 계산된다는 전제에서,
이 절은 **그 상태들을 사용자가 정의한 구동 경로와 도달 구간으로
완전하게 분류할 수 있는가**를 검토한다. solver를 교체하는 제안과 구분한다.

결론은 다음과 같다.

- 일반적인 Euclidean 물리적 branch의 minority crease 분류는 이론적 근거가 있다.
- 전체 입력을 developable / elliptic / hyperbolic 셋으로만 나누고,
  구동 주름을 e1/e4로 고정하고, 두 개의 최근접 추적 배열만 유지하는
  현재 조합으로는 특수 기하와 모든 경계를 포함한 완전 분리를 보장할 수 없다.
- 기존 추적 틀은 유지하되, 기하의 특수 조건·번호 정규화·안정적인
  branch 명명·구간 연결 관계를 보완하는 방향이 적합하다.

여기서 branch는 지정한 시작 상태에서 선택한 구동각을 변화시킬 때
같은 접힘 경로로 도달할 수 있는 구간을 뜻한다.
문헌의 folding mode와 실제 구현의 구동각 구간은 별도 필드로 관리한다.

### 9.2 확인한 1차 문헌

1. **Foschi, Hull & Ku (2022)**,
   Explicit kinematic equations for degree-4 rigid origami vertices,
   Euclidean and non-Euclidean, Physical Review E 106, 055001.
   [논문](https://arxiv.org/pdf/2206.12691)
   [DOI](https://doi.org/10.1103/PhysRevE.106.055001)
   - 확인 위치: Theorem 1, Corollary 1, Remark 1–2, Figure 3, Appendix C.
   - 일반 관계식에서 개별 각도의 함수와 folding mode를 얻는 과정에는
     제곱근 부호의 일관된 선택이 필요하다.
   - Remark 2는 전 구간을 움직일 수 있는 구동 주름 선택을 권고한다.
     예시 bird's-foot에서는 섹터각 조건을 명시하고 elliptic의 rho2를 선택한다.
     따라서 논문이 elliptic의 원래 e1, hyperbolic의 원래 e4를
     임의 입력에 보편적으로 지정한 것은 아니다.
   - Corollary 1의 두 folding-mode 곡선 설명을, 특수 퇴화점과 경계의
     모든 움직임이 코드의 두 배열에 단일값 함수로 들어간다는 보장으로
     해석하지 않는다. branch라는 용어의 집계 단위도 맞춰야 한다.

2. **Waitukaitis & van Hecke (2016)**,
   Origami building blocks: Generic and special four-vertices,
   Physical Review E 93, 023003.
   [논문](https://ir.amolf.nl/pub/6626/16107OA.pdf)
   [DOI](https://doi.org/10.1103/PhysRevE.93.023003)
   - 확인 위치: Section II.A, Eq. (1), Section III.B, Figures 8–9.
   - 비자기교차 Euclidean 기하의 일반적인 경우, unique fold가
     minority crease에 해당한다. 가능한 unique fold는 섹터각 부등식
     alpha_(i-1)+alpha_i < alpha_(i+1)+alpha_(i+2)로 판정한다.
   - generic 3유형과 각 subtype 외에 special 16유형을 구분한다.
     collinear branch에는 계속 0인 null fold가 있다.
   - 완전접힘 경계에서 생기는 추가 움직임을 포함하면 CFF는 4개,
     2CFF는 6개 branch로 기술한다. 이 문헌의 branch 계수 방식과
     현재 UI의 도달 구간 계수 방식은 대응 관계를 명시해야 한다.

3. **Waitukaitis, Dieleman & van Hecke (2020)**,
   Non-Euclidean origami, Physical Review E 102, 031001(R).
   [논문](https://ir.amolf.nl/pub/10313/16648M_Waitukaitis-S..pdf)
   [DOI](https://doi.org/10.1103/PhysRevE.102.031001)
   - 확인 위치: Figure 2, Table I, branch splitting 및 단조성 설명.
   - 주로 Euclidean에서 작은 각도 surplus를 준 generic 기하를 다루며,
     unique fold와 주름 번호를 먼저 정렬한다.
   - 해당 번호 convention에서 elliptic은 A/C, hyperbolic은 B/D이며,
     한 branch 안에서도 M/V 패턴이 바뀐다.
     A–D는 두 부호의 surplus를 합친 명칭으로, 한 기하에 항상 4개가
     공존한다는 의미가 아니다.
   - 위 조건의 generic 기하에서 elliptic은 unique fold,
     hyperbolic은 non-unique fold가 부호를 바꾸는 구동 좌표가 된다.
     임의 주름 쌍의 관계는 비단조일 수 있다.
     이 규칙의 적용 범위를 큰 surplus나 대칭 퇴화 기하 전체로
     증명 없이 확대하지 않는다.

4. **Hull (2025)**,
   A Rigid Origami Elliptic-Hyperbolic Vertex Duality,
   Results in Mathematics 80, 138.
   [논문](https://link.springer.com/article/10.1007/s00025-025-02457-8)
   - 확인 위치: Definition 1 및 Theorem 3의 증명.
   - alpha_i를 pi-alpha_i로 바꾼 dual 기하의 운동학은
     접힘각의 교대 부호 변환과 대응한다.
   - 수학적 기구학과 branch 대응 검증에 사용할 수 있다.
     원래 패널의 물리적 접촉 허용 여부는 별도 확인한다.

논문마다 crease/sector 인덱스와 M/V 부호 convention이 다를 수 있으므로,
직접적인 패턴·주름 번호 비교에는 변환표가 필요하다.

### 9.3 현재 규칙을 어떻게 개선할 것인가

#### A. 각도 합 외에 특수 기하 조건을 저장한다

입력의 순환 순서를 유지하면서 다음 특징을 계산한다.

- Euclidean 여부와 angular surplus
- flat-foldability 조건
- 인접·대향 섹터각의 동일성
- 연속한 섹터각 합이 pi가 되는 collinearity
- 두 개 이상의 조건이 겹치는 퇴화 기하

예를 들어 [60,60,120,120]은 collinear 및 flat-foldable 조건이 겹치고,
[90,90,90,90]에는 두 collinear 주름 쌍이 있다.
이들은 일반적인 M/V 두 그룹 규칙의 별도 처리 대상이다.

부동소수점 오차로 거의 같은 것과 정확한 기하 조건을 구분한다.
near_euclidean / near_collinear 같은 수치 경고를 두되,
허용오차 하나로 실제 위상 분류를 자동 변경하지 않도록 명세한다.

#### B. generic Euclidean의 minority 규칙은 유지하되 명명 방식을 바꾼다

현재 샘플 수 상위 두 그룹을 branch 1/2로 삼는 대신,
허용되는 unique crease를 각도 부등식에서 먼저 구하고
원래 crease index를 포함한 안정적인 이름을 붙인다.

    예: E_unique_e2, E_unique_e3

샘플 개수나 순회 방향이 바뀌어도 물리적으로 같은 branch의 이름이
바뀌지 않아야 한다. 이 부등식은 strict generic 조건에서 사용하고,
등호가 되는 특수 기하를 일반 규칙으로 밀어 넣지 않는다.

F가 포함된 상태와 collinear mode는 독립적으로 보존한다.
generic Euclidean에서 2:2 패턴이 나오는 상태는 우선 수학적
자기교차 해인지 확인하며, 이를 singularity라고 곧바로 부르지 않는다.
앞선 S07의 regular 판정은 폐합 Jacobian에 관한 것이며
물리적 비관통을 보장한다는 뜻이 아니다.

#### C. 번호를 정규화하거나 주름 선택을 실제 기하에서 판정한다

논문 속 e1/e4는 번호 convention을 전제로 한다.
현재 입력을 순환 이동하면 같은 기하라도 구동 주름의 역할이 바뀐다.

- 원래 주름과 내부 주름의 대응을 명시적으로 유지한다.
- 사용자가 구동 주름을 고르면 해당 구동각으로 가능한 구간을 분리한다.
- 자동 선택 모드라면 실제 기하의 단조성·도달 범위를 평가한다.
- 문헌의 unique/non-unique 규칙은 그 조건이 확인된 범위에서 활용한다.

구동 주름 전환은 자동으로 사용자 branch를 연장하는 수단으로 쓰지 않는다.
다른 좌표를 통한 경계 검증 또는 전체 공간의 보조 표현으로 사용할 수 있다.

#### D. M/V는 branch의 불변 특징과 변화 특징을 구분해 기록한다

generic Euclidean에서는 unique crease index를 불변 라벨로 활용할 수 있다.
non-Euclidean에서는 한 branch의 M/V 변화 이력이 정상적인 정보다.

2020 논문의 정렬과 적용 조건이 만족되는 경우,
elliptic의 non-unique fold 부호나 hyperbolic의 unique fold 부호처럼
경로에서 유지되는 특징을 seed의 교차 검증에 활용할 수 있다.
그 밖의 기하에서는 이 부호 불변성을 직접 확인해야 한다.

접힘각 전체의 연속성에 의해 같은 branch를 추적하고,
M/V 변화는 상태 이벤트로 기록한다. M/V 재계산만으로 branch ID를 바꾸지 않는다.

#### E. 전체 모드와 구동 가능한 구간을 연결 관계로 관리한다

다음 개념을 나눠 저장한다.

    문헌의 folding mode
        └─ 선택 구동각의 단조 구간
            └─ 정의역·접촉 정책에서 시작 상태로부터 도달 가능한 구간

구동각 극값, 정의역 끝, 완전접힘 접촉, 특수 기하의 새 움직임을
명시적인 경계로 기록한다. 같은 점에서 만났다는 이유만으로
모든 후속 구간을 자동으로 같은 branch에 합치지 않는다.

일반 구간에서는 기존의 최근접 추적을 사용할 수 있다.
다만 전체 접힘각·이전 접선·연결 구간 정보를 사용하고,
각도 seam에서는 정의역과 접촉 정책을 적용한다.

경계 자체에서 후보가 합쳐지면, 경계에서 약간 안쪽의 서로 다른 상태를
seed로 사용해 출발 경로를 구분할 수 있다.
고정 구동각에서 다른 각도가 연속적으로 변하는 특수 모드는
구동 범위가 한 점으로 퇴화하는 별도 움직임으로 기록한다.

### 9.4 검증할 항목

이는 향후 구현의 검증 계획이며 이 절에서 Python 코드를 변경한 것은 아니다.

- [ ] 논문의 generic Euclidean 유형·subtype과 special 유형을 사례표로 만든다.
- [ ] 기준 non-Euclidean 예제의 정렬·부호 convention과 A/C, B/D 연결을 비교한다.
- [ ] 섹터각 순환 이동 후 crease 번호를 되돌리면 같은 branch 구간이 나오는지 확인한다.
- [ ] 전체 접힘각 반전 시 대응 branch와 끝점이 일관되게 매핑되는지 확인한다.
- [ ] 적합한 도메인에서 elliptic–hyperbolic duality를 검증한다.
- [ ] 샘플 개수·순회 방향을 바꿔도 ID와 연결 관계가 유지되는지 확인한다.
- [ ] collinear mode, 완전접힘 경계, 단조성 경계에서 해를 잃거나 잘못 연결하지 않는지 확인한다.
- [ ] 수학적 해의 전체성과 물리적 허용 구간의 전체성을 별도 기준으로 평가한다.

문헌과의 정성적 일치 및 유한 개 사례 통과는 전체 입력에 대한
수학적 완전성 증명과 구분한다. 전체 case를 보장하려면 입력 섹터각 범위,
특수 기하, 각도 끝점 포함 여부, 자기교차 정책을 명세한 다음
각 분류·경계에 대한 처리의 전체성을 입증해야 한다.

## 10. v2 구현 기록

이 절은 변경 후 구현을 설명한다. 이전 절의 재현값과 행 번호는 변경 전
기준선이며, 실제 수정 파일·검증 범위와 남은 한계는 이 절에서 구분한다.
수학적 관계식의 유효성과 패널 비관통의 유효성은 별개다.

### 10.1 모듈 구성과 입력 범위

- origami_kinematics.py: 기하 분류, 순수 후보 해 계산, 특수형 규칙,
  폐합 잔차 및 fixed-drive family.
- origami_simulator.py: 공개 API, branch 연결, 구간·경계와 세션 상태.
- origami_viewer.py: Matplotlib 표시와 위젯 콜백. 경로 선택·제약의
  최종 결정은 모델 세션에 위임한다.
- origami_tesellator_1D.py / origami_forward_kinematics.py:
  기존 호출부의 호환성을 검사하며 독립적인 기능 확장은 이번 범위에서 제외한다.

기하 입력은 유한한 degree 섹터각 네 개이며, 각각 0°보다 크고 360°보다
작아야 한다. 180° 섹터는 기하 정보 자체는 반환하지만 현재의 패널 법선
복원식이 퇴화하므로 UNSUPPORTED_DEGENERATE_SECTOR로 명시한다.
이를 해가 없는 정상 기하로 처리하지 않는다.
모든 reflex-sector 기하에 대한 완전성은 입증하지 않았다.
180°보다 큰 섹터는 방향 있는 회전 관계식으로 계산하지만, 삼각형 네 개로
보여주는 기존 패널 패치는 reflex material의 전체 내부 영역을 표현하지 않는다.
이 표시를 실제 재료의 자기교차 검증으로 해석하지 않는다.

공개 driving_crease는 원래 입력의 e1–e4, 즉 1부터 4까지다.
생략하면 기존 호환 동작인 elliptic=e1, 그 외=e4를 유지한다.
명시적으로 지정한 구동 주름은 추적 중 자동으로 변경하지 않는다.
원래 입력 배열은 복사해 보존하고 모델의 각도·구동 주름은 읽기 전용으로
관리한다. 다른 기하를 계산하려면 새 simulator를 만든다.

Python 3.10 이상을 사용하며, 이번 실행은 Python 3.12에서 확인했다.
의존성은 requirements.txt에 기록했다.

    python -m pip install -r v2/requirements.txt

### 10.2 기하 분류

geometry에는 다음 정보가 들어간다.

- case: developable / elliptic / hyperbolic
- angular_surplus_deg
- flat_foldable: 일반화한 Kawasaki 조건인 교대 섹터각 합의 동일성
- euclidean_flat_foldable: developable이면서 flat_foldable
- collinear 및 collinear_pairs
- rotations / reflections: 섹터각 및 주름 번호의 대응 permutation
- rotational_symmetry / reflection_symmetry
- equal_angle_pairs
- unique_folds: developable에서 각도 부등식으로 가능한 unique crease
- special: 특수 조건의 존재
- near: 거의 Euclidean·flat-foldable·collinear·동일각인 수치 상태
- degenerate_sectors: 180° 섹터의 원래 번호

입력 순서는 보존한다. 반사는 섹터각과 주름의 permutation이 한 칸
다르므로 두 map을 별도로 저장한다.

정확 조건의 수치 허용오차는 입력 크기에 비례한 64 machine-epsilon,
near 조건은 0.001°다. near 판정만으로 기하 유형을 바꾸지 않는다.
이는 유한 정밀도 입력에서 사용하는 명시적 정책이며 수학적 상징 연산은 아니다.

일반화한 flat-foldable은 alpha1+alpha3=alpha2+alpha4다.
예를 들어 [110,110,120,120]은 hyperbolic이면서 이 조건을 만족한다.
이 경우와 Euclidean의 반대각 보완 조건을 혼동하지 않는다.

### 10.3 일반 solver와 특수형 식

#### 일반형의 반각 관계식

원래 Foschi 이차식은 유지한다. x=tan(d/2), y=tan(r1/2) 대신
s=sin(d/2), c=cos(d/2), p=sin(r1/2), q=cos(r1/2)를 사용하여

    A*p² + B*p*q + C*q² = 0

으로 계산한다. 여기서 내부 구동 주름은 e4이고 내부 섹터각은 a1–a4다.

    A = s²[cos(a1+a3-a4)-cos(a2)]
        + c²[cos(a1-a3-a4)-cos(a2)]
    B = 4*s*c*sin(a1)*sin(a3)
    C = c²[cos(a1+a3+a4)-cos(a2)]
        + s²[cos(a1-a3+a4)-cos(a2)]

작은 cos 차이는 삼각항등식으로 계산하고, 가능한 부분은 longdouble을
사용한다. 계수를 정규화한 뒤 안정적인 이차근 표현을 사용한다.
q=0은 r1=pi라는 유한 각도 상태로 보존하며, 중복근은 하나로 합친다.

다른 두 접힘각의 부호는 제곱값의 부호 추정으로 정하지 않는다.
기존 두 패널을 배치한 뒤 방향 있는 패널 법선 사이의 atan2로 복원한다.
모든 후보에 독립적인 SO(3) 폐합 검사를 적용하고 residual을 저장한다.
일반 허용 잔차는 2e-8이며, 잔차를 만족하지 않는 값을 정상 해로 반환하지 않는다.
생성자의 residual_tol로 유한한 양의 잔차 허용값을 지정할 수 있다.
렌더링도 폐합을 만족하지 않는 입력 tuple을 정상적인 패널로 표시하지 않는다.

#### Euclidean flat-foldable의 별도 규칙

내부 섹터각이 (u,v,pi-u,pi-v)이면 일반식에 직접 대입해

    sin(u+v)*y² - 2*sin(u)*x*y + sin(u-v)*x² = 0

를 얻는다. 보통은 두 반각 기울기로 분해되지만, 기울기의 분모가 0인
경우까지 유지하기 위해 다음 homogeneous factor를 직접 사용한다.

    cos((u+v)/2)*p*c - cos((u-v)/2)*q*s = 0
    sin((u+v)/2)*p*c - sin((u-v)/2)*q*s = 0

일반화한 non-Euclidean flat-foldable에는 이 Euclidean 전용 식을
잘못 적용하지 않고 일반 관계식을 사용한다.

#### 특수형의 fixed-drive family

d=0 또는 pi를 일반식에 먼저 대입해 B=0으로 만들고, 남은 A=C=0의
기하각 항등식을 확인하면 고정 구동각에서 내부 r1이 자유로운 family로
표현한다. 이는 Euclidean collinear뿐 아니라 연속한 섹터각 두 쌍이 같은
일반화 bird's-foot 등의 경우도 포함한다. 작은 계수라는 이유만으로
family를 만들지 않고, 입력 기하의 삼각항등식이 성립하는지를 검사한다.

각 family는 다음을 제공한다.

- fixed_drive
- free_crease: 원래 입력 번호
- mode_id
- valid_intervals
- evaluate(free_angle): 네 접힘각과 폐합 잔차

epsilon으로 기하를 바꾸지 않는다. family를 임의의 두 표본으로 축소하지 않는다.
다음 식은 해당 입력의 일반 관계식에서 얻는 재현 가능한 예다.

    [60,60,120,120], d=rho4=pi:
        rhos(t) = [t, pi, -t, pi]
    [70,110,70,110], d=rho4=0:
        rhos(t) = [t, 0, t, 0]
    [90,90,90,90], rho1=rho3=pi:
        rhos(t) = [pi, t, pi, -t]
    [110,110,120,120], d=rho4=pi:
        rhos(t) = [t, pi, -t, pi]

앞의 세 예는 t∈[-pi,pi]의 101개 값에 대해 독립 회전 폐합 오차가 5e-16
이하였다. 일반화 bird's-foot를 포함한 evaluator의 별도 표본도 독립 폐합
검증을 수행한다. 이는 명시한 family의 검증이며 특수 기하 전체의 분류 증명은 아니다.

대칭성은 있으나 위 식의 별도 factor나 collinear family로 처리하지 않은
입력은 symmetric_general_fallback 규칙으로 기록한다.
이 경우에도 후보의 폐합을 검사하되, 모든 대칭 특수형에 대한 독립적인
닫힌 형식의 모드 규칙을 완성했다고 주장하지 않는다.


### 10.4 Branch, seed와 도달 구간

Branch는 id, mode_id, states, seed_state, segments, endpoints, family,
physical_segments, seed_checks, reachable_intervals를 제공한다.
수학적 모드와 구동 가능한 연결 성분을 분리한다.

일반 developable은 후보에 부여된 unique crease로 E_unique_eN을 사용한다.
Flat-foldable의 factor는 ff_plus / ff_minus, collinear에서는 null fold 모드와
나머지를 구분한다. 나머지는 입력의 순환 순서에서 정한 canonical 상태와
구동 성분에 따른 mode 번호를 사용한다. 이는 전역적인 기하 파라미터 변화에
대한 mode의 위상적 동일성을 증명한 ID는 아니다.

추적에서는 네 접힘각을 이전 연속 상태 또는 접선 예측값에 lift한다.
후보와 예측 사이의 거리와 모호성을 검사하며, 필요하면 구동각 간격을
재귀적으로 줄인다. 같은 branch를 먼저 선택한 뒤 contact bound를 검사한다.

구동 정의역은 일반식의 판별식에 대한 tan²(d/2) 다항식의 실근으로 나눈다.
내부 구간에서 해의 존재를 확인하고, 구동 불가능한 공백을 가로지르는
이동은 DRIVE_EXTREMUM으로 거부한다. 실제 branch가 불명확한 지점에서
다른 후보로 임의 탈출하지 않는다. 이 수치 구간 분할은 정확 대수적인
전체성 인증과 구분한다.

추적이 필요한 경우 각 구동 성분에서 기본 10개 정상 내부 후보를
확인한다. 후보 하나의 검사에는 그 구동각의 해들이 기준 상태의 서로 다른
경로에 일관되게 연결되는지 확인하는 과정이 포함된다.
최대 시도 수는 기본 50개이며, 정상 검사가 충분하지 않으면 실패를 알린다.
직접 모드식으로 처리하는 경우에는 다중 seed 검사를 생략한다.
실제 시도 수는 metrics.seed_attempts와 metrics.seed_checks에서 확인한다.

연결 성분의 끝, 각도 정의역 끝, contact bound, 수치 실패를 Endpoint의
reason으로 구분한다. 실제 연속 각도 bound의 유효/무효 전이가 확인된
contact bracket에는 22회의 이분 탐색을 적용한다. 이는 각도 제한의 경계이며
패널 충돌 시점은 아니다. admissible_domain은 관계식에서 얻은 구동 정의역,
reachable_intervals는 추적으로 확인한 범위로 분리한다. 수치 실패가 먼저
발생하면 도달 범위를 실제 도달점으로 줄이고 실패 이유를 보존한다.
그래프는 단절 구간과 각도 seam을 분리해 그린다.

대표 구동각 구간의 양 끝에서 네 접힘각 및 접선 이웃을 검사하여
seam_connections를 기록한다. 같은 물리적 경로가 대표 구간의 서로 다른
성분에 나타날 수 있으므로 mode_id와 principal 구간의 branch ID는
별개다. 사용자 세션의 branch ID는 정상적인 seam 통과 중 유지한다.

### 10.5 Contact와 주기적 slider

현재 contact=True는 **연속 접힘각의 [-pi,pi] bound**다.
패널-패널 비관통이나 재료 두께 검사는 구현하지 않는다.
branch의 principal angle이 범위 안에 있다는 이유만으로 경로를 허용하지 않고,
선택한 경로를 따라 lift한 연속 각도가 bound를 만족하는지 검사한다.

SimulationSession은 다음 상태를 별도로 관리한다.

- drive_unwrapped: 진행 이력을 가진 연속 구동각
- drive_wrapped: 표시용 대표 구동각
- winding_count
- rhos: 연속 접힘각
- branch_id / residual / events / mv_pattern

대부분의 대표 각도는 [-pi,pi)이며, 정확한 양의 pi seam은 +pi로 표시한다.
대표값이 바뀌어도 실제 구동 진행 방향은 연속 각도에서 계산한다.

contact=False에서는 slider에 기본 0.1 rad의 확장 구간을 둔다.
Native Matplotlib slider가 초과량을 먼저 잘라버리는 것을 피하기 위해
양 끝을 실제로 확장하며, 허용 seam을 통과하면 초과량을 보존해 반대편으로
위젯을 이동시킨다. 콜백 이벤트와 재진입 guard를 사용해 이 재배치가
두 번째 사용자 이동으로 처리되지 않도록 한다.

    -pi-epsilon -> +pi-epsilon
    +pi+epsilon -> -pi+epsilon

경계 통과는 매 단계의 네 접힘각과 예측 접선으로 확인한다.
실제 구동 불가능 구간, 모호한 분기, 수치 실패는 wrap으로 건너뛰지 않는다.
정상적인 seam 통과는 branch ID를 유지하며 ANGLE_WRAP 이벤트를 기록한다.

contact=True는 구동각을 [-pi,pi]로 제한하고 자동 주기 이동을 적용하지 않는다.
contact=False에서 True로 바꿀 때 현재 연속 구동각과 접힘각이 bound 밖이면
전환을 거부하고 checkbox 및 기존 세션 상태를 되돌린다.
다른 branch로 옮기거나 상태를 principal 값으로 덮어 허용시키지 않는다.

고정 구동각 family를 선택하면 주 구동 slider 대신 별도의 free-fold
slider로 그 family의 자유 접힘각을 조절한다.
고정 구동각 자체를 움직여 family를 벗어나려는 요청은 거부한다.
한 대표 상태만 보여주고 연속 family 전체를 계산한 것처럼 취급하지 않는다.

### 10.6 실행과 API

v2 디렉토리를 작업 디렉토리로 하는 예:

    python origami_simulator.py --alphas 70 20 110 160 --drive 4 --resolution 1000 --no-contact

Headless 실행:

    MPLBACKEND=Agg python origami_simulator.py --alphas 60 60 120 120 --drive 4 --resolution 200 --no-contact --no-show

Python API:

    import numpy as np
    from origami_simulator import OrigamiDegree4Simulator

    sim = OrigamiDegree4Simulator(
        [70,20,110,160], driving_crease=4,
        contact=False, verbose=False, seed_count=10, max_seeds=50,
    )
    print(sim.geometry)
    sim.run_simulation(resolution=1000)
    print(list(sim.branches))
    session = sim.create_session(branch=next(iter(sim.branches)), contact=False)
    session.move_to(np.pi + 0.01)
    print(session.state.drive_wrapped, session.state.winding_count)
    viewer = sim.show_3d_interactive(branch=session.branch_id, contact=False)

기존의 _solve_quadratic_rho1 / solve_full_rhos_from_drive_and_rho1 /
get_3d_geometry 호출은 유지한다. 공개 driving_crease는 1-based,
기존 output_index는 0-based다.

단일 입출력 조회 결과의 다중 해 shape는 다음과 같이 수정했다.

    output_angles.shape == (n_inputs, max_solutions)
    folding_angles_all.shape == (n_inputs, max_solutions, 4)
    num_solutions.shape == (n_inputs,)

해가 없는 입력은 num_solutions=0으로 표현하며,
구현 오류를 bare except로 숨기지 않는다.
branch를 지정하지 않은 sign 조회는 해를 부호로 필터링하는 기존 의미를
유지하지만, 부호 하나가 branch의 고유 ID임을 뜻하지 않는다.

실제 선택 가능한 branch 전체는 sim.branches에서 확인한다.
branch1/2와 math_branch1/2는 기존 코드 호환을 위한 첫 두 일반 경로의 alias이며,
특수 family나 세 번째 이후의 경로를 포함하는 전체 인터페이스가 아니다.


### 10.7 실제 검증 결과와 실행 비용

실행 환경: Python 3.12.14, NumPy 2.3.5, Matplotlib 3.10.8, Agg backend.

재현 명령:

    MPLBACKEND=Agg MPLCONFIGDIR=/tmp/origami-test-mpl python -m unittest discover -s v2/tests -v

현재 자동 검증은 **38개 테스트 통과, 약 11.872초**다. 검사 범위는 다음과 같다.

- 여덟 기준 입력과 원래 e1–e4 구동 주름에서의 독립 회전 폐합
- 0에서의 정확한 부호 복원, 반각 무한대 근, near-flat 기울기
- null-fold 모드와 고정 구동각 family
- 기하의 회전·반사 permutation과 주름 incidence
- 원래 번호로 되돌린 순환 재번호화 해 집합의 일치
- 접힘각 반전 및 적합한 입력의 elliptic–hyperbolic duality
- 샘플 수 및 내부 seed 순서 변경에 대한 branch ID 일관성
- 정방향 이동 후 역방향 복귀의 상태·branch 일치
- 10–50개 seed 검사의 조건부 실행
- 다중 해 조회 shape, 해 개수와 NaN 오류 회귀
- 세 종류의 기존 tessellator/forward-kinematics 호출
- 실제 모델의 periodic slider 및 family slider 콜백
- 확장 구간의 초과량 보존, 반복 회전, winding_count, 콜백 재귀 방지
- contact 설정 전환 거부와 정확한 +pi에서의 비주기적 후퇴
- 좁은 구동 불가능 구간을 포함한 gap 통과 거부
- 캐시된 빈 후보의 실패 이유 보존
- 수치 추적 실패 시 미도달 구간을 reachable로 보고하지 않는지

전체 테스트의 최종 실행 횟수와 시간은 사용 환경의 부하에 따라 달라진다.
실제 OS 창에서 마우스를 드래그하는 수동 시험은 수행하지 않았다.
Agg에서 실제 Matplotlib 위젯의 콜백과 모델 상태 갱신을 실행했고,
3D 및 family 화면을 파일로 생성했다.

변경 전 solver의 **유효한** 해 보존을 별도로 비교했다.
108개 기하 × 45개 구동각 표본에서 폐합을 만족한 기존 후보 5,353개 중
새 solver가 놓친 후보는 0개였다. 새 후보는 5,513개였고 최대 폐합 오차는
약 2.13e-15였다. 이는 명시한 표본의 비교 결과이며 연속 입력 공간 전체의
완전성 증명은 아니다. 기존 solver의 잘못된 부호 tuple은 보존하지 않는다.

다음은 동일 실행 환경에서 resolution=1000으로 측정한 한 차례의 실행이다.
시간은 run_simulation의 벽시계 시간이고 GUI 초기화 비용은 포함하지 않는다.

| 섹터각 [deg] | 변경 전 [s] | 변경 후 [s] | 저장된 branch/family 항목 수 | family 수 | seed 검사 수 |
| --- | ---: | ---: | ---: | ---: | ---: |
| [70,20,110,160] | 0.141 | 0.541 | 2 | 0 | 0 |
| [60,90,135,75] | 0.131 | 0.887 | 2 | 0 | 10 |
| [70,70,120,110] | 0.129 | 0.861 | 2 | 0 | 10 |
| [60,60,120,120] | 0.135 | 0.880 | 4 | 2 | 0 |
| [70,110,70,110] | 0.133 | 0.766 | 3 | 1 | 0 |
| [90,90,90,90] | 실행 실패 | 1.063 | 5 | 3 | 0 |
| [60,90,120,150] | 0.129 | 0.815 | 2 | 0 | 10 |
| [40,80,50,110] | 0.097 | 0.623 | 4 | 0 | 20 |

이 표의 항목 수는 **선택한 구동각의 일반 경로와 fixed-drive family를 저장한
자료구조의 개수**다. 논문의 CFF 4개 / 2CFF 6개라는 branch 집계와 동일한
수를 세는 값이 아니다. ±pi의 구동 seam을 별도 항목으로 저장하고 연결
정보로 대응시키며, family의 자유각 구간도 별도 자료로 보존한다.

여덟 기준 실행의 새 full_math_data에서 최대 폐합 오차는 약 2.18e-15였다.
잘못된 상태 제거와 family 표본 추가 때문에 이전 전체 배열의 길이나 값과
bitwise 동일성을 목표로 하지 않는다.

새 실행은 기준의 정상 실행보다 약 4–8배 느렸다. 네 접힘각의 독립 폐합과
법선 복원, 조건부 다중 seed, adaptive continuation 및 contact 경계 계산이
추가되었기 때문이다. 속도 향상은 주장하지 않는다. 캐시는 동일 모델의
반복 후보 계산을 줄이며, 모델 입력 변경 시 이전 결과를 재사용하지 않도록
입력과 캐시 정책을 관리한다.

변경 전/후 상세 수치, 비교 스크립트의 결과, 프로파일과 화면 이미지는
작업 디렉토리의 baseline.json, after.json, solver_comparison.json,
verification-full.log 및 profile_after.txt에 남겼다.
이 임시 실행 기록은 제품 코드의 의존성이 아니다.

### 10.8 남은 한계와 완료 범위

- 180° 섹터는 사용자와 확인한 지원 범위에서 제외한다. 접힘각 ±180°는 지원한다.
- 모든 대칭 특수형에 별도의 간소화식을 구현한 것은 아니다.
  일반 관계식에 직접 대입하는 경로는 지원하며, 이를 모드 계산 미지원으로
  해석하지 않는다. 관련 유도와 자체 검증 결과는 11절에 정리했다.
- 관계식 후보와 도달 경계의 처리는 수치 알고리즘이다.
  전체 섹터각 공간에 대한 interval arithmetic 또는 대수적 완전성 증명을
  수행하지 않았다.
- contact는 연속 각도 제한이다. 패널 비관통, 층 순서, 두께 및 재료 변형은
  검증하지 않는다.
- 입력 기하를 연속적으로 바꿀 때 생성·소멸하는 branch의 전역 ID 추적은
  이번 인터페이스의 보장 범위가 아니다.
- 수학적으로 같은 회전 표현과 contact 경로의 동일성을 구분한다.
  fixed-drive family의 free-fold 이동도 명시적으로 선택하며,
  모호한 실제 분기에서 후속 경로를 자동 선택하지 않는다.
- 테스트에 포함한 일반형·특수형·경계 사례에서는 유효 해 보존과 동작을
  확인했다. 이것을 “모든 case의 완벽한 분리”로 확대해서 보고하지 않는다.

v1은 변경 대상에서 제외했다. 이번 구현은 기존 세 모듈의 상위 목적을
유지하면서 후보 계산, 모드/branch 관리, 세션 및 표시의 책임을 분리한다.

## 11. 대칭 조건의 직접 대입 검증과 설명 정정 (2026-11-16)

### 11.1 결론과 범위

180° 섹터각은 사용자와 합의한 지원 범위에서 제외한다.
접힘각 rho=±180°는 계속 지원한다.

앞서 사용한 “일부 대칭 특수형의 전용 모드 규칙이 덜 됐다”는 표현을 정정한다.
일반 관계식에 정확한 섹터각 조건을 대입하면 해당 기하의 해가 이미 포함된다.
별도의 독립적인 물리 법칙을 추가해야 정상 해를 얻는다는 의미가 아니다.
현재 symmetric_general_fallback 메타데이터는 이 일반식 직접 대입 경로를 뜻한다.
각도 동일성만 있는 기하도 이 이름으로 기록되므로, 전체 구조가 반드시 반사
대칭이라는 의미로 해석하지 않는다.

대칭 조건으로 식이 간단해지는 경우는 있지만, 모든 대칭형이 두 개의 선형
인수나 상수 비율 모드로 나뉘는 것은 아니다. 전용 표현은 일반식에서 유도한
간소화·수치 안정화·명명 방식으로 이해한다. 퇴화점의 연속 해 집합 보존은
일반 관계식 자체가 요구하는 처리다.

### 11.2 두 쌍의 인접각이 같은 경우

원래 driving crease=e4, x=tan(rho4/2), y=tan(rho1/2), 섹터각은 radian으로 둔다.
alpha=(a,a,b,b)를 기존 일반 이차식 P=Ay²+By+C에 대입하면 다음 항등식이 나온다.

    P/(2 sin b) = sin(a-b)y² + 2 sin(a)xy - sin(a+b).

sin(a-b)가 0이 아니고 x가 유한한 정상 구간에서는 다음 두 후보를 얻는다.

    y± = [-sin(a)x ± sqrt(sin²(a)x² + sin(a-b)sin(a+b))] / sin(a-b).

이는 추가한 법칙이 아니라 같은 일반식의 해다. 예를 들어 [80,80,110,110]도
이 관계로 계산할 수 있다. a+b=pi이면 더 인수분해되어 y=0 또는
x=cos(a)y가 되며, 추가 퇴화 조건은 동차좌표에서 따로 확인해야 한다.

특히 x=s/c, y=p/q로 동차화하면 전체 식은 다음과 같다.

    P_h = 2 sin(b)c [sin(a-b)c p² + 2 sin(a)s p q - sin(a+b)c q²].

c=0, 즉 rho4=±pi에서는 대괄호만 검사하면 안 된다. 이때 전체 식은
모든 p:q에 대해 0이 되는 고정 구동각 family를 포함한다. 유한 x 식을 얻는
과정에서 c를 나누어 버리면 이 성분이 사라진다. 현재 solver는 이 family를
명시적으로 보존한다. 법선 복원과 전체 회전 폐합도 계속 확인한다.

### 11.3 대향각 하나가 같은 경우와 인접각 하나가 같은 경우

alpha=(a,b,a,c)에서 다음 계수를 정의한다.

    u = cos(c)-cos(b)
    v = cos(2a-c)-cos(b)
    w = cos(2a+c)-cos(b)

일반식은 다음과 같이 정리된다.

    P = v x²y² + u(x²+y²) + 4 sin²(a)xy + w.

x와 y를 교환해도 같다는 대칭성은 있지만, 일반적으로 다항 인수 두 개로
나뉘지는 않는다. 예를 들어 (a,b,a,c)=(60°,90°,60°,120°)에서는

    2P = 2x²y² - x² + 6xy - y² - 1
    discriminant_y(2P) = 8x⁴ + 40x² - 4.

이 판별식은 실수 다항식의 제곱이 아니다. 따라서 반사 대칭만으로 두 개의
단순 선형 인수형 모드를 가정할 수 없다. 일반 이차식의 근 계산은 그대로
적용할 수 있다. 양쪽 대향각이 모두 같은 b=c 등의 추가 조건에서는
xy에 관한 인수분해가 가능하다.

alpha=(a,a,b,c)는 인접각 하나가 같다는 조건일 뿐 전체 반사 대칭을 뜻하지
않는다. [70,70,100,120]이 그 예다. 이 경우에도 일반 관계식에 그대로
대입하는 경로는 정상적인 계산 방식이다.

위 두 항등식은 SymPy의 전개·삼각함수 정리로 잔차가 정확히 0임을 확인했다.
이는 각도 범위 전체의 수치 branch 완전성 증명과는 구분한다.

### 11.4 독립 수치 검증과 발견된 실제 문제

고정 seed로 생성한 30개 기하와 앞서 설명한 3개 입력을 합쳐 검증했다.

- 33개 기하 × 구동 주름 4개 × 구동각 21개 = 2,772개 비교
- 독립적인 주름 벡터 내적 관계와 일반 solver의 유한 해 비교
- 고립 후보 4,540개, 구동 주름 변경 왕복 18,160회
- 순환 재번호화 13,620회, 반사 상태 3,012개 확인
- 고정 구동각 family 44개, 자유각 평가 924회
- 별도의 quaternion 폐합 제약을 푼 비선형 다중 시작 2,304회
- 비선형 비교의 정상 구간 36개에서 예상한 해 72개와 일치
- solver 후보와 변환 상태의 독립 quaternion 폐합 오차 최대 약 2.46e-15
- 별도 비선형 최적화로 얻은 해의 폐합 오차 최대 약 1.23e-11

검사한 정상 구간에서는 대칭 전용식을 추가하지 않아도 누락된 별도 해가
발견되지 않았다. 수치 다중 시작법은 전역 완전성의 증명이 아니다.

초기 왕복 검사에서는 투영 경계 67건을 NO_SOLUTION으로 놓치는 문제가
발견됐다. 예를 들면 alpha=[80,80,110,110], driving=e2에서 다음 상태다.

    rho = [-1.065056942892582, 0.6077184810757758,
           -1.0650569428925818, 0] rad.

입력 상태의 폐합은 유효하지만, 정규화된 판별식이 약 -5.01e-16으로 계산됐다.
기존 허용오차가 longdouble 중간 연산 정밀도에 치우쳐 float64 입력의 반올림
오차를 충분히 반영하지 못했다. 이는 전용 모드 법칙의 부재가 아닌 수치 판정 문제다.

입력 정밀도를 반영하는 스케일 기반 판별식 허용오차로 수정하고,
작은 음수를 0으로 처리한 후보에는 DISCRIMINANT_ROUNDOFF 이벤트를 기록한다.
전체 폐합 검증은 유지한다. 수정 후 위 독립 검사는 실패 0건이며, 대표 경계의
불가능한 쪽으로 1e-12 rad 벗어난 입력은 계속 NO_SOLUTION으로 거부한다.
수치 허용오차 내 판정이 정확한 실수 대수적 존재성 증명을 의미하지는 않는다.

### 11.5 회귀 검증

일반식 직접 대입과 간소화식의 근 집합 일치, 반사 대칭 오인 방지,
±pi family 보존, 경계 반올림과 불가능 영역 구분을 검사하는
[추가 테스트](./tests/test_symmetry_relations.py)를 포함했다.

    python -m unittest discover -s v2/tests -v

현재 전체 43개 테스트 통과, 약 11.981초다. 기존 38개 검증과 그 성능 표는
이전 구현 시점의 기록으로 유지한다. 이번에는 새 모드를 추가하지 않았으며,
판별식의 경계 허용오차와 관련 설명을 보완했다. 작업용 독립 재현 스크립트와
상세 결과는 work/symmetry-validation에 저장했다.
