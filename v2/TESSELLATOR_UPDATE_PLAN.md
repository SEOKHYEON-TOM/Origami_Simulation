# v2 Tessellator 개선 기록

## 범위와 기준선

작업 기준은 실행 안내서가 추가된 `2847cce`다. v1과 v2의 기존 tessellator·FK,
현행 simulator·kinematics, 실행 안내서와 simulator 개선 기록을 확인했다.
저장소에 AGENTS.md는 없었다. v1은 수정하지 않았다.

수정 중심은 origami_tesellator_1D.py이며 origami_forward_kinematics.py의 소비 코드를 맞췄다.
Simulator에는 후보·상태·family를 함께 조회하고 고정 구동 주름으로 추적하는 공개 어댑터를 추가했다.
기존 Foschi 이차식, 각도 복원, 단일 정점 viewer와 기존 계산 API의 수식은 유지한다.

변경 전 소스 복사본과 60개 입력 조합의 결과를 작업용 work/tess-refactor에 저장했다.
기록은 Python·NumPy 버전, 성공 유닛 수, 접힘각, 정점·면, FK 위치·회전,
solver 호출 수와 성능 측정을 포함한다. 이전 검토의 v1 비교 실행 기록도 보존했다.
아래 숫자는 유한 사례의 검증 결과이며 모든 기하의 완전성을 뜻하지 않는다.

## 1. 기하 분류와 번호 변환

### 공유 분류

classify_geometry를 단일 판정원으로 사용한다. Tessellator의 별도 np.isclose 분류와
flat-foldability 허용오차를 제거했다. 사용자 순서의 geometry에는 회전·반사 대응,
collinear 쌍, flat-foldability와 near 상태가 그대로 들어간다.

기존 [70,20,110,159.999]는 tessellator가 Euclidean으로 처리하면서 새 solver의 Elliptic
기본 구동축과 충돌했다. 입력 60°가 c0에서 약 38.08°로 해석됐지만, 이제 정확히 60°다.

### 명시적인 주름 대응

사용자 theta0..theta3는 c0/c1, c1/c2, c2/c3, c3/c0 사이의 섹터각이다.
c0는 입력, c1/c2/c3 중 iout이 출력이다. 기하 유형별 기존 세계 좌표계를 보존하면서
solver의 driving_crease를 명시했다.

| 유형 | solver에 전달하는 섹터각 | 명시적 drive | rhos의 기존 저장 순서 | CLV[c0,c1,c2,c3] |
| --- | --- | --- | --- | --- |
| Developable/Hyperbolic | [theta1,theta2,theta3,theta0] | e4 | [rho1,rho2,rho3,rho0] | [4,1,2,3] |
| Elliptic | [theta0,theta1,theta2,theta3] | e1 | [rho0,rho1,rho2,rho3] | [1,2,3,4] |

여기서 rho 번호는 tessellator의 c0..c3 기준이다. solver의 내부 순환 번호와 구별한다.
cell의 fold_indices는 위 변환을 담고, get_fold_angles(t)는 항상 c0..c3 순서의 복사본을 반환한다.
get_fold_angles(t, order="solver")와 rhos는 기존 solver 순서를 유지한다.
미계산 rhos 행은 가짜 0도 대신 NaN이다. 미계산 유닛 getter는 IndexError를 낸다.
각도는 radian이며 GUI·rho0_deg 인자만 degree다.

1단계에서 12개 입력/출력 대응을 검사했고 정상 사례 9개는 기존 접힘각·정점 배열과 정확히 일치했다.
최종 정상 기준 [70,20,110,160]의 iout=2/3 비교도 정점 차이 약 1.5e-15 이하를 유지했다.

## 2. 계산·Cobweb·진단의 공통 경로

### Sigma 정의 변경과 호환성

기본 sigma_role="opposite"는 sign(rho0)*sign(rho2)=sigma라는 초기 선택 조건이다.
iout=1/3에서도 같은 의미를 사용한다. 이는 기존 Cobweb·대향각 관계식 주석의 convention을
일관되게 적용한 선택이다. 기존 주 계산의 iout=1/3 출력 부호 의미는 명시적으로
sigma_role="output"을 지정해 요청한다. 이 변경은 실행 안내서에도 명시했다.

Sigma는 초기 hint이지 경로 ID가 아니다. 영 접힘이나 동일 부호 후보 때문에 하나로
정해지지 않으면 AMBIGUOUS이며, 반환 순서로 선택하지 않는다.
mode_id, branch_id, initial_state로 경로를 구체적으로 지정할 수 있다.
명시적 경로가 sigma hint보다 우선한다. 지원 mode는 available_configurations로 조회한다.

unique_eN이라는 solver 표기는 사용자 주름에 대응한 unique_cN으로 변환한다.
branch_id는 원래 solver의 raw 경로 이름도 사용할 수 있다.
mode·branch 식별은 한 셀의 입력·구동축에 대한 것이며 다른 기하의 셀 사이에서
같은 문자열이 같은 물리 모드를 뜻한다고 가정하지 않는다.

### 대향 주름 관계식

theta3/theta0 쪽과 theta1/theta2 쪽에서 같은 구면 대각선의 내적을 쓰면 다음을 얻는다.

    cos(theta3)*cos(theta0)-sin(theta3)*sin(theta0)*cos(rho0)
      = cos(theta1)*cos(theta2)-sin(theta1)*sin(theta2)*cos(rho2)

따라서 기존 관계식을 유지한다.

    cos(rho2) = A*cos(rho0)+B
    A = sin(theta3)*sin(theta0)/(sin(theta1)*sin(theta2))
    B = (cos(theta1)*cos(theta2)-cos(theta3)*cos(theta0))/(sin(theta1)*sin(theta2))

단위 부호와 인덱스를 확인했고 독립 회전 폐합·정점 섹터각과 비교했다.
계수 스케일로 roundoff를 판단하되 큰 범위 초과를 clip하지 않는다.
실제 cosine 범위 이탈은 DRIVE_EXTREMUM, 큰 계수 취소로 경계 자체를 분해하지 못하면
NUMERICAL_FAILURE로 남긴다. validate_with_algebraic_solver의 잘못된 각도 쌍도 수정했다.
코드 주석에 있던 Imada 식 번호의 원문 전체 대조까지 수행한 것은 아니다.

### 단일 평가기

Snapshot, preview, Cobweb의 주황색 전파선, debug가 같은 _evaluate/_seed 경로를 사용한다.
기존 [60,90,135,75], iout=1, sigma=-1, 입력 0.6 rad에서 주 계산 -9.88°와
Cobweb -72.54°의 불일치가 있었다. 새 opposite convention은 해당 사례에서
같은 경로의 약 -72.54°를 반환한다. 숫자를 고정한 것이 아니라 명시된 부호 조건으로 선택했다.
output convention에서는 두 후보가 남아 AMBIGUOUS이며 mode를 지정해야 한다.

Cobweb은 branch1/branch2의 호환 배열에 의존하지 않고 전체 branch/family의 분리된
polyline을 사용한다. 배경은 모든 경로, 주황색은 현재 선택 정책·이력에 따른 정확한 전파다.
같은 경로의 조각은 같은 색을 쓰고 canonical mode와 raw ID를 함께 표시한다.
서로 다른 해를 unique(x)로 합치거나 정의역 공백을 보간하지 않는다.
resolution은 표시 행 수의 상한이며 경로 선택 atlas를 바꾸지 않는다.

[40,80,50,110]의 입력 0에서 가짜 보간 해가 나오던 v1 문제와,
v2 호환 배열의 첫 두 경로만 사용해 양의 유효 구간이 빠지던 문제를 회귀 검사했다.

## 3. 공간 전파·시간 추적·Contact

### 상태와 원자적 갱신

- Cell 정의와 공유 solver 모델은 유닛별 시간 상태와 분리한다.
- UnitState는 c-order folds, solver_rhos, mode/branch ID, 접선, 잔차, 구간,
  입력·출력각, family 자유 주름과 이벤트를 보존한다.
- StripResult는 성공 prefix, 정점·면, 접합 잔차, 종료 유닛·cell·입력각·이유를 보존한다.
- compute_strip_kinematics는 독립 snapshot이며 성공 prefix 개수를 반환한다.
- update 또는 compute_strip_kinematics(..., track=True)는 이전 전체 strip을 추적한다.
  도중 실패하면 last_attempt만 갱신하고 마지막 허용 전체 pose는 유지한다.
- preview·plot_cobweb·debug·export는 accepted pose를 덮어쓰지 않는다.
  preview는 기존 전체 이력이 있으면 그 경로를 사용하며 track=False로 독립 조회할 수 있다.
- reset은 이력을 명시적으로 지운다. 동일 cell 반복이어도 UnitState와 각도 배열은 공유하지 않는다.

시간 추적에서 sigma를 매 프레임 다시 적용하지 않는다. 이전 네 각도와 접선을 예측에 사용하고,
주 구동각 간격을 최대 0.035 rad로 나누어 공간 전파와 각 유닛의 고정 drive 추적을 수행한다.
모호성·수치 실패는 최대 10회, 1e-7 rad 수준까지 전역 입력 간격을 줄여 재검사한다.
도메인·접촉 한계를 다른 후보로 우회하지 않는다. 원형 거리는 좌표 비교에 쓰고,
연속 각도의 물리적 제한과 독립적으로 관리한다.

Simulator 공개 어댑터는 configuration_solutions, kinematic_intervals,
continue_configuration이다. 기존 일반식·map 생성 코드를 교체하지 않았다.
Continuum 경계에서는 A,C=O(delta^2), B=O(delta)이므로 p*q=0이라는 극한을 사용해
진입 경로의 정확한 자유각 0/pi를 선택한다. 섹터각에 epsilon을 더하지 않는다.
선택된 초기 family에서 정규 경로로 나가는 분기는 임의 결정하지 않는다.
자유각이 없으면 UNDERDETERMINED이며 free_angle 또는 유닛별 free_angles가 필요하다.

### Contact 범위

contact=True는 현재 경로에 대해 연속 접힘각 제한과 단일 정점의 횡방향 패널 관통을 검사한다.
상위 유닛이 전달한 입력각을 안전한 각도로 바꾸지 않는다. 현재 경로가 막혔다고 다른 해를 택하지 않는다.
set_contact(True)는 현재 전체 pose의 각 유닛을 재검사하고 실패 시 policy·각도·형상을 유지한다.
Strip은 각도 lift를 자동 재설정하지 않는다. 범위를 벗어난 누적 각도는 전환을 거부한다.

범위는 두께 0, 0<theta_i<pi인 정점의 삼각 패널이다. 180° 섹터는 기존 solver 미지원,
reflex sector의 물리 Contact는 UNSUPPORTED_CONTACT_GEOMETRY다.
전역 strip 자기 충돌, 두께, 마찰, 접촉력, 층 순서, CCD 전체성은 구현하지 않았다.
모든 결과의 global_contact_checked=False와 contact_scope에 이 범위를 명시한다.

종료 이유에는 DOMAIN_END, DRIVE_EXTREMUM, CONTACT_LIMIT, NUMERICAL_FAILURE,
AMBIGUOUS, UNDERDETERMINED, MODE_UNAVAILABLE, JOINT_INCOMPATIBLE 등이 있다.
수치 실패를 물리적 도달 한계로 바꾸지 않는다.

GUI는 전체 허용 pose를 유지하며 실패 요청의 slider를 복귀시킨다.
Local contact checkbox, family의 Free cN slider, 정확한 ±180° 초기화를 지원한다.
degree 표시 범위는 [-180,180]이고 strip에 새 주기 wrap UI는 추가하지 않았다.

## 4. FK·길이·접합

### 실제 주름과 각이등분 frame

FK는 get_fold_angles로 실제 c0와 cout 각도를 조회한다.
예전 기본 예제의 입력 45°에 약 -35.82°가 사용되던 raw index 오류를 제거했다.
X는 해당 주름축이고 Z는 양쪽 패널 법선의 이등분 방향이다.

rho가 앞 패널 법선을 뒤 패널 법선으로 회전시키는 convention에서,
입력 frame은 뒤 패널에서 시작하므로 -rho0/2, 출력 frame은 앞 패널에서 시작하므로
+rho_out/2로 보정한다. 기존 입력 쪽 +rho/2 부호도 독립적인 법선 이등분 검사로 수정했다.
R은 epsilon을 분모에 더하는 방식 대신 정규직교 basis로 구성한다.
정확한 완전접힘에서 법선 합이 0인 경우 이등분 방향은 정적 합으로 유일하지 않으며,
signed half-angle과 연속 각도 이력으로 frame을 정한다.

extract_relative_se3(..., compute=False)는 현재 pose를 재계산하지 않고 읽는다.
run_trajectory_sweep는 기본적으로 시간 추적하며 각 프레임에 입력각·상태·segment_id를 넣는다.
실패 프레임은 NaN 위치·회전으로 남아 그림 선을 끊는다. FK frame 수치 실패도
NUMERICAL_FAILURE gap이며 성공 프레임으로 위장하지 않는다.
track=False는 독립 snapshot sweep, include_failures=False는 명시적 압축 출력이며
이 경우에도 segment_id/sample_index로 분리를 보존해야 한다.

### 주름축과 유한 선분

joint_policy="axis"가 기본이다. 기존 유한 길이를 보존하면서 다음을 검사한다.

- 다음 원점 = 이전 출력 주름 끝점
- 다음 입력축 = 이전 출력축의 반대 방향
- 접합 패널 법선의 대응
- R^T R=I 및 det(R)=+1

축·법선으로 두 정규직교 frame을 만들고 R=F_target*F_source^T로 배치한다.
법선이 반평행인 경우 임의 회전축을 골라 입력축 정렬을 깨뜨리던 위험도 제거했다.

joint_policy="shared_edge"는 추가로 다음 입력 끝점 = 이전 원점을 요구한다.
길이를 자동 변경하지 않는다. lengths=(1,1,2), iout=3, scale=1은 axis에서
edge_error=1을 기록하고 shared_edge에서 JOINT_INCOMPATIBLE로 거부한다.
입력 길이 1과 scale_factor^n을 사용하는 convention에서는 앞 출력의 실제 길이와
다음 입력의 실제 길이가 같아야 유한 선분을 공유한다.
예를 들어 iout=2, lengths=(1,2,1), scale_factor=2는 그 길이 조건을 만족한다.
이는 전역 패널 비관통이나 전체 제작 가능한 패널 경계까지 인증하는 조건은 아니다.

## 5. 실행 예제와 호환성

다음 Python 예제는 저장소 루트에서 실행한다.

```python
import sys
import numpy as np
sys.path.insert(0, "v2")
from origami_tesellator_1D import OrigamiTesellator1D

strip = OrigamiTesellator1D(
    [{"alphas": [70,20,110,160], "iout": 2, "sigma": 1}],
    num_periods=5, contact=True, verbose=False,
)
strip.compute_strip_kinematics(np.deg2rad(45))
print(strip.get_fold_angles(0))
print(strip.available_configurations(np.deg2rad(45)))
result = strip.update(np.deg2rad(60))
print(result.complete, result.termination)
strip.setup_interactive_viewer()
```

주 계산과 Cobweb을 비교하려면 iout=1, alphas=[60,90,135,75], sigma=-1을 사용한다.
출력 부호 hint가 필요하면 sigma_role="output"을 명시하고 필요 시 mode_id="unique_c0" 또는
"unique_c1"을 지정한다. Family 예는 alphas=[90,90,90,90], free_angle=0.4,
입력 0 rad이며 GUI 초기값은 initial_rho0_deg=0으로 지정한다.

셀 설정·기하·길이·정책을 바꾸려면 새 strip을 구성한다. cells의 compiled mapping을 직접 수정하는
방식은 지원하지 않는다. 기존 private solver 호출을 전제로 한 소비 코드는 공개 API로 이전한다.
v1의 helix/tube는 그대로 보관했으며 향후 v2로 옮길 때 raw rhos 인덱스와 Contact 범위를 맞춰야 한다.

## 6. 검증과 성능

관련 unit 테스트는 test_tessellator.py와 test_strip_forward_kinematics.py에 추가했다.
기존 simulator 테스트도 함께 실행한다. 실제 결과는 tests/validation_summary.json의
tessellator_validation과 work/tess-refactor의 로그·baseline.json·matrix.json에 기록한다.

최종 전체 자동 테스트 **117개 통과, 72.312초**. 새 strip/FK 검증 35개를 추가했고
기존 단일 정점 테스트 81개도 모두 통과했다. 10개 입력×구동축 e1/e4의 20개 모델에서
기존 full_math_data, branch states, seed_state와 ID가 원소 단위로 같았다.
실행 안내의 simulator 명령 17개와 Python 코드 블록 8개도 Agg에서 실행했다.

10개 대표 입력 × iout 3개 × sigma 2개, 각각 3개 반복 유닛의 60개 조합을 실행했다.
전체 strip 41개, DRIVE_EXTREMUM 11개, MODE_UNAVAILABLE 2개, UNDERDETERMINED 6개였다.
실패한 조합에 해가 있다고 가정하거나 다른 경로로 바꾸지 않았다.
전체가 정의된 41개는 입력 1.4→1.5→1.3→1.4 rad 왕복을 수행했고 최대 각도 차이는 약 3.11e-15,
모든 기록된 상태의 최대 회전 폐합 잔차는 약 1.58e-15였다.

독립 검증은 회전 폐합, 정점 섹터각, 대향 주름 관계식, 양 패널 법선 이등분,
R의 정규직교성, 강체변환에 대한 상대 SE(3) 불변성, 유한 접합 조건을 포함한다.
후보 순서·atlas 81/129/257점·그림 표시 점 수 변화, 실제 slider·checkbox callback,
모호한 초기 상태, 정확한 0/±pi, null fold/family, Contact 전환 거부와 rollback도 검사한다.

동일 flat-foldable strip, iout=2, 입력 0.4→0.7 rad의 12프레임을 tracemalloc과 함께 한 차례 측정했다.

| 유닛 수 | 기존 snapshot | 새 snapshot | 새 시간 추적 |
| --- | ---: | ---: | ---: |
| 2 | 0.062s | 0.071s | 0.077s |
| 10 | 0.342s | 0.308s | 0.339s |
| 50 | 1.754s | 1.483s | 1.622s |

50유닛에서 root/recover 호출은 기존 600/1200회, 새 시간 추적 14/28회였다.
추적 상태·캐시 때문에 측정 peak allocation은 약 80KB→334KB로 증가했다.
이 예제에서는 sampled map을 만들지 않았다. 태그 없는 일반형이나 branch 지정·그래프 조회는
공유 model당 필요한 atlas를 지연 생성하며 각 유닛마다 다시 만들지 않는다.
API 후보 cache는 model당 4096개를 상한으로 한다. 이 수치는 전체 프로세스 RSS가 아니며
긴 구동의 다른 cache·atlas 메모리나 모든 기하의 성능을 대표하지 않는다.

기본 tessellator·FK main을 Agg로 실행했고 MAT 6개를 생성했다.
FK 예제의 Case 2에서는 100개 중 1개 실패 frame을 NaN으로 보존했다.
GUI는 headless 위젯 callback과 렌더링을 실행했으며 실제 Windows 마우스 조작은 수행하지 않았다.

## 7. 남은 한계

- 전역 strip 충돌, 두께·마찰·층 순서, 완전한 CCD는 미구현이다.
- 모든 특수형의 전역 mode 분류와 연결 전체성을 증명하지 않았다.
- 고정 구동각 projection 극값·분기에서 후속 경로를 확정하지 못하면 중단한다.
  임의로 구동 주름을 바꾸거나 pseudo-arclength로 우회하는 기능은 추가하지 않았다.
- 미지정 family 자유도와 모호한 초기 조건은 명시적 추가 입력이 필요하다.
- Atlas·추적의 수치 실패는 별도로 남고 전체 수학 공간의 부재로 해석하지 않는다.
- 축 연결은 유한 선분 공유·전역 패널 접합과 구분한다.
- FK 결과는 물리적 주름과 이등분 frame의 정의를 수정했으므로 이전 잘못된 수치를 보존하지 않는다.
