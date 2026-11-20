# Backbone 중첩과 말단 좌표계 표시

## 바로 실행

저장소 루트의 PowerShell에서 다음 명령을 실행한다.

```powershell
python .\v2\origami_forward_kinematics.py --demo backbone --periods 5 --angles 15 45 90 135 170
```

두 개의 창을 표시한다.

1. 선택한 입력각별 전체 backbone 중첩과 유닛 중심 이동 경로
2. 같은 표본들의 마지막 유닛 중심 궤적과 선택한 위치의 좌표계

이 예제는 Imada–Tachi 문서 Fig. 3F의 각도·연결·sigma를 사용한다.
4개 정점을 한 cell로 하여 `--periods 5`이면 총 20개 유닛이다.

| cell 내 정점 | alphas (degree, 입력 순서 보존) | iout | sigma |
| --- | --- | --- | --- |
| 0 | [120,60,60,120] | 1 | -1 |
| 1 | [60,60,120,120] | 2 | +1 |
| 2 | [120,60,60,120] | 3 | -1 |
| 3 | [120,120,60,60] | 2 | +1 |

길이 (1,1,1), scale_factor=1, local contact=True를 사용한다.
논문의 유한 패널 외형을 복원하는 길이 지정은 아니며 전역 strip 충돌은 검사하지 않는다.
`--angles`는 sector angle이 아니라 구동 접힘각 `rho0_deg`의 **순서 있는 목록**이다.
추적은 입력 순서대로 진행하며 불가능한 구간을 자동으로 뛰어넘지 않는다.

기존 비교 데모도 유지한다.

```powershell
python .\v2\origami_forward_kinematics.py
```

기준 2유닛과 ±1° 섹터각 변형 5개의 기존 말단 궤적 비교에 좌표축을 추가한다.
궤적들이 거의 겹치므로 비교 개요에는 입력 기준 프레임과 Original의 마지막 유효 말단
프레임을 표시한다. 사례별 추가 좌표축은 `plot_end_trajectory`의 표본 옵션으로 지정한다.
별도 창에는 기준 strip의 -150,-120,-90,-60,-20° backbone 중첩을 표시한다.
비교 데모는 기존처럼 현재 폴더에 계산 가능한 `case_N_kinematics.mat`을 저장한다.
`--demo backbone`은 MAT 파일을 생성하지 않는다.

## 그래프 읽기

- 색이 다른 원형 마커 실선: 같은 입력각에서 유닛 중심을 순서대로 연결한 backbone.
  마지막 점 하나는 중심 정점이 아니라 **마지막 출력 주름 바깥쪽 끝점**이다.
- `u0`, `u3` 등의 라벨: 유닛 번호. 주름 번호 c0…c3와 다르다.
  혼잡을 줄이기 위해 첫 유효 상태에 일부 번호만 표시한다.
- 얇은 회색 점선: 같은 유닛 중심의 입력각별 이동을 연결한 보조선.
  선택한 표본 사이의 실제 곡선을 추가로 계산한 보간 결과는 아니다.
- `B0`, `B2` 등의 RGB 화살표: 해당 표본의 입력 주름 프레임.
- `E0`, `E2` 등의 RGB 화살표: 해당 표본의 마지막 출력 주름 프레임.
  X=빨강, Y=초록, Z=파랑이며 축 끝에 이름을 표시한다.
- 말단 궤적의 위치는 **마지막 유닛 중심**이다. 출력 주름 끝점 궤적과 구분한다.
- 사용 불가능한 입력은 이유를 그래프에 표시한다. 그 구간을 선으로 연결하지 않는다.

좌표축 크기는 표시된 위치 범위에 비례해 정하며 `frame_length`로 직접 지정할 수 있다.
그래프 범위에는 주름선과 좌표축 화살표 끝을 포함하고 세 축의 길이 배율을 같게 한다.
여러 사례나 서로 가까운 표본을 겹치면 라벨이 겹칠 수 있으므로
`frame_count` 또는 `frame_indices`로 표시할 좌표계를 줄인다.

## 좌표계 정의

주름 좌표계 원점은 해당 유닛의 중심 정점이다.
X는 **중심 정점 → 주름 바깥쪽 끝점**이며 방향 convention은 기존 FK와 같다.
Z는 정상 구간에서 방향이 정해진 두 패널 법선 사이의 이등분 방향이고 Y=Z×X다.
입력은 -rho/2, 출력은 +rho/2 보정한다. 완전접힘에서는 반각 및 경로에 따른
좌표계 convention이 적용되며 법선합을 정규화하는 식 자체는 퇴화할 수 있다.

`sample_backbone_motion`은 다음 세 기준을 지원한다.

| reference | 기준 |
| --- | --- |
| initial (기본) | **첫 유효 표본**의 입력 프레임을 한 번 저장해 전체 표본에 고정 |
| current | 표본마다 입력 프레임을 재계산하는 기존 상대 SE(3) 방식 |
| world | tessellator가 배치한 공통 assembly 좌표 |

첫 요청이 실패하면 첫 **유효** 입력각을 고정 기준으로 사용하고
`reference_rho0_deg`와 그래프 제목에 그 각도를 기록한다. 전부 실패하면 프레임을 만들지 않는다.
initial에서 나중 입력 프레임의 Y/Z 방향이 달라지는 것은 기준을 다시 잡아서가 아니라
입력 주름 자체의 반각 보정 방향이 변했기 때문이다.
이는 입력각에 따른 기하학적 표본이며 시간·속도·힘을 계산하는 동역학이 아니다.

## Python API

저장소 루트에서 실행하는 예제다.

```python
import sys
sys.path.insert(0, "v2")
import matplotlib.pyplot as plt
from origami_tesellator_1D import OrigamiTesellator1D
from origami_forward_kinematics import (
    sample_backbone_motion, plot_backbone_motion, plot_end_trajectory,
)

strip = OrigamiTesellator1D(
    [{"alphas": [70,60,110,120], "iout": 2, "sigma": 1}],
    num_periods=20, contact=True, verbose=False,
)
motion = sample_backbone_motion(
    strip, [15,45,90,135,170], reference="initial",
)
ax1 = plot_backbone_motion(
    motion, frame_indices=[0,2,4], unit_label_step=4, show=False,
)
ax2 = plot_end_trajectory(
    motion, frame_indices=[0,2,4], frame_length=0.5, show=False,
)
plt.show()
```

`sample_backbone_motion`은 strip을 계산하고 accepted state를 갱신한다.
기본 `reset=True, track=True`이며 새 추적을 시작하고 이후 표본은 같은 경로를 유지한다.
`reset=False`는 현재 선택 상태에서 시작한다. `track=False`는 각 입력을 독립 snapshot으로
계산하는 명시적 선택이다. 실패한 요청의 이전 정상 형상을 그 입력각의 형상으로 재사용하지 않는다.

반환값은 요청 순서의 dictionary 목록이다.

| 필드 | 의미 |
| --- | --- |
| unit_centers | 선택 기준에서 모든 중심, (유닛 수,3) |
| backbone | 중심들과 마지막 출력 끝점, (유닛 수+1,3) |
| P_rel, R_rel | **선택 기준**에서 말단 중심 위치·출력 주름 회전 |
| P_input, R_input | 선택 기준에서 현재 입력 프레임 위치·회전 |
| input_line, output_line | 선택 기준에서 주름 선분 양 끝점 |
| reference, reference_rho0_deg | 기준 종류와 고정 기준 입력각 |
| valid, status, termination | 성공 여부와 실패 이유 |
| sample_index, segment_id | 원래 표본 번호와 연결 구간 |

실패 표본은 NaN 좌표를 가지며 `frame_indices`에 지정하면 ValueError다.
`frame_count=0`으로 말단 프레임 자동 선택을 끄거나 backbone의 모든 프레임을 끌 수 있다.
말단 그래프의 기준 프레임도 끄려면 `show_base_frame=False`를 함께 지정한다.
`show_unit_paths=False`는 backbone 사이의 점선만 끈다.
두 plot 함수는 Axes3D를 반환하며 `ax=...`로 기존 그래프에 추가할 수 있다.

기존 `extract_relative_se3`·`run_trajectory_sweep`의 P_rel/R_rel 의미는 바뀌지 않았다.
기존 sweep도 `plot_end_trajectory(sweep)`로 좌표축을 표시할 수 있다.
이때는 current 기준이다. 단일 정점 관계식·번호·branch 선택·contact 정책은 수정하지 않았다.

## 검증 범위

`tests/test_fk_motion_plot.py`는 고정 기준의 좌표 복원, 전체 강체변환 불변성,
기존 current 기준과의 일치, 주름선과 프레임 X축 대응, 실제 구동 불가능 구간의 단절,
첫 표본 실패·전체 실패, 화살표 끝의 표시 범위 포함, 20유닛 왕복과 실제 렌더링을 검사한다.
기본 비교 데모와 20유닛 backbone 데모는 Agg로 실행하고 생성한 그림을 확인했다.
전체 회귀 테스트 127개가 78.806초에 통과했고 최종 수정 후 FK 관련 22개도 10.920초에 통과했다.
변경 전 FK와 18개 설정을 비교하여 성공/실패가 일치했으며 14개 유효 설정의
기존 P_rel/R_rel·입출력 선분 배열은 정확히 동일했다.
실제 Windows GUI 마우스 조작은 수행하지 않았다.
유닛 간 전역 충돌, 두께, 모든 특수형에서 경로 분류의 전체성을 보장하지 않는다.
