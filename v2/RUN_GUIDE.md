# Origami Simulation v2 실행 안내서

이 문서는 설치부터 입력 선택, 화면 조작, 사례별 예상 결과, 데이터 저장과 오류 해결까지 설명하는 사용자 안내서다.
개발 이력과 수학적 도출은 [UPDATE_PLAN.md](./UPDATE_PLAN.md)에 별도로 기록한다.
이 안내를 읽기 위해 개발 문서를 먼저 읽을 필요는 없다.

- 실행 예제의 기준 코드: `7540d27` 및 해당 코드를 포함하는 후속 커밋
- 확인일: 2026-11-16
- 명령어 환경: Windows PowerShell, 저장소 위치 `C:\python\Origami_Simulation`
- 수치·실행 확인 환경: Python 3.12.14, NumPy 2.3.5, Matplotlib 3.10.8, headless Agg
- 대표 10개 입력을 1000점으로 계산하고 CLI 실행과 Contact ON/OFF의 실제 viewer 구성을 확인했다.
  실제 Windows 데스크톱에서 마우스를 조작한 검증과 구분한다.

## 목차

1. [바로 실행하기](#quick-start)
2. [설치·폴더·Python 환경](#setup)
3. [입력 각도와 주름 번호](#angles)
4. [명령행 옵션 전체](#cli)
5. [화면 조작과 결과 읽기](#viewer)
6. [Branch·Contact·랜덤 초기값·wrap](#policies)
7. [케이스별 실행 예제와 예상 결과](#cases)
8. [Python으로 결과 조회·저장](#python-api)
9. [Tessellator와 forward kinematics 실행](#other-programs)
10. [창 없이 실행·검증](#headless)
11. [Git 업데이트와 실행 오류 해결](#troubleshooting)
12. [검사 범위와 결과 해석의 한계](#limits)

<a id="quick-start"></a>

## 1. 바로 실행하기

이미 저장소를 받아 놓았다면 다음 명령으로 시작한다.

```powershell
Set-Location C:\python\Origami_Simulation
python .\v2\origami_simulator.py --alphas 70 20 110 160 --drive 4
```

왼쪽에 3D 접힘 형상, 오른쪽에 접힘각 관계 그래프가 나타난다.
아래 `Fold e4` slider를 움직여 접고, 왼쪽 아래에서 두 branch를 비교한다.
기본값은 **Contact bounds ON**이다.

±180°를 넘는 수학적 경로와 주기 이동을 확인하려면 다음을 사용한다.

```powershell
python .\v2\origami_simulator.py --alphas 70 20 110 160 --drive 4 --no-contact
```

빠르게 확인할 때는 `--resolution 200`, 기본 상세 계산에는 `--resolution 1000`을 사용한다.
점 개수를 바꾸는 것은 기하 입력이나 branch의 정의를 바꾸는 조작이 아니다.
창을 닫으면 프로그램이 끝난다. 터미널에서 중단하려면 `Ctrl+C`를 누른다.

<a id="setup"></a>

## 2. 설치·폴더·Python 환경

### 2.1 처음 저장소를 받는 경우

이미 `C:\python\Origami_Simulation`이 있으면 아래 clone 단계는 생략한다.

```powershell
Set-Location C:\python
git clone https://github.com/SEOKHYEON-TOM/Origami_Simulation.git
Set-Location .\Origami_Simulation
```

이후 명령은 별도 설명이 없으면 **저장소 루트**에서 실행한다.
`C:\python`과 `C:\python\Origami_Simulation`은 서로 다른 실행 위치다.

```powershell
Get-Location
Test-Path .\v2\origami_simulator.py
```

두 번째 결과가 `True`여야 이 문서의 상대 경로 실행문을 사용할 수 있다.
다른 위치에서도 다음 절대 경로로 실행할 수 있다.

```powershell
python C:\python\Origami_Simulation\v2\origami_simulator.py --alphas 70 20 110 160 --drive 4
```

### 2.2 Python과 패키지

Python 3.10 이상을 사용한다. 이 문서의 실행은 Python 3.12에서 확인했다.
패키지가 이미 설치되어 있다면 다음 확인만 하면 된다.

```powershell
python --version
python -c "import sys; print(sys.executable)"
python -c "import numpy, matplotlib, scipy; print(numpy.__version__, matplotlib.__version__, scipy.__version__)"
```

패키지가 없으면 현재 실행하는 Python에 설치한다.

```powershell
python -m pip install -r .\v2\requirements.txt
```

`requirements.txt`의 기본 의존성은 NumPy, Matplotlib, SciPy다.
`pip`만 단독 실행하는 것보다 `python -m pip`를 사용하면 실행 인터프리터와 설치 대상을 맞추기 쉽다.
VS Code에서는 `Python: Select Interpreter`로 같은 Python을 선택한다.
전역 패키지 환경이 준비되어 있으면 가상환경을 새로 만들 필요는 없다.

### 2.3 어떤 파일을 실행하는가

| 파일 | 역할 | 직접 실행 |
| --- | --- | --- |
| `origami_simulator.py` | 단일 4-crease 정점의 계산·branch·Contact·3D viewer 진입점 | 이 문서의 주 실행 파일 |
| `origami_kinematics.py` | 기하 분류와 접힘각 후보 계산 | simulator가 import |
| `origami_viewer.py` | slider, branch 선택, 3D 주름 클릭과 그래프 | simulator가 import |
| `origami_tesellator_1D.py` | 여러 유닛을 연결하는 1D strip | 별도 데모 실행 가능 |
| `origami_forward_kinematics.py` | strip 말단의 상대 위치·자세와 궤적 | 별도 데모 실행 가능 |

`origami_simulator.py` 하나만 복사하면 import 파일이 빠질 수 있으므로 저장소의 `v2` 폴더 구성을 유지한다.
`v1`은 이전 코드 보관용이다. 이 문서의 옵션과 화면 설명은 `v2` 기준이다.

<a id="angles"></a>

## 3. 입력 각도와 주름 번호

### 3.1 Alpha와 rho는 다른 각도다

`--alphas`에는 패널 사이의 **섹터각 네 개를 도(degree)**로 넣는다.
입력한 순환 순서는 그대로 보존한다. 크기순으로 정렬하지 않는다.

```text
e1 --alpha1--> e2 --alpha2--> e3 --alpha3--> e4 --alpha4--> e1
```

| 입력·표기 | 뜻 |
| --- | --- |
| `alpha1` | e1과 e2 사이의 섹터각 |
| `alpha2` | e2와 e3 사이의 섹터각 |
| `alpha3` | e3와 e4 사이의 섹터각 |
| `alpha4` | e4와 e1 사이의 섹터각 |
| `rho1` … `rho4` | 각각 e1 … e4에서의 접힘각 |
| `--drive 4` | 원래 입력 번호의 e4를 구동 |

`rho=0`은 그 주름에서 두 패널이 펼쳐진 상태이고, `rho=±pi`는 완전접힘 경계다.
어떤 branch에서 구동각이 0이라고 해서 다른 세 접힘각도 모두 0인 것은 아니다.
화면 그래프와 slider는 radian을 사용하고, 3D 제목은 degree도 함께 보여준다.
`pi rad=180°`, `pi/2 rad=90°`, `1 rad≈57.296°`다.

현재 3D 좌표계는 e4를 +X 방향에 두고 e1을 XY 평면에서 alpha4만큼 회전한 방향에 둔다.
이는 그림의 기준 자세다. e4가 항상 구동 주름이라는 뜻은 아니다.

### 3.2 기하 유형과 특수 조건

| 조건 | 표시되는 기본 유형 |
| --- | --- |
| 섹터각 합이 360° | Developable, 콘솔에서는 `Euclidean (Flat)` |
| 섹터각 합이 360°보다 작음 | Elliptic, `Elliptic (Cone-like)` |
| 섹터각 합이 360°보다 큼 | Hyperbolic, `Hyperbolic (Saddle-like)` |

여기에 회전·반사 대칭, collinearity, flat-foldability 등의 조건이 겹칠 수 있다.
같은 각도 두 개가 있다는 사실만으로 전체 구조가 반사 대칭인 것은 아니다.
Developable의 Kawasaki flat-foldability는 `alpha1+alpha3=alpha2+alpha4=180°` 조건이다.
회전·반사 대응과 collinear 주름 쌍은 콘솔의 `geometry` 정보나 Python API로 확인할 수 있다.

입력 분류기는 `0<alpha_i<360°`를 검사한다. 정확한 180° 섹터는 solver 미지원이다.
이 문서에서 계산·물리적 Contact를 확인한 예제는 모두 `0<alpha_i<180°`다.
180° 초과의 reflex sector에 대한 패널 표현과 Contact 판정은 이 안내서의 검증 범위 밖이다.
360°에 가깝다는 이유만으로 입력을 자동으로 developable로 바꾸지 않는다.

<a id="cli"></a>

## 4. 명령행 옵션 전체

```powershell
python .\v2\origami_simulator.py --help
```

| 옵션 | 기본값 | 사용법·의미 |
| --- | --- | --- |
| `--alphas a1 a2 a3 a4` | `70 20 110 160` | 섹터각 네 개, degree |
| `--drive N` | Elliptic=e1, 그 외=e4 | 원래 번호 1,2,3,4 중 선택. 비교 실험에서는 명시 권장 |
| `--resolution N` | `1000` | 계산 샘플링 해상도, 8 이상의 정수 |
| `--branch NAME` | 첫 선택지 | 시작할 raw branch ID 또는 no-contact의 묶인 표시 이름 |
| `--no-contact` | 생략 시 Contact ON | 각도 접힘 제한·패널 관통 제한을 해제하고 수학적 경로 탐색 |
| `--random-seed N` | 미지정 | No-contact viewer의 무작위 초기값 재현용 정수 시드 |
| `--no-show` | 생략 시 창 표시 | viewer까지 구성하되 GUI 창을 표시하지 않음 |
| `-h`, `--help` | — | 도움말을 출력하고 종료 |

모든 옵션을 조합한 예:

```powershell
python .\v2\origami_simulator.py --alphas 70 80 70 100 --drive 4 --resolution 1000 --no-contact --branch NE_mode2 --random-seed 42
```

`--resolution 1000`이 모든 반환 배열의 행 수를 정확히 1000으로 맞추는 뜻은 아니다.
해가 없는 구간, 두 후보, 별도 seed·경계·family 때문에 실제 저장 개수는 달라진다.
또한 branch 추적용 내부 seed 후보 수 10–50개와 viewer의 랜덤 초기값은 서로 다른 설정이다.
`--no-show`는 자동 파일 저장 옵션이 아니다. CSV·PNG 저장은 [8절](#python-api)을 참고한다.

<a id="viewer"></a>

## 5. 화면 조작과 결과 읽기

| 화면 요소 | 사용법과 결과 |
| --- | --- |
| 왼쪽 3D 형상 | 공통 정점과 네 삼각 패널. 드래그하여 시점 회전 |
| 주황색 굵은 주름 | 현재 driving crease. 라벨도 같은 색으로 강조 |
| e1 … e4 선·라벨 | 다른 주름을 왼쪽 클릭하면 현재 형상을 유지하며 구동축 변경 |
| 오른쪽 그래프 | X축은 선택 구동각, 세 곡선은 나머지 접힘각 |
| 세로 점선·원형 마커 | 현재 구동각과 세 출력 접힘각 |
| 왼쪽 아래 radio 목록 | branch 또는 고정 구동각 family 선택 |
| `Fold eN` slider | 선택한 주름의 구동각을 radian으로 조절 |
| `Free eN` slider | family에서 독립적으로 움직일 수 있는 접힘각 |
| `Contact bounds` checkbox | Contact ON/OFF 전환 |
| 아래 상태 문구 | wrap 안내, 경로 제한, 고정 상태 또는 계산 오류 |

다른 주름을 선택하면 그래프의 X축·곡선·색·branch 목록·slider 이름이 함께 바뀐다.
e1=파랑, e2=초록, e3=빨강, e4=보라를 기본 색으로 사용하되 구동 주름은 주황색으로 강조한다.
새 구동축의 계산은 최초 선택 시 실행하고 같은 창에서는 축별 결과를 재사용한다.
그림을 돌리는 드래그와 오른쪽 클릭은 구동축 변경으로 처리하지 않는다.
선이 겹치면 시점을 돌리거나 eN 라벨을 클릭한다. 모든 주름이 만나는 원점은 선택 대상에서 제외한다.

구동축을 바꾸면 그 축에 맞는 branch 이름과 선택지 개수가 달라질 수 있다.
이름 변경만으로 형상이 다른 경로로 점프했다고 판단하지 않는다.
정확한 분기점에서 대응 경로를 정할 수 없으면 전환을 거부한다.
가능하면 slider를 정상적인 내부 상태로 조금 이동한 뒤 다시 선택한다.

현재 GUI는 창을 닫을 때 선택 branch·누적 회전·시점을 설정 파일로 자동 저장하지 않는다.
다시 재현할 입력과 branch·시드는 실행문에 기록해 둔다.

<a id="policies"></a>

## 6. Branch·Contact·랜덤 초기값·wrap

### 6.1 Branch 이름 읽기

| 이름 예 | 의미 |
| --- | --- |
| `ff_plus_c0`, `ff_minus_c0` | Developable flat-foldable 관계식의 인수분해 모드. plus/minus는 구동각 부호나 M/V 자체가 아님 |
| `E_unique_e1`, `E_unique_e2` | 일반 developable의 기준 경로를 unique crease로 식별 |
| `NE_mode1_c0`, `NE_mode2_c1` | Non-Euclidean의 경로 라벨과 구동 정의역 성분 |
| `NE_mode1`, `NE_mode2` | No-contact에서 검증된 ±pi 연결 구간을 묶은 표시 이름 |
| `collinear_null_c0` | 일부 주름이 0으로 남는 collinear 경로 |
| `collinear_other_c0` | 같은 collinear 입력의 다른 수학적 경로. Contact 허용과는 별개 |
| `family_drive_e4_180_free_e1_plus_pi` | e4=+180°에 고정하고 e1을 독립적으로 움직이는 family |
| `family_drive_e1_0_free_e2` | e1=0°에 고정하고 e2를 독립적으로 움직이는 family |

`c0/c1`은 M/V나 근의 +/- 반환 순서가 아니다. 지정 구동각의 구간을 구분하는 이름이다.
family 이름의 `_minus_pi`와 `_plus_pi`는 같은 회전의 두 대표 경계값을 Contact에서 구별하기 위해 남는다.
M/V는 현재 네 각도의 부호 정보이며 branch와 별도로 변할 수 있다.

### 6.2 Contact ON/OFF와 선택지 개수

Contact ON은 연속 접힘각의 ±pi 제한과 비인접 패널 내부 관통 검사를 적용한다.
선택한 시작 상태에서 같은 경로로 도달할 수 있는 허용 구간만 viewer에 표시한다.
No-contact는 이 물리적 제한을 해제하지만 회전 폐합·해 존재 여부·구동 극값은 계속 적용한다.

예를 들어 `[70,80,70,100]`, e4 구동은 Contact ON에서 4개 구간 선택지,
OFF에서 `NE_mode1/NE_mode2` 두 선택지를 표시한다.
숫자를 항상 4개/2개로 맞추는 것은 아니다. 단일 주기 경로로 묶이거나 독립 family가 더 있는 경우가 있다.
이 안내서의 [사례 표](#cases)에 실제 개수를 적었다.

Contact ON으로 새 branch를 선택하면 그 경로의 허용 상태에서 시작한다.
원래 seed가 관통하면 같은 raw branch의 가까운 허용 상태를 택하고, 허용 상태가 없으면 선택을 거부한다.
Contact OFF에서 창을 열거나 다른 경로를 선택하면 유효 구간 내부에서 무작위 초기화한다.
구간 길이에 비례해 후보 구간을 선택하고 양 끝 1%를 제외한 내부 각도를 계산·폐합 검증한다.
이 1%는 초기값 정책이며 실제 도달 범위를 잘라내는 규칙은 아니다.
family는 구동각 대신 자유각을 무작위로 선택한다.

동일한 입력·해상도·branch·시드·조작 순서를 사용하면 초기값을 재현할 수 있다.
같은 옵션 재클릭, Contact 전환, driving crease 클릭 때는 무작위 초기화하지 않는다.
No-contact에서 `--branch NE_mode1_c1`을 지정해도 그 이름이 속한 주기 경로 전체에서 시작하므로
반드시 c1 구간에서 초기화된다는 뜻은 아니다.

### 6.3 ±pi를 넘어가는 방법

No-contact에서 `Fold eN` slider를 양 끝의 작은 확장 구간까지 끌어 본다.
기본 확장 폭은 각 끝에서 0.1 rad다. 같은 경로 연결이 확인되면 다음처럼 이동한다.

```text
+pi + epsilon  → -pi + epsilon
-pi - epsilon  → +pi - epsilon
```

초과한 각도 크기를 보존하고, 제목의 `continuous`와 `turns`는 누적 구동 이력을 유지한다.
정확한 양의 pi 경계는 +pi로, 음의 pi 경계는 -pi로 표시한다.
대표 각도가 넘어가는 지점에서는 그래프 선을 끊어 화면을 가로지르는 잘못된 선을 방지한다.
내부에서는 경계 양쪽의 연결과 원래 경로를 유지한다.

해가 없거나 구동 극값·실제 분기·수치적 모호성 때문에 이어질 수 없으면 wrap하지 않는다.
모든 입력이 한 바퀴 연속 회전할 수 있는 것은 아니다. Contact ON에서는 자동 주기 이동을 적용하지 않는다.

### 6.4 Wrap 후 Contact를 켜는 경우

현재 형상이 비관통이면 UI는 필요할 때 각도 표현의 기준을 [-pi,pi]로 재설정하고 같은 형상을 유지한다.
`CONTACT_REFERENCE_RESET` 안내가 나타날 수 있다. 이전 누적 상태는 세션 이력에 보존한다.
현재 형상이 관통하면 Contact 전환을 거부하고 checkbox를 다시 OFF로 돌린다.
각도를 2*pi만큼 바꾸는 것만으로 패널 관통이 없어지지는 않는다.

구동 slider가 고정되는 경우도 구별한다.

| 상황 | 의미와 조작 |
| --- | --- |
| family에서 주 slider 고정 | 다른 자유각이 있으므로 `Free eN` slider 사용 |
| `CONTACT_LIMIT: isolated contact state` | 현재 branch의 허용 구동 상태가 고립되어 있음. 그 branch에서 주 slider로 움직일 수 없음 |
| 이동 요청만 거부되고 slider 복귀 | 요청한 각도가 현재 경로의 제약을 벗어남. 마지막 허용 상태 유지 |

<a id="cases"></a>

## 7. 케이스별 실행 예제와 예상 결과

아래 개수는 표시된 구동 주름·1000점 기준의 **viewer 선택지 수**다.
전체 configuration space의 위상적 연결 성분 수나 물리적 독립 folding mode 수와 동일시하지 않는다.

| 사례 | alpha 입력 | 합·기본 유형 | 주요 특징 | drive | Contact ON / OFF |
| --- | --- | --- | --- | --- | --- |
| [A](#case-a) | 70,20,110,160 | 360°, Developable | Flat-foldable | e4 | 2 / 2 |
| [B](#case-b) | 60,90,135,75 | 360°, Developable | 일반 unique-fold 예제 | e4 | 2 / 1 |
| [C](#case-c) | 70,80,70,100 | 320°, Elliptic | 반사 대칭, 중앙 구동 불가능 구간 | e4 | 4 / 2 |
| [D](#case-d) | 40,80,50,110 | 280°, Elliptic | 일부 Contact 경로가 경계점에 고립 | e1 | 4 / 2 |
| [E](#case-e) | 70,70,120,110 | 370°, Hyperbolic | e2/e4 collinear 조건, 같은 각도 쌍 | e4 | 2 / 2 |
| [F](#case-f) | 61,91,136,76 | 364°, Hyperbolic | 일반형 saddle 예제 | e4 | 2 / 2 |
| [G](#case-g) | 60,60,120,120 | 360°, Developable | Flat-foldable·collinear·반사 대칭 | e4 | 4 / 4 |
| [H](#case-h) | 70,110,70,110 | 360°, Developable | 회전·반사 대칭·두 collinear 쌍 | e1 | 3 / 3 |
| [I](#case-i) | 90,90,90,90 | 360°, Developable | 높은 대칭성·flat-foldable·family | e4 | 5 / 5 |
| [J](#case-j) | 60,90,120,150 | 420°, Hyperbolic | 두 raw 경로가 ±pi에서 연결 | e4 | 2 / 1 |

<a id="case-a"></a>

### A. 기본 flat-foldable: [70,20,110,160]

```powershell
python .\v2\origami_simulator.py --alphas 70 20 110 160 --drive 4
```

`ff_plus_c0`, `ff_minus_c0` 두 선택지가 나온다.
plus에서는 `rho1=rho3`, `rho2=-rho4`, minus에서는 `rho1=-rho3`, `rho2=rho4`인 경로를 볼 수 있다.
e4=45°에서 plus의 rho1은 약 55.93°, minus는 약 27.81°로 서로 다른 형상이 나온다.
보통 조작과 주름 클릭을 처음 익히기 좋은 예제다.
주기 이동은 `--no-contact`를 추가해서 확인한다.

```powershell
python .\v2\origami_simulator.py --alphas 70 20 110 160 --drive 4 --branch ff_minus_c0 --no-contact --random-seed 42
```

<a id="case-b"></a>

### B. 일반 developable과 unique fold: [60,90,135,75]

```powershell
python .\v2\origami_simulator.py --alphas 60 90 135 75 --drive 4 --branch E_unique_e1
```

Contact에서 `E_unique_e1`, `E_unique_e2` 두 경로를 비교한다.
기준 부분접힘 상태에서 각각 e1 또는 e2가 나머지 세 주름과 다른 부호를 갖는다.
e4=45°에서 두 경로의 네 각도는 대략 다음과 같다.

| 경로 | rho1 | rho2 | rho3 | rho4 |
| --- | ---: | ---: | ---: | ---: |
| E_unique_e1 | -16.89° | 39.74° | 18.38° | 45° |
| E_unique_e2 | 101.06° | -39.74° | 114.20° | 45° |

E_unique_e2의 Contact 구간은 e4 약 ±117.65° 부근에서 제한되고, E_unique_e1과 범위가 다르다.
현재 no-contact viewer에서는 두 raw 경로가 ±pi에서 연결된 것으로 묶여 **E_unique_e1 한 선택지**로 나온다.
E_unique_e2가 사라졌다는 뜻이 아니라 그 표시 선택지 안에 두 raw ID가 들어간다는 뜻이다.
같은 모양의 주기 경로 전체에서 unique crease 표시가 항상 동일하다고 해석하지 않는다.
두 기준 경로를 분리해 비교하려면 Contact ON과 `--branch E_unique_e2`를 사용한다.

<a id="case-c"></a>

### C. Elliptic의 4개/2개 선택지: [70,80,70,100]

```powershell
python .\v2\origami_simulator.py --alphas 70 80 70 100 --drive 4 --no-contact --random-seed 42
```

섹터각 합은 320°다. 수학적으로 e4는 대략 [-180°,-29.35°]와 [29.35°,180°]에서 정의된다.
0° 주위의 공백을 slider로 건너뛸 수 없다.

| 설정 | 표시 선택지 |
| --- | --- |
| Contact ON | NE_mode1_c0, NE_mode2_c0, NE_mode1_c1, NE_mode2_c1 |
| Contact OFF | NE_mode1, NE_mode2 |

No-contact의 NE_mode1에는 mode1_c0/c1, NE_mode2에는 mode2_c0/c1이 들어간다.
Contact를 켜면 일부 경로가 ±74.9° 부근의 패널 접촉 등으로 더 짧아질 수 있다.
No-contact에서 pi를 넘은 현재 형상이 관통하면 Contact ON 전환이 거부되는 것이 정상이다.
이는 단순 winding_count 문제와 구별한다.

<a id="case-d"></a>

### D. 일반 elliptic과 고립된 Contact 상태: [40,80,50,110]

```powershell
python .\v2\origami_simulator.py --alphas 40 80 50 110 --drive 1 --no-contact --random-seed 42
```

합이 280°이므로 **Elliptic**이다. Hyperbolic 예제가 아니다.
e1의 수학적 구간은 대략 [-180°,-50.92°]와 [50.92°,180°]다.
No-contact에서는 2개 선택지로 구간 연결을 살펴볼 수 있다.

Contact에서 움직이는 경로를 먼저 보려면 다음을 실행한다.

```powershell
python .\v2\origami_simulator.py --alphas 40 80 50 110 --drive 1 --branch NE_mode2_c0
```

NE_mode2_c0와 NE_mode1_c1에는 연속적으로 움직일 수 있는 허용 구간이 있다.
현재 검사에서는 NE_mode1_c0가 -180°, NE_mode2_c1이 +180°의 고립된 허용 경계 상태로 남는다.
따라서 Contact ON에서 기본 첫 경로가 고정되어 보여도 계산 자체가 멈춘 것은 아니다.
branch를 바꾸어 허용 구간과 형상을 비교한다.

<a id="case-e"></a>

### E. 특수 조건이 겹친 hyperbolic: [70,70,120,110]

```powershell
python .\v2\origami_simulator.py --alphas 70 70 120 110 --drive 4 --no-contact --random-seed 42
```

합은 370°다. `NE_mode1_c0`, `NE_mode2_c0` 두 선택지가 나온다.
동일한 70° 쌍이 있어도 전체 섹터각 배열의 반사 대칭은 성립하지 않는다.
한편 alpha4+alpha1=180°이므로 e2/e4의 collinear 조건은 기록된다.
이 입력의 이름은 `collinear_*` 대신 NE_mode*이며, 특수 조건과 경로 라벨은 별개다.
Contact ON에서는 한 경로의 음의 끝, 다른 경로의 양의 끝이 약 141.3° 크기 부근에서 제한될 수 있다.

<a id="case-f"></a>

### F. 일반 hyperbolic: [61,91,136,76]

```powershell
python .\v2\origami_simulator.py --alphas 61 91 136 76 --drive 4 --no-contact --random-seed 42
```

합은 364°다. `NE_mode1_c0`, `NE_mode2_c0`가 나온다.
두 경로 모두 e4=0 부근에서도 다른 접힘각이 0이 아닌 형상을 가질 수 있다.
No-contact의 수학적 e4 구간은 [-pi,pi] 전체지만 Contact에서는 경로별 범위가 다르다.
예를 들어 mode1은 음의 방향, mode2는 양의 방향에서 약 117.2° 크기 부근의 제한을 볼 수 있다.
단순한 일반형 saddle을 비교할 때 이 입력을 사용한다.

<a id="case-g"></a>

### G. Collinear·flat-foldable과 ±180° family: [60,60,120,120]

```powershell
python .\v2\origami_simulator.py --alphas 60 60 120 120 --drive 4
```

`ff_minus_c0`, `ff_plus_c0`와 e4=±180°의 family 두 개로 총 4개 선택지가 나온다.
ff_minus에서는 e1/e3가 0에 머물고 e2/e4가 접히는 null-fold 경로가 보인다.
반면 ff_plus는 다른 정상 경로다.

고정 구동각의 추가 자유도를 직접 실행하려면 다음을 사용한다.

```powershell
python .\v2\origami_simulator.py --alphas 60 60 120 120 --drive 4 --branch family_drive_e4_180_free_e1_plus_pi
```

주 slider는 e4=+180°에서 고정되고 **Free e1** slider가 나타난다.
Free e1을 움직이면 주 구동각이 고정된 상태에서도 형상이 달라진다.
이는 계산 실패나 Contact에 의해 모든 자유도가 막힌 상태가 아니다.

<a id="case-h"></a>

### H. 관통 때문에 0°에 고정되는 경로: [70,110,70,110]

```powershell
python .\v2\origami_simulator.py --alphas 70 110 70 110 --drive 1 --branch collinear_other_c0
```

회전·반사 대칭과 두 collinear 쌍이 있지만 Kawasaki flat-foldable 조건은 만족하지 않는다.
표시되는 세 선택지는 다음처럼 다르다.

| 선택지 | Contact에서 예상되는 결과 |
| --- | --- |
| collinear_null_c0 | e1/e3가 함께 움직이고 e2/e4는 0 |
| collinear_other_c0 | [rho1,rho2,rho3,rho4]=[0°,180°,0°,-180°]에서 e1 고정 |
| family_drive_e1_0_free_e2 | e1=0°에서 Free e2로 다른 자유도 구동 |

collinear_other는 0°에서 양의 구동각으로 움직이는 즉시 P1/P3가 관통한다.
각도 네 개가 ±180° 안에 있다는 사실만으로 허용되지 않는다.
Contact 그래프에는 현재 도달 가능한 0° 상태만 점으로 보이고 `isolated contact state`가 표시된다.
180°의 별도 비관통 경계 상태까지 중간의 관통 구간을 건너뛸 수는 없다.

수학적 경로 자체를 확인하려면 제한을 해제한다.

```powershell
python .\v2\origami_simulator.py --alphas 70 110 70 110 --drive 1 --branch collinear_other_c0 --no-contact --random-seed 42
```

이때 보이는 관통은 no-contact의 허용 범위다. 관통 중에 Contact를 켜면 전환을 거부한다.

<a id="case-i"></a>

### I. 정사각 대칭과 여러 family: [90,90,90,90]

```powershell
python .\v2\origami_simulator.py --alphas 90 90 90 90 --drive 4
```

현재 Contact ON/OFF 모두 다음 5개 선택지를 유지한다.

```text
ff_plus_c0
ff_minus_c0
family_drive_e4_180_free_e1_minus_pi
family_drive_e4_0_free_e1
family_drive_e4_180_free_e1_plus_pi
```

ff_minus는 e1/e3가 0인 경로이고, ff_plus는 e1/e3가 완전접힘 경계에 놓인 다른 경로다.
family는 주어진 e4=0 또는 ±180°에서 Free e1을 움직인다.
대칭성이 높다고 모든 해를 두 유한 이차근이나 두 메뉴 항목으로 줄이지 않는다.
No-contact에서도 독립 family는 일반 branch에 강제로 합치지 않는다.

<a id="case-j"></a>

### J. 큰 surplus와 주기 경로 병합: [60,90,120,150]

```powershell
python .\v2\origami_simulator.py --alphas 60 90 120 150 --drive 4 --no-contact --random-seed 42
```

합은 420°인 Hyperbolic이다. alpha1+alpha3=180°여도 다른 대각 합은 240°이므로
developable flat-foldable 예제로 분류하지 않는다.
Contact에서는 `NE_mode1_c0`, `NE_mode2_c0` 두 선택지를 표시한다.
No-contact에서는 검증된 ±pi 연결에 따라 **NE_mode1 한 선택지**로 묶인다.
Contact에서는 한 경로의 음의 끝, 다른 경로의 양의 끝이 e4 약 90° 크기 부근에서 제한된다.
단일 raw 구간의 두 근과 주기적으로 연결된 viewer 선택지의 차이를 보기 위한 예제다.

숫자로 적은 접촉 범위는 현재 설정의 근삿값이다. 끝점 근방의 허용오차·샘플링과
선택 시작 상태를 고려해야 하며, 표시된 양 끝 사이의 모든 경로가 연결됐다고 일반화하지 않는다.

<a id="python-api"></a>

## 8. Python으로 결과 조회·저장

다음은 Python 코드다. 저장소 루트에서 연 Python 세션에 입력하거나 별도 실행 스크립트로 사용한다.
파일로 저장할 때도 작업 디렉토리가 저장소 루트여야 `sys.path.insert`의 상대 경로가 맞는다.

### 8.1 분류·branch·특정 상태 조회

```python
import sys
import numpy as np
sys.path.insert(0, "v2")
from origami_simulator import OrigamiDegree4Simulator

sim = OrigamiDegree4Simulator(
    [70, 20, 110, 160], driving_crease=4, contact=True, verbose=False
).run_simulation(resolution=1000)

print(sim.geometry)
print(list(sim.branches))
session = sim.create_session("ff_plus_c0", contact=True)
state = session.move_to(np.deg2rad(45))
print("rho1..rho4 [deg]:", np.rad2deg(state.rhos))
print("closure residual:", state.residual)
print("MV:", state.mv_pattern)
print(sim.contact_report(state.rhos))
```

공개 driving crease는 1…4이고 반환 `rhos`의 순서는 항상 원래 rho1…rho4다.
Python 배열의 `q[0]`은 rho1, `q[3]`은 rho4다. `solve_states`, `move_to`, `rhos`는 radian을 사용한다.
기존 호환 API `compute_folding_angles_from_input_output`은 입력 인자 `input_angle_deg`가 degree이고
`output_index`만 0…3을 사용한다. 이 API의 `sign=±1`은 부호 조건이며 branch 선택을 대신하지 않는다.

고정 구동각 family의 접힘각 후보는 유한한 목록 대신 연속 자유각으로 정의될 수 있다.
그런 경우에는 대응 branch를 `create_session`으로 고르고 `session.move_family(free_angle_rad)`를 사용한다.

### 8.2 GUI를 Python에서 열고 선택하기

```python
viewer = sim.show_3d_interactive(
    branch="ff_plus_c0", contact=False, random_seed=42, wrap_margin=0.1, show=False
)
print(viewer.branch_options)  # 화면 표시 이름 -> raw branch ID tuple
viewer.set_driving_crease(1)
print(viewer.sim.driving_crease)  # 현재 viewer 축. 원래 sim 객체의 축과 구별
import matplotlib.pyplot as plt
plt.show()
```

`sim.branches`와 `sim.create_session()`은 raw ID를 사용한다.
묶인 표시 이름은 viewer의 `set_branch`와 CLI `--branch`에서 해석한다.
viewer와 별도로 만든 `sim.create_session()`은 기존의 결정적인 seed를 사용한다.
No-contact의 랜덤 시작 정책은 viewer에만 적용한다.
Python 세션의 Contact 기준 재설정은 명시적으로 `session.set_contact(True, reanchor=True)`로 요청한다.
반환값이 False이면 `session.last_error`를 확인한다. 관통을 무시하는 옵션은 아니다.

### 8.3 Kinematic map PNG와 CSV 저장

앞의 sim 생성 코드에 이어서 실행한다. 결과는 저장소 밖 `C:\python\Origami_results`에 저장된다.

```python
from pathlib import Path
import matplotlib.pyplot as plt

out = Path("../Origami_results")
out.mkdir(exist_ok=True)
fig = sim.plot_2d_static(branch="ff_plus_c0", contact=True, show=False)
fig.savefig(out / "ff_plus_map.png", dpi=180)
plt.close(fig)

# 각 단절 구간을 별도 CSV로 저장한다. 서로 떨어진 구간을 한 선으로 잇지 않는다.
for index, piece in enumerate(sim.plot_data("ff_plus_c0", contact=True)):
    np.savetxt(
        out / f"ff_plus_segment_{index}.csv", piece, delimiter=",",
        header="rho1_rad,rho2_rad,rho3_rad,rho4_rad", comments="",
    )
```

CSV 열 순서는 원래 rho1…rho4이고 단위는 radian이다.
`full_math_data`는 수학적 후보를 담고, `physical_segments`는 현재 Contact 정책으로 걸러진 조각을 담는다.
`branch.reachable_intervals`는 수학적 추적 구간이므로 Contact에서 모두 이동 가능하다는 뜻으로 사용하지 않는다.
현재 세션에서 도달 가능한 Contact 그래프는 `session.plot_data()`를 사용한다.

<a id="other-programs"></a>

## 9. Tessellator와 forward kinematics 실행

### 9.1 실행 경로와 출력 폴더

두 파일은 simulator의 CLI 옵션을 공유하지 않는다. `--alphas`, `--branch`, `--no-show`를 뒤에 붙여
설정하는 방식 대신 현재 파일 하단의 데모 구성 또는 Python API를 사용한다.
`.mat`은 현재 작업 디렉토리에 생성되므로 결과 폴더를 별도로 두는 것을 권한다.

```powershell
New-Item -ItemType Directory -Force C:\python\Origami_results | Out-Null
Push-Location C:\python\Origami_results
python C:\python\Origami_Simulation\v2\origami_tesellator_1D.py
Pop-Location
```

현재 tessellator 기본 실행은 `[70,70,120,110]`, `iout=3`, 2개 유닛의 입력 -20° 예제를 사용한다.
Cobweb 그래프가 나타나며 **그 창을 닫으면** 뒤의 배치 디버그 출력과 `strip_debug.mat` 저장이 이어진다.
기본 main이 곧바로 새 simulator와 같은 3D branch viewer를 여는 것은 아니다.

| 출력 | 의미 |
| --- | --- |
| Cobweb 그림 | 입력·출력각을 유닛 사이에 전파하는 과정 |
| `valid/total_units` | 전체 strip 중 계산·배치에 성공한 유닛 수 |
| 배치·법선 디버그 로그 | 인접 유닛의 연결 방향·프레임 점검 |
| `strip_debug.mat` | 해당 스냅샷의 정점·면·연결·배치 진단 정보 |

### 9.2 Strip 3D slider를 Python에서 여는 예

다음 Python 코드는 저장소 루트 기준이다.

```python
import sys
sys.path.insert(0, "v2")
from origami_tesellator_1D import OrigamiTesellator1D

strip = OrigamiTesellator1D(
    cell_configs=[{"alphas": [70, 20, 110, 160], "sigma": 1, "iout": 3}],
    num_periods=2, lengths=(1.0, 1.0, 1.0), verbose=False,
)
strip.setup_interactive_viewer()
```

`cell_configs`는 한 주기의 유닛 목록, 전체 유닛 수는 `len(cell_configs)*num_periods`다.
`sigma`는 출력 부호 조건이고 `iout`은 tessellator 자체 번호의 출력 주름 1,2,3이다.
Tessellator는 입력 주름을 c0로 부르는 별도 convention을 사용하므로 simulator의 공개 e1…e4와
번호만 보고 그대로 대응시키지 않는다. 내부 `CLV`와 solver index 변환이 대응을 관리한다.
`compute_strip_kinematics`는 radian, `plot_3d(rho0_deg=...)`와 `export_to_mat(..., rho0_deg=...)`는 degree를 받는다.

### 9.3 Forward kinematics 데모

```powershell
New-Item -ItemType Directory -Force C:\python\Origami_results | Out-Null
Push-Location C:\python\Origami_results
python C:\python\Origami_Simulation\v2\origami_forward_kinematics.py
Pop-Location
```

현재 기본값은 기준 섹터각 `[70,20,110,160]`, 2개 유닛, 입력 -150°…-20°의 100점 sweep이다.
난수 시드 24로 섹터각을 각 성분 ±1° 범위에서 바꾼 5개 비교 strip도 계산한다.
기준·변형 사례의 말단 궤적, 물리적 입출력 주름, 상대 SE(3) frame과 평면성 로그를 볼 수 있다.
기준 실행에서는 비교 사례 다섯 개의 궤적이 평면으로 판정됐다. 임의 입력에서도 평면이라는 뜻은 아니다.

성공한 변형 사례에 대해 `case_1_kinematics.mat` … `case_5_kinematics.mat`이 생성된다.
현재 main의 export 호출은 **rho0_deg=45°의 스냅샷**을 저장한다.
화면 sweep의 최종 입력 -20° 상태나 전체 궤적 시계열을 저장하는 파일로 해석하지 않는다.
파일의 `meta.rho0_deg`, `valid_units`를 확인한다.
`vertices`, `faces`, `backbone`, `e1_pts`, `e3_pts` 등이 들어가고 MATLAB faces는 1부터 시작한다.

두 strip 모듈은 기존 부호·전파·배치 흐름을 사용한다.
단일 정점 viewer의 새 branch 메뉴, 주기 slider, 패널 관통 검사가 strip 전체에 자동 적용되지는 않는다.
특히 tessellator의 contact 설정을 전역 strip 충돌 검사 완료로 해석하지 않는다.

<a id="headless"></a>

## 10. 창 없이 실행·검증

### 10.1 Simulator 계산과 viewer 구성 확인

PowerShell에서 다음을 사용한다.

```powershell
$env:MPLBACKEND = "Agg"
python .\v2\origami_simulator.py --alphas 70 80 70 100 --drive 4 --no-contact --random-seed 42 --no-show
Remove-Item Env:MPLBACKEND
```

Linux/bash에서는 같은 환경 변수를 다음처럼 지정한다.

```bash
MPLBACKEND=Agg python v2/origami_simulator.py --alphas 70 80 70 100 --drive 4 --no-contact --random-seed 42 --no-show
```

`--no-show`에서도 branch 선택·랜덤 초기화·Matplotlib 위젯 구성까지 실행된다.
실제 창 표시·마우스 조작 성공까지 확인했다는 뜻은 아니다.
다시 GUI를 열 때는 `MPLBACKEND=Agg`를 해제하고 `--no-show`도 생략한다.

### 10.2 자동 회귀 테스트

저장소 루트에서 실행한다.

```powershell
python -m unittest discover -s .\v2\tests -v
```

현재 기준 코드에는 회전 폐합, 대칭·재번호화, branch·wrap, 주름 클릭,
Contact·관통·고립 경계 등을 검사하는 81개 테스트가 있다.
`Ran 81 tests ... OK`가 기준 실행 결과다. 향후 테스트가 추가되면 개수는 늘 수 있다.
이는 모든 기하·충돌 상황의 수학적 전체성을 증명하는 결과가 아니다.

<a id="troubleshooting"></a>

## 11. Git 업데이트와 실행 오류 해결

### 11.1 평소 업데이트 순서

```powershell
Set-Location C:\python\Origami_Simulation
git status --short
git pull --ff-only
git log -1 --oneline
```

Pull이 성공한 후 실행한다. `origin/main`만 갱신되고 `Aborting`이 출력됐다면
작업 폴더의 코드는 여전히 이전 버전일 수 있다. `--ff-only`는 Git 갱신 방식이며 alpha나 기하 유형과 무관하다.
Simulator의 일반 실행은 UPDATE_PLAN.md나 이 안내서를 수정하지 않는다.
개인 메모는 저장소 밖 파일로 보관하면 원격 문서 업데이트와 겹치지 않는다.

### 11.2 UPDATE_PLAN.md의 로컬 변경 때문에 pull이 중단될 때

다음은 **그 문서를 저장소 밖에 백업한 후 해당 파일의 로컬 변경만 되돌리는** 순서다.
다른 Python 파일의 변경은 대상으로 삼지 않는다.

```powershell
$planBackup = "..\UPDATE_PLAN_$(Get-Date -Format 'yyyyMMdd_HHmmss').md"
Copy-Item .\v2\UPDATE_PLAN.md $planBackup -ErrorAction Stop
git update-index --no-skip-worktree -- v2/UPDATE_PLAN.md
git update-index --no-assume-unchanged -- v2/UPDATE_PLAN.md
git restore --source=HEAD --staged --worktree -- v2/UPDATE_PLAN.md
git pull --ff-only
git status --short
```

이후 개인 메모는 백업본에 적고 저장소의 추적 문서는 원격 버전으로 유지한다.
`skip-worktree`, `assume-unchanged`, `.gitignore`는 이미 추적 중인 문서의 개인 수정본을
보존하면서 원격 변경을 계속 받아들이는 해결책이 아니다.
다른 파일이 원인으로 나오는 경우에는 그 파일의 diff와 보관할 변경을 먼저 확인한다.

### 11.3 실행 메시지별 확인 사항

| 증상·메시지 | 의미와 확인 방법 |
| --- | --- |
| `can't open file ... No such file` | 현재 폴더와 경로 확인. `Test-Path .\v2\origami_simulator.py` 또는 절대 경로 사용 |
| `ModuleNotFoundError` | 현재 Python에 패키지가 없거나 v2의 관련 모듈이 빠짐. `python -m pip`와 폴더 구성 확인 |
| `Unknown branch` | 그 입력·drive·Contact 설정에 없는 이름. 우선 `--branch`를 빼고 실제 메뉴 확인 |
| `CONTACT_LIMIT ... panel penetration` | 실제 패널 내부 관통. No-contact에서 관찰은 가능하지만 Contact ON으로 허용되지 않음 |
| `CONTACT_LIMIT ... continuous fold bound` | 경로를 유지한 연속 접힘각이 ±pi 제한에 도달 |
| `isolated contact state` | 허용 구간이 한 점으로 남음. 사례 H의 other처럼 slider 고정이 정상일 수 있음 |
| `DRIVE_EXTREMUM` | 현재 구동 주름의 극값·해가 없는 구간·고정 구동 family. 다른 후보로 자동 갈아타지 않음 |
| `DOMAIN_END` | 지정한 구동 정의역 끝. Contact ON에서 ±pi 밖 요청 등 |
| `AMBIGUOUS` 또는 경로 대응 불명확 | 교차·특이점에서 다음 상태를 하나로 정하지 못함. 내부 상태에서 다시 선택·작은 각도 이동 시도 |
| `NUMERICAL_FAILURE` | 계산·폐합·추적 검증 실패. 물리적 도달 한계로 단정하지 말고 입력·drive·branch·상태 기록 |
| 180° 섹터 입력 후 해 없음 | 해당 섹터는 현재 solver 미지원. 구동각 180°의 지원 여부와는 다른 문제 |
| 창이 안 뜸 | `--no-show`와 `MPLBACKEND=Agg` 확인. VS Code의 Python·GUI backend 확인 |
| 매 실행 형상이 다름 | No-contact 랜덤 초기화. 같은 조건과 `--random-seed 42`로 재현 |
| 선택지가 1·3·4·5개로 나옴 | 주기 경로 병합이나 별도 family일 수 있음. 모든 입력이 두 선택지를 갖는 것은 아님 |

변경 전 코드가 실행되는지 확인하려면 실제 파일 경로와 인터프리터를 확인한다.

```powershell
python -c "import sys; print(sys.executable)"
git log -1 --oneline
git status --short
```

Strip 데모의 Unicode 출력에서 `UnicodeEncodeError`가 발생하면 해당 터미널에서 다음을 설정한 뒤 재실행한다.

```powershell
$env:PYTHONIOENCODING = "utf-8"
```

계산 오류를 기록할 때는 실행문 전체, Git 커밋, Contact 상태, branch 이름,
오류 직전의 구동각·클릭한 주름, 출력 메시지를 함께 남기면 재현하기 쉽다.
`--resolution`을 높이는 것만으로 특이점·불가능 구간·패널 관통이 해결되지는 않는다.

<a id="limits"></a>

## 12. 검사 범위와 결과 해석의 한계

- 수학적 해는 Foschi 관계식과 회전 폐합을 만족하는 접힘 상태다.
  그 자체가 패널 비관통·실물 제작 가능성·접힘 층 순서를 보장하지 않는다.
- 현재 단일 정점 Contact는 두께 0, 공통 정점, 단위 주름 길이의 네 삼각 패널을 대상으로 한다.
  공유 주름·원점·경계 접촉과 같은 평면에서의 겹침은 접촉으로 허용한다.
- 비인접 패널 내부의 횡방향 관통을 검사한다. 두께·마찰·접촉력·재료 변형은 계산하지 않는다.
- 이동 중 검사는 최대 0.035 rad의 내부 간격에서 수행한다.
  임의로 좁은 모든 충돌 구간을 탐지하는 연속 충돌 검사(CCD)의 전체성을 보장하지 않는다.
- `RANDOM_START`, `ANGLE_WRAP`, `MV_CHANGE`, `CONTACT_REFERENCE_RESET`,
  `DRIVING_CREASE_CHANGED`는 상태 변화 기록이며 모두 물리적 종료 이유인 것은 아니다.
- 기하 대칭, algebraic candidate, folding mode, branch, M/V와 Contact 허용 상태를 각각 구분한다.
  메뉴 항목 수만으로 전체 configuration space를 완전히 분리했다고 판단하지 않는다.

실행과 화면 해석은 이 안내서를, 상세 도출·수정 내역·검증 한계는 [UPDATE_PLAN.md](./UPDATE_PLAN.md)를 참고한다.
