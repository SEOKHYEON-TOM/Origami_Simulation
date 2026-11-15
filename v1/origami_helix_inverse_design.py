"""
helix_inverse_design_v2.py
===========================
헬리컬 오리가미 Strip 역설계 관찰 도구 — Version 2

┌──────────────────────────────────────────────────────────────────────┐
│  수학적 핵심 (Imada 2025)                                            │
│                                                                       │
│  점화식:  cos ρ_{t+1} = A · cos ρ_t + B        (Eq.1)              │
│                                                                       │
│  ★ cos-공간에서 완전한 선형 맵!                                      │
│    → 고정점:  x* = B/(1-A),  ρ* = arccos(x*)                       │
│    → ρ-공간의 비선형성은 오직 arccosine 변환에서만 비롯됨           │
│                                                                       │
│  동역학 분류 (A-B 위상도):                                           │
│    Class I  : A=1, B=0  → 모든 ρ₀이 고정점 (FF 헬릭스)            │
│    Class II : |A|<1     → 수렴 (stable ρ*)                         │
│    Class III: |A|>1     → 발산 (unstable ρ*)                       │
│    Class IV : A=-1      → 2-주기                                    │
│                                                                       │
│  역설계 DOF:                                                          │
│    FF (Class I-2):    (r*, g*, ρ₀) → (θ₀, θ₁)   [exactly det.]   │
│    NE (α₁=α₂):       (r*, g*, ρ*)  → (α₁, α₃, α₄) [exactly det.] │
└──────────────────────────────────────────────────────────────────────┘

v1 대비 새 기능:
  ① cos-공간 선형 맵 시각화 (점화식의 선형성을 직접 확인)
  ② (A,B) 위상 다이어그램 (Class I~IV 경계 명시)
  ③ 설계공간 heatmap (FF: t₀×t₁→r,g  /  NE: α₃×α₄→r,ρ*)
  ④ 대화형 ρ₀ 슬라이더 (FF 헬릭스 실시간 탐색)
  ⑤ 최적화 수렴 추적 (역설계 경로를 설계공간에 오버레이)
  ⑥ 민감도 분석 (최적해 주변 국소 민감도)

사용법:
    python helix_inverse_design_v2.py              # 전체 비교 데모
    python helix_inverse_design_v2.py --forward    # 순설계 + A/B 위상도
    python helix_inverse_design_v2.py --scan-ff    # FF 설계공간 스캔
    python helix_inverse_design_v2.py --scan-ne    # NE 설계공간 스캔
    python helix_inverse_design_v2.py --interactive # 대화형 슬라이더
    python helix_inverse_design_v2.py --target-r 2.5 --target-g 0.3

의존성:
    origami_simulator.py, origami_tesellator_1D.py (동일 디렉토리 필요)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgs
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import minimize, differential_evolution
import warnings
import sys
import argparse

warnings.filterwarnings('ignore')


# ── 필수 모듈 임포트 ─────────────────────────────────────────────────────────
try:
    from origami_simulator import OrigamiDegree4Simulator
    from origami_tesellator_1D import OrigamiTesellator1D
except ImportError as e:
    print(f"\n[오류] 필수 모듈 없음: {e}")
    print("origami_simulator.py, origami_tesellator_1D.py를 같은 디렉토리에 두세요.\n")
    sys.exit(1)

# ── 전역 설정 ─────────────────────────────────────────────────────────────────
_N_PERIODS = 18
_LENGTHS   = (1.0, 1.0, 1.0)
_SIGMA     = 1
_IOUT      = 2

# ── 색상 팔레트 ───────────────────────────────────────────────────────────────
_C_FF   = '#2563EB'   # Flat-foldable (파란)
_C_NE   = '#EA580C'   # Non-Euclidean (주황)
_C_FP   = '#DC2626'   # Fixed point (빨간)
_C_CW   = '#7C3AED'   # Cobweb 궤도 (보라)
_C_BG   = '#F8FAFC'   # 배경
_PAL_FF = ['#BFDBFE', '#93C5FD', '#60A5FA', '#3B82F6']
_PAL_NE = ['#FED7AA', '#FCA5A5', '#FB923C', '#F97316']


# =============================================================================
#  Part 1: 수학 도구
# =============================================================================

def compute_AB(alphas_deg):
    """
    Imada 2025 Eq.1 — A, B 계수 해석 계산 (iout=2 기준).

    A = sin(α₄)·sin(α₁) / [sin(α₂)·sin(α₃)]
    B = [cos(α₂)·cos(α₃) - cos(α₄)·cos(α₁)] / [sin(α₂)·sin(α₃)]

    Note: 인덱스 α₁,α₂,α₃,α₄ = alphas_deg[0..3]
    """
    a = np.deg2rad(np.asarray(alphas_deg, dtype=float))
    a0, a1, a2, a3 = a
    denom = np.sin(a1) * np.sin(a2)
    if abs(denom) < 1e-12:
        return None, None
    A = np.sin(a3) * np.sin(a0) / denom
    B = (np.cos(a1)*np.cos(a2) - np.cos(a3)*np.cos(a0)) / denom
    return float(A), float(B)


def compute_fixed_point(A, B):
    """
    고정점 계산.

    cos-공간 고정점:  x* = B/(1-A)
    ρ-공간 고정점:   ρ* = arccos(x*)

    Returns
    -------
    cos_star : float or None
    rho_star_deg : float or None
    """
    if A is None or B is None:
        return None, None
    if np.isclose(float(A), 1.0, atol=1e-4):
        return None, None  # Class I: 모든 점이 고정점
    cos_star = float(B) / (1.0 - float(A))
    if not (-1.0 <= cos_star <= 1.0):
        return cos_star, None  # 고정점이 물리 영역 밖
    rho_star = float(np.degrees(np.arccos(np.clip(cos_star, -1.0, 1.0))))
    return cos_star, rho_star


def classify_AB(A, B):
    """
    (A, B) → Class I/II/III/IV + stability 분류.

    Returns
    -------
    label : str
    is_stable : bool or None
    class_num : int (1/2/3/4/0=unknown)
    """
    if A is None or B is None:
        return "Unknown", None, 0
    A, B = float(A), float(B)
    cos_star, rho_star = compute_fixed_point(A, B)
    fp_valid = rho_star is not None

    if np.isclose(A, 1.0, atol=1e-3) and np.isclose(B, 0.0, atol=1e-3):
        return "Class I  (A=1, B=0 — 모든 ρ₀ 고정점)", True, 1
    elif np.isclose(A, -1.0, atol=1e-3) and fp_valid:
        return "Class IV (A=−1 — 2주기)", None, 4
    elif abs(A) < 1.0 and fp_valid:
        return f"Class II  (|A|<1 — 수렴, ρ*≈{rho_star:.1f}°)", True, 2
    elif abs(A) > 1.0 and fp_valid:
        return f"Class III (|A|>1 — 발산, ρ*≈{rho_star:.1f}°)", False, 3
    else:
        return f"경계/특이 (A={A:.3f})", None, 0


def is_kawasaki(alphas_deg):
    """Kawasaki 평평접힘 조건: α₁+α₃=π, α₂+α₄=π"""
    a = np.deg2rad(np.asarray(alphas_deg, dtype=float))
    return (np.isclose(a[0]+a[2], np.pi, atol=1e-3) and
            np.isclose(a[1]+a[3], np.pi, atol=1e-3))


def get_geom_type(alphas_deg):
    """Euclidean / Elliptic / Hyperbolic 분류 + δ 반환"""
    total = sum(alphas_deg)
    if np.isclose(total, 360.0, atol=0.5):
        return "euclidean", 0.0
    return ("elliptic" if total < 360.0 else "hyperbolic"), total - 360.0


# =============================================================================
#  Part 2: 순설계 (Forward Analysis)
# =============================================================================

def _extract_helix_params(backbone_pts, trim_frac=0.15):
    """
    백본 점들로부터 나선 파라미터 (r, g) 추출.

    Imada 2025 Appendix B:
        g = a₁ (나선축 방향의 normalized pitch)
        r = Δs/(2√(1-g²)) · cot(ω/2)
    여기서는 수치적 PCA 기반 근사 사용.
    """
    pts = np.asarray(backbone_pts, dtype=float)
    n = len(pts)
    trim = max(1, int(n * trim_frac))
    if n > 2*trim + 4:
        pts = pts[trim:-trim]
    n = len(pts)
    if n < 4:
        return None

    chords = np.diff(pts, axis=0)
    cnorms = np.linalg.norm(chords, axis=1)
    vm = cnorms > 1e-8
    if vm.sum() < 3:
        return None

    # 나선축: PCA 최소 분산 방향
    cv = chords[vm]
    eigvals, eigvecs = np.linalg.eigh(cv.T @ cv)
    axis = eigvecs[:, 0]
    if np.dot(axis, pts[-1]-pts[0]) < 0:
        axis = -axis

    axp = pts @ axis
    perp = pts - axp[:, None]*axis
    r = float(np.median(np.linalg.norm(perp - perp.mean(0), axis=1)))

    axial_step  = float(np.mean(np.diff(axp)))
    mean_chord  = float(np.mean(cnorms[vm]))
    g = float(np.clip(axial_step/mean_chord, -1.0, 1.0)) if mean_chord > 1e-8 else 0.0

    return {"r": r, "g": g, "axis": axis, "axial_step": axial_step,
            "mean_chord": mean_chord}


def build_and_analyze(alphas_deg, rho0_deg, sigma=_SIGMA, iout=_IOUT,
                       num_periods=_N_PERIODS, lengths=_LENGTHS, verbose=False):
    """
    sector angles + ρ₀ → Strip 생성 후 나선 분석.

    Returns: dict{helix, A, B, rho_star_deg, dyn_class, tessellator, ...}
             or None on failure.
    """
    try:
        tess = OrigamiTesellator1D(
            cell_configs=[{"alphas": list(alphas_deg), "sigma": sigma, "iout": iout}],
            num_periods=num_periods, lengths=lengths,
            scale_factor=1.0, verbose=verbose
        )
        valid = tess.compute_strip_kinematics(np.deg2rad(rho0_deg))
        if valid < 5:
            return None

        res = tess._build_crease_lines()
        backbone = res[0] if (res and res[0] is not None) else None
        if backbone is None or len(backbone) < 6:
            backbone = np.array([gv[0] for gv in tess.global_vertices[:valid]])

        helix = _extract_helix_params(backbone)

        cell = tess.cells[0]
        A, B = cell.get("A"), cell.get("B")
        if A is None:
            A, B = compute_AB(alphas_deg)

        cos_star, rho_star = compute_fixed_point(A, B)
        cls, stable, cnum = classify_AB(A, B)
        geom, delta = get_geom_type(alphas_deg)
        ff = is_kawasaki(alphas_deg)

        return {
            "helix": helix, "A": A, "B": B,
            "cos_star": cos_star, "rho_star_deg": rho_star,
            "dyn_class": cls, "is_stable": stable, "class_num": cnum,
            "valid_units": valid, "tessellator": tess,
            "alphas_deg": list(alphas_deg), "rho0_deg": float(rho0_deg),
            "geom": geom, "delta_deg": delta, "is_ff": ff,
        }
    except Exception as e:
        if verbose:
            print(f"[build_and_analyze] {e}")
        return None


# =============================================================================
#  Part 3: 핵심 시각화 — cos-공간 선형 맵 (신규)
# =============================================================================

def plot_cosspace_linear_map(A, B, ax, n_cobweb=30, x0=None,
                              color='steelblue', label='', show_stability=True):
    """
    cos-공간에서 선형 점화식 f(x) = Ax + B 직접 시각화.

    ★ 이것이 v1과의 핵심 차이점 ★
    ρ-공간 cobweb은 비선형처럼 보이지만,
    cos ρ를 변수로 쓰면 f는 완전한 선형 함수.

    Parameters
    ----------
    A, B   : 점화식 계수
    ax     : matplotlib Axes
    n_cobweb: cobweb 반복 횟수
    x0     : 시작점 cos(ρ₀) (None이면 cobweb 생략)
    """
    if A is None or B is None:
        return

    A, B = float(A), float(B)

    xs = np.linspace(-1.0, 1.0, 400)
    ys = A * xs + B

    # 유효 영역 마스킹 (|y| ≤ 1)
    valid = np.abs(ys) <= 1.0
    ax.plot(xs[valid], ys[valid], color=color, lw=2.5, label=f'f(x) = {A:.3f}x + {B:.3f}{" "+label}')
    ax.plot(xs, xs, 'k--', lw=1.2, alpha=0.5, label='y = x  (identity)')

    # 허용 영역 음영
    ax.fill_between(xs, -1, 1, alpha=0.04, color='gray', label='physical domain')
    ax.axhline(1, color='gray', lw=0.7, ls=':')
    ax.axhline(-1, color='gray', lw=0.7, ls=':')
    ax.axvline(1, color='gray', lw=0.7, ls=':')
    ax.axvline(-1, color='gray', lw=0.7, ls=':')

    # 고정점 마킹
    cos_star, rho_star = compute_fixed_point(A, B)
    if cos_star is not None and -1.0 <= cos_star <= 1.0 and rho_star is not None:
        ax.plot(cos_star, cos_star, '*', color=_C_FP, ms=16, zorder=7,
                label=rf'fixed point $x^*$={cos_star:.3f}  ($\rho^*$={rho_star:.1f}°)')
        ax.axvline(cos_star, color=_C_FP, lw=1.0, ls=':', alpha=0.6)
        ax.axhline(cos_star, color=_C_FP, lw=1.0, ls=':', alpha=0.6)

    # 안정성 주석
    if show_stability:
        cls_label, _, cnum = classify_AB(A, B)
        slope_txt = rf"$|A|$ = {abs(A):.3f} {'< 1 → 안정 ✓' if abs(A)<1 else ('= 1 → 중립' if np.isclose(abs(A),1) else '> 1 → 불안정 ✗')}"
        ax.text(0.03, 0.97, slope_txt, transform=ax.transAxes,
                fontsize=9, va='top',
                bbox=dict(facecolor='#FFF8DC', edgecolor='#DAA520', boxstyle='round,pad=0.3'))

    # Cobweb 궤도 (cos-공간에서 계단 그림)
    if x0 is not None and -1.0 <= float(x0) <= 1.0:
        x = float(x0)
        cx, cy = [x], [0.0]  # 시작: (x0, 0)으로 올라가기
        for _ in range(n_cobweb):
            y = A * x + B
            if not (-1.0 <= y <= 1.0):
                break
            cx += [x, y]   # 수직선: (x, x) → (x, y)
            cy += [y, y]   # 수평선: (x, y) → (y, y)
            x = y
        ax.plot(cx, cy, color=_C_CW, lw=1.5, alpha=0.8, zorder=5,
                label=rf'cobweb ($x_0$={x0:.3f})')
        ax.plot(float(x0), 0, 'o', color=_C_CW, ms=8, zorder=8)

    ax.set_xlabel(r'$\cos\rho_t$  (current fold angle)', fontsize=9)
    ax.set_ylabel(r'$\cos\rho_{t+1}$  (next fold angle)', fontsize=9)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.legend(fontsize=7.5, loc='upper left')
    ax.grid(True, alpha=0.3)


def plot_rho_space_cobweb(A, B, ax, rho0_deg=None, n_cobweb=20,
                           color='steelblue', label=''):
    """
    ρ-공간 cobweb 다이어그램 (v1과 동일하지만 cos-공간과 비교용으로 유지).
    """
    if A is None or B is None:
        return
    A, B = float(A), float(B)

    rhos = np.linspace(-np.pi, np.pi, 600)
    f_vals = []
    for rho in rhos:
        cos_out = A * np.cos(rho) + B
        if -1.0 <= cos_out <= 1.0:
            f_vals.append(np.degrees(np.arccos(cos_out)))
        else:
            f_vals.append(np.nan)

    ax.plot(np.degrees(rhos), f_vals, color=color, lw=2, label=rf'$f(\rho)${" "+label}')
    ax.plot(np.degrees(rhos), np.degrees(rhos), 'k--', lw=1.2, alpha=0.5, label='identity')

    cos_star, rho_star = compute_fixed_point(A, B)
    if rho_star is not None:
        ax.plot(rho_star, rho_star, '*', color=_C_FP, ms=14, zorder=6,
                label=rf'$\rho^*$ = {rho_star:.1f}°')
        ax.axvline(rho_star, color=_C_FP, lw=1.0, ls=':', alpha=0.6)

    if rho0_deg is not None:
        rho = np.radians(rho0_deg)
        cx, cy = [np.degrees(rho)], [np.degrees(rho)]
        for _ in range(n_cobweb):
            cos_out = A * np.cos(rho) + B
            if not (-1.0 <= cos_out <= 1.0):
                break
            rho_next = np.arccos(cos_out)
            cx += [np.degrees(rho), np.degrees(rho_next)]
            cy += [np.degrees(rho_next), np.degrees(rho_next)]
            rho = rho_next
        ax.plot(cx, cy, color=_C_CW, lw=1.3, alpha=0.8, label='cobweb')

    ax.set_xlabel(r'$\rho_t$ [deg]', fontsize=9)
    ax.set_ylabel(r'$\rho_{t+1}$ [deg]', fontsize=9)
    ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7.5)


# =============================================================================
#  Part 4: (A,B) 위상 다이어그램 (신규)
# =============================================================================

def plot_AB_phase_diagram(ax, sample_points=None, highlight_pts=None):
    """
    (A,B) 평면에서 Class I~IV 경계와 현재 설계점 위치를 시각화.

    Class 경계:
      Class I  : A=1, B=0  (단점)
      Class II : 영역 |A|<1, -1 ≤ B/(1-A) ≤ 1
      Class III: 영역 |A|>1, -1 ≤ B/(1-A) ≤ 1
      Class IV : A=-1, 선분
    """
    # 배경 영역 표시
    A_grid = np.linspace(-3.0, 3.0, 400)

    # Class II: |A|<1
    A_cl2 = np.linspace(-1.0, 1.0, 200)
    B_lo_cl2 = (1.0 - A_cl2) * (-1.0)
    B_hi_cl2 = (1.0 - A_cl2) * 1.0
    ax.fill_between(A_cl2, B_lo_cl2, B_hi_cl2, alpha=0.15, color='steelblue', label=r'Class II ($|A|<1$, stable)')

    # Class III: A>1
    A_cl3r = np.linspace(1.0, 3.0, 100)
    B_lo_r = (1.0 - A_cl3r) * 1.0
    B_hi_r = (1.0 - A_cl3r) * (-1.0)
    ax.fill_between(A_cl3r, np.minimum(B_lo_r, B_hi_r), np.maximum(B_lo_r, B_hi_r),
                    alpha=0.12, color='tomato', label=r'Class III ($|A|>1$, unstable)')

    # Class III: A<-1
    A_cl3l = np.linspace(-3.0, -1.0, 100)
    B_lo_l = (1.0 - A_cl3l) * 1.0
    B_hi_l = (1.0 - A_cl3l) * (-1.0)
    ax.fill_between(A_cl3l, np.minimum(B_lo_l, B_hi_l), np.maximum(B_lo_l, B_hi_l),
                    alpha=0.12, color='tomato')

    # 경계선
    for sgn, col in [(1, 'steelblue'), (-1, 'darkorange')]:
        A_line = np.linspace(-3.0, 3.0, 400)
        B_line = sgn * (1.0 - A_line)
        ax.plot(A_line, B_line, '--', color=col, lw=1.2, alpha=0.7)

    # Class I 점
    ax.plot(1.0, 0.0, 'o', color='gold', ms=14, zorder=8,
            markeredgecolor='darkgoldenrod', markeredgewidth=1.5,
            label=r'Class I ($A=1,B=0$: every $\rho_0$ is fixed)')

    # Class IV 선
    A_cv4 = np.linspace(-3.0, -1.0, 2)
    ax.axvline(-1.0, color='mediumpurple', lw=2.0, ls='-.', label=r'Class IV ($A=-1$)', alpha=0.7)

    # 영역 레이블
    ax.text(0.0, 0.02, 'II', transform=ax.transAxes, fontsize=20,
            color='steelblue', alpha=0.3, fontweight='bold', ha='center', va='bottom')
    ax.text(0.82, 0.5, 'III', transform=ax.transAxes, fontsize=20,
            color='tomato', alpha=0.3, fontweight='bold', ha='center', va='center')
    ax.text(0.08, 0.5, 'III', transform=ax.transAxes, fontsize=20,
            color='tomato', alpha=0.3, fontweight='bold', ha='center', va='center')

    # 샘플 포인트 오버레이 (e.g. design space 스캔 결과)
    if sample_points is not None:
        As, Bs, rs = sample_points
        sc = ax.scatter(As, Bs, c=rs, cmap='plasma', s=20, alpha=0.6, zorder=5)
        plt.colorbar(sc, ax=ax, label=r'$r$ (helix radius)', shrink=0.7)

    # 강조 포인트 (역설계 결과)
    if highlight_pts:
        for (A_h, B_h, lbl, clr) in highlight_pts:
            ax.plot(A_h, B_h, 'D', color=clr, ms=12, zorder=10,
                    markeredgecolor='black', markeredgewidth=1,
                    label=lbl)
            ax.annotate(lbl, (A_h, B_h), xytext=(8, 8),
                        textcoords='offset points', fontsize=8)

    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-4.0, 4.0)
    ax.axhline(0, color='black', lw=0.7, alpha=0.4)
    ax.axvline(0, color='black', lw=0.7, alpha=0.4)
    ax.axvline(1, color='darkgray', lw=1.0, ls=':', alpha=0.5)
    ax.set_xlabel(r'$A$  (amplification / attenuation)', fontsize=9)
    ax.set_ylabel(r'$B$  (offset coefficient)', fontsize=9)
    ax.set_title(r'$(A,\,B)$ Phase Diagram' + '\n' + 'Origami Strip Dynamics Classification', fontsize=9, fontweight='bold')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.25)


# =============================================================================
#  Part 5: 설계공간 스캔 (신규)
# =============================================================================

def scan_ff_design_space(rho0_deg=-90.0, n_grid=18, num_periods=10,
                          t0_range=(8, 82), t1_range=(8, 82)):
    """
    Flat-foldable 설계공간 스캔: (θ₀, θ₁) → (r, g, A, B).

    FF 조건: θ = (t₀, t₁, 180-t₀, 180-t₁)  →  A=1, B=0 자동.
    ρ₀ 고정 시 (t₀, t₁)만으로 (r, g)가 결정.

    Returns
    -------
    dict{t0s, t1s, r_map, g_map}
    """
    t0s = np.linspace(t0_range[0], t0_range[1], n_grid)
    t1s = np.linspace(t1_range[0], t1_range[1], n_grid)
    r_map = np.full((n_grid, n_grid), np.nan)
    g_map = np.full((n_grid, n_grid), np.nan)

    total = n_grid * n_grid
    done  = 0
    print(f"\n  [FF 설계공간 스캔] ρ₀={rho0_deg}°, grid={n_grid}×{n_grid} = {total}점")

    for i, t0 in enumerate(t0s):
        for j, t1 in enumerate(t1s):
            done += 1
            if done % 40 == 0:
                print(f"    진행: {done}/{total} ({100*done/total:.0f}%)")
            if abs(t0 - t1) < 2.0:   # 너무 비슷하면 skip
                continue
            alphas = [t0, t1, 180.0 - t0, 180.0 - t1]
            res = build_and_analyze(alphas, rho0_deg, num_periods=num_periods)
            if res and res["helix"]:
                r_map[i, j] = res["helix"]["r"]
                g_map[i, j] = res["helix"]["g"]

    print(f"  스캔 완료. 유효 점: {np.sum(~np.isnan(r_map))}/{total}")
    return {"t0s": t0s, "t1s": t1s, "r_map": r_map, "g_map": g_map}


def scan_ne_design_space(alpha1_deg=100.0, n_grid=18, num_periods=10,
                          a3_range=(91.0, 160.0), a4_range=(91.0, 160.0),
                          geom_type="hyperbolic"):
    """
    Non-Euclidean 설계공간 스캔: (α₃, α₄) → (r, g, ρ*).

    α₁=α₂ 고정 (bird's foot 대칭).
    A = sin(α₄)/sin(α₃)  → α₃≠α₄이면 A≠1 (NE 고정점 존재).

    Returns
    -------
    dict{a3s, a4s, r_map, g_map, rhostar_map, A_map}
    """
    a3s = np.linspace(a3_range[0], a3_range[1], n_grid)
    a4s = np.linspace(a4_range[0], a4_range[1], n_grid)
    r_map      = np.full((n_grid, n_grid), np.nan)
    g_map      = np.full((n_grid, n_grid), np.nan)
    rs_map     = np.full((n_grid, n_grid), np.nan)  # ρ* 지도
    A_map      = np.full((n_grid, n_grid), np.nan)

    total = n_grid * n_grid
    done  = 0
    print(f"\n  [NE 설계공간 스캔] α₁=α₂={alpha1_deg}°, {geom_type}, grid={n_grid}×{n_grid}")

    for i, a3 in enumerate(a3s):
        for j, a4 in enumerate(a4s):
            done += 1
            if done % 40 == 0:
                print(f"    진행: {done}/{total} ({100*done/total:.0f}%)")

            alphas = [alpha1_deg, alpha1_deg, a3, a4]
            total_a = 2*alpha1_deg + a3 + a4
            if geom_type == "hyperbolic" and total_a <= 360.5:
                continue
            if geom_type == "elliptic"   and total_a >= 359.5:
                continue

            A_val, B_val = compute_AB(alphas)
            if A_val is None:
                continue
            cos_star, rho_star = compute_fixed_point(A_val, B_val)
            if rho_star is None:
                continue

            res = build_and_analyze(alphas, rho_star, num_periods=num_periods)
            if res and res["helix"]:
                r_map[i, j]  = res["helix"]["r"]
                g_map[i, j]  = res["helix"]["g"]
                rs_map[i, j] = rho_star
                A_map[i, j]  = A_val

    print(f"  스캔 완료. 유효 점: {np.sum(~np.isnan(r_map))}/{total}")
    return {"a3s": a3s, "a4s": a4s, "r_map": r_map, "g_map": g_map,
            "rhostar_map": rs_map, "A_map": A_map, "alpha1": alpha1_deg}


def plot_ff_scan_dashboard(scan_data, rho0_deg, save_path="ff_design_space.png"):
    """
    FF design space scan result — r-map, g-map, r-g scatter.
    """
    t0s, t1s = scan_data["t0s"], scan_data["t1s"]
    r_map, g_map = scan_data["r_map"], scan_data["g_map"]

    fig = plt.figure(figsize=(16, 5))
    fig.suptitle(rf"Flat-foldable Design Space Scan  ($\rho_0$ = {rho0_deg}°)\n"
                 r"$\theta=(\theta_0,\theta_1,180°{-}\theta_0,180°{-}\theta_1)\;\to\;(r,g)$ mapping",
                 fontsize=12, fontweight='bold')
    axes = [fig.add_subplot(1, 3, k+1) for k in range(3)]

    T0, T1 = np.meshgrid(t0s, t1s, indexing='ij')

    # r heatmap
    vm = ~np.isnan(r_map)
    if vm.any():
        c1 = axes[0].contourf(T0, T1, r_map, levels=20, cmap='plasma')
        plt.colorbar(c1, ax=axes[0], label='r  (helix radius)')
        axes[0].contour(T0, T1, r_map, levels=10, colors='white', linewidths=0.5, alpha=0.4)
    axes[0].set_title(r'Helix Radius  $r(\theta_0,\theta_1)$', fontsize=10, fontweight='bold')
    axes[0].set_xlabel(r'$\theta_0$ [deg]'); axes[0].set_ylabel(r'$\theta_1$ [deg]')

    # g heatmap
    if vm.any():
        c2 = axes[1].contourf(T0, T1, g_map, levels=20, cmap='RdYlGn',
                               vmin=-1, vmax=1)
        plt.colorbar(c2, ax=axes[1], label='g (normalized pitch)')
    axes[1].set_title(r'Normalized Pitch  $g(\theta_0,\theta_1)$', fontsize=10, fontweight='bold')
    axes[1].set_xlabel(r'$\theta_0$ [deg]'); axes[1].set_ylabel(r'$\theta_1$ [deg]')

    # r vs g scatter (색=t₀)
    rs_flat = r_map[vm]
    gs_flat = g_map[vm]
    t0_flat = T0[vm]
    sc = axes[2].scatter(rs_flat, gs_flat, c=t0_flat, cmap='coolwarm',
                          s=25, alpha=0.7)
    plt.colorbar(sc, ax=axes[2], label=r'$\theta_0$ [deg]')
    axes[2].set_xlabel(r'Helix radius  $r$'); axes[2].set_ylabel(r'Normalized pitch  $g$')
    axes[2].set_title(r'Achievable $(r,g)$ Region' + '\n' + r'(color = $\theta_0$)', fontsize=10, fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  [저장] {save_path}")
    return fig


def plot_ne_scan_dashboard(scan_data, save_path="ne_design_space.png"):
    """
    NE 설계공간 스캔 결과 시각화 — r-map, ρ*-map, A-map.
    """
    a3s, a4s = scan_data["a3s"], scan_data["a4s"]
    r_map, rs_map, A_map = scan_data["r_map"], scan_data["rhostar_map"], scan_data["A_map"]
    alpha1 = scan_data["alpha1"]

    fig = plt.figure(figsize=(17, 5))
    fig.suptitle(rf"Non-Euclidean Design Space Scan  ($\alpha_1=\alpha_2$={alpha1:.0f}°)\n"
                 r"$\alpha=(\alpha_1,\alpha_1,\alpha_3,\alpha_4)\;\to\;(r,\rho^*,A)$ mapping",
                 fontsize=12, fontweight='bold')
    axes = [fig.add_subplot(1, 4, k+1) for k in range(4)]

    A3, A4 = np.meshgrid(a3s, a4s, indexing='ij')
    vm = ~np.isnan(r_map)

    titles = [r'Helix Radius  $r$', r'Fixed Point  $\rho^*$ [deg]',
              r'$A$ 계수' + '\n' + r'($|A|<1$: 안정, $|A|>1$: 발산)', r'$r$ vs $\rho^*$ 산점도']

    if vm.any():
        c0 = axes[0].contourf(A3, A4, r_map, levels=20, cmap='plasma')
        plt.colorbar(c0, ax=axes[0], label='r')
        axes[0].set_title(titles[0], fontsize=9, fontweight='bold')
        axes[0].set_xlabel(r'$\alpha_3$ [deg]'); axes[0].set_ylabel(r'$\alpha_4$ [deg]')

        c1 = axes[1].contourf(A3, A4, rs_map, levels=20, cmap='RdYlBu_r')
        plt.colorbar(c1, ax=axes[1], label=r'$\rho^*$ [deg]')
        axes[1].set_title(titles[1], fontsize=9, fontweight='bold')
        axes[1].set_xlabel(r'$\alpha_3$ [deg]'); axes[1].set_ylabel(r'$\alpha_4$ [deg]')

        # A 지도: 0 중심
        A_abs = np.abs(A_map)
        c2 = axes[2].contourf(A3, A4, np.log10(np.clip(A_abs, 1e-3, 1e3)),
                               levels=20, cmap='seismic', vmin=-1, vmax=1)
        plt.colorbar(c2, ax=axes[2], label=r'$\log_{10}|A|$')
        axes[2].contour(A3, A4, A_abs, levels=[1.0],
                        colors='black', linewidths=2, linestyles='--')
        axes[2].text(0.5, 0.02, r'─── $|A|=1$ 경계', transform=axes[2].transAxes,
                     ha='center', fontsize=8, color='black')
        axes[2].set_title(titles[2], fontsize=9, fontweight='bold')
        axes[2].set_xlabel(r'$\alpha_3$ [deg]'); axes[2].set_ylabel(r'$\alpha_4$ [deg]')

        # r vs ρ* 산점도 (색=A)
        r_flat  = r_map[vm]
        rs_flat = rs_map[vm]
        A_flat  = A_map[vm]
        sc = axes[3].scatter(r_flat, rs_flat, c=A_flat, cmap='coolwarm',
                              vmin=-2, vmax=2, s=20, alpha=0.6)
        plt.colorbar(sc, ax=axes[3], label=r'$A$ coefficient')
        axes[3].set_xlabel(r'Helix radius  $r$')
        axes[3].set_ylabel(r'Fixed point  $\rho^*$ [deg]')
        axes[3].set_title(titles[3], fontsize=9, fontweight='bold')
        axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  [저장] {save_path}")
    return fig


# =============================================================================
#  Part 6: 역설계 엔진 (수렴 추적 포함)
# =============================================================================

class HelixInverseDesigner:
    """
    헬리컬 오리가미 역설계 클래스 — v2 개선판.

    [모드 1] Flat-foldable:
        θ = (t₀, t₁, 180-t₀, 180-t₁)  →  A=1, B=0 자동
        입력: (r*, g*, ρ₀)  →  출력: (t₀, t₁)

    [모드 2] Non-Euclidean bird's foot:
        α = (α₁, α₁, α₃, α₄)  →  A=sin(α₄)/sin(α₃) ≠ 1
        입력: (r*, g*, ρ*)  →  출력: (α₁, α₃, α₄)

    v2 추가: 수렴 이력 추적 (convergence_history)
    """

    def __init__(self, sigma=_SIGMA, num_periods=_N_PERIODS, lengths=_LENGTHS):
        self.sigma       = sigma
        self.num_periods = num_periods
        self.lengths     = lengths
        self.results     = {}
        self.convergence_history = {}   # {mode: [(params, residual), ...]}

    # ── 목적함수 (내부) ──────────────────────────────────────────────────────

    def _obj_ff(self, params, r_t, g_t, rho0):
        t0, t1 = params
        if not (3.0 < t0 < 87.0 and 3.0 < t1 < 87.0):
            return 1e6
        alphas = [t0, t1, 180.0-t0, 180.0-t1]
        res = build_and_analyze(alphas, rho0, self.sigma, 2,
                                 self.num_periods, self.lengths)
        if res is None or res["helix"] is None:
            return 1e6
        h = res["helix"]
        err = (h["r"] - r_t)**2 / (r_t**2 + 1e-4) + (h["g"] - g_t)**2
        # 수렴 이력 저장
        self.convergence_history.setdefault("ff", []).append(
            (tuple(params), float(err)))
        return err

    def _obj_ne(self, params, r_t, g_t, rho_star_t, geom_type, w_rho=0.5):
        a1, a3, a4 = params
        total = 2*a1 + a3 + a4
        if geom_type == "hyperbolic" and total <= 360.5:
            return 1e6 + 100*(360.5 - total)**2
        if geom_type == "elliptic"   and total >= 359.5:
            return 1e6 + 100*(total - 359.5)**2
        for a in [a1, a3, a4]:
            if not (5.0 < a < 175.0):
                return 1e6

        alphas = [a1, a1, a3, a4]
        A_val, B_val = compute_AB(alphas)
        if A_val is None: return 1e6

        cos_star, rho_star_calc = compute_fixed_point(A_val, B_val)
        if rho_star_calc is None: return 1e6 + 1000.0

        res = build_and_analyze(alphas, rho_star_calc, self.sigma, 2,
                                 self.num_periods, self.lengths)
        if res is None or res["helix"] is None: return 1e6

        h = res["helix"]
        err = ((h["r"] - r_t)**2 / (r_t**2 + 1e-4) +
               (h["g"] - g_t)**2 +
               w_rho*(rho_star_calc - rho_star_t)**2 / (rho_star_t**2 + 1e-4))
        self.convergence_history.setdefault("ne", []).append(
            (tuple(params), float(err)))
        return err

    # ── 모드 1: Flat-foldable ────────────────────────────────────────────────

    def inverse_flat_foldable(self, r_target, g_target, rho0_deg=-90.0, verbose=True):
        """
        FF 역설계: (r*, g*, ρ₀) → (θ₀, θ₁)

        수학적 구조:
          θ = (t₀, t₁, π-t₀, π-t₁)  →  A=1, B=0  (Class I-2)
          → 임의 ρ₀가 고정점
          → ρ₀는 '운동학적 선택' (연속 접힘 가능)
        """
        self.convergence_history["ff"] = []

        if verbose:
            print("\n" + "="*60)
            print("  [역설계 Mode 1] Flat-foldable Helix (Class I-2)")
            print(f"  목표: r*={r_target:.4f},  g*={g_target:.4f}")
            print(f"  운동학 상태: ρ₀={rho0_deg:.1f}°  ← 운동학적 자유도")
            print(f"  설계 변수: (θ₀, θ₁)")
            print("="*60)

        bounds = [(5.0, 85.0), (5.0, 85.0)]

        # Stage 1: Differential Evolution (전역)
        if verbose: print("  [1단계] Differential Evolution 전역 탐색...")
        best_val, best_x = np.inf, None
        try:
            res_de = differential_evolution(
                self._obj_ff, bounds,
                args=(r_target, g_target, rho0_deg),
                seed=42, maxiter=200, tol=1e-7,
                popsize=14, mutation=(0.5, 1.5), recombination=0.7
            )
            if res_de.fun < best_val:
                best_val, best_x = res_de.fun, res_de.x
        except Exception as e:
            if verbose: print(f"    [경고] {e}")

        # Stage 2: Nelder-Mead (지역 정밀화)
        if verbose: print("  [2단계] Nelder-Mead 정밀화 (다중 시작점)...")
        starts = [(60, 45), (45, 60), (70, 30), (30, 70), (50, 50)]
        if best_x is not None:
            starts = [tuple(best_x)] + starts
        for x0 in starts:
            try:
                res_nm = minimize(self._obj_ff, x0, method='Nelder-Mead',
                                   args=(r_target, g_target, rho0_deg),
                                   options={'xatol': 0.05, 'fatol': 1e-8, 'maxiter': 1200})
                if res_nm.fun < best_val:
                    best_val, best_x = res_nm.fun, res_nm.x
            except Exception:
                pass

        if best_x is None:
            if verbose: print("  [오류] 최적화 실패")
            return None

        t0, t1 = float(best_x[0]), float(best_x[1])
        alphas_opt = [t0, t1, 180.0-t0, 180.0-t1]
        A_val, B_val = compute_AB(alphas_opt)
        final = build_and_analyze(alphas_opt, rho0_deg, self.sigma, 2,
                                   self.num_periods, self.lengths)

        result = {
            "type": "flat_foldable", "alphas_deg": alphas_opt,
            "t0_deg": t0, "t1_deg": t1, "rho0_deg": rho0_deg,
            "r_target": r_target, "g_target": g_target,
            "r_achieved": final["helix"]["r"] if (final and final["helix"]) else None,
            "g_achieved": final["helix"]["g"] if (final and final["helix"]) else None,
            "A": A_val, "B": B_val,
            "residual": best_val, "dyn_class": "Class I (A=1, B=0)",
            "tessellator": final["tessellator"] if final else None,
            "full_result": final,
        }

        if verbose:
            print(f"\n  ✅ 완료!")
            print(f"     θ = ({t0:.2f}°, {t1:.2f}°, {180-t0:.2f}°, {180-t1:.2f}°)")
            print(f"     A={A_val:.6f} (이론: 1.0),  B={B_val:.6f} (이론: 0.0)")
            if final and final["helix"]:
                print(f"     r: {r_target:.4f} → {result['r_achieved']:.4f}")
                print(f"     g: {g_target:.4f} → {result['g_achieved']:.4f}")
            print(f"     최적화 이력: {len(self.convergence_history.get('ff',[]))} 호출")

        self.results["flat_foldable"] = result
        return result

    # ── 모드 2: Non-Euclidean ────────────────────────────────────────────────

    def inverse_non_euclidean(self, r_target, g_target, rho_star_target=60.0,
                               geom_type="hyperbolic", verbose=True):
        """
        NE 역설계: (r*, g*, ρ*) → (α₁, α₃, α₄)  [α₁=α₂ 조건]

        수학적 구조:
          α = (α₁, α₁, α₃, α₄)  →  A=sin(α₄)/sin(α₃)
          ρ* = arccos(B/(1-A))  ← geometry에 내재
          → ρ*는 '기하학적 선택' (접힘 상태 고정됨)
        """
        self.convergence_history["ne"] = []

        if verbose:
            print("\n" + "="*60)
            print(f"  [역설계 Mode 2] Non-Euclidean ({geom_type})")
            print(f"  목표: r*={r_target:.4f},  g*={g_target:.4f},  ρ*={rho_star_target:.1f}°")
            print(f"  설계 변수: (α₁, α₃, α₄)  [α₁=α₂ 대칭 조건]")
            print("="*60)

        if geom_type == "hyperbolic":
            bounds = [(91.0, 165.0), (91.0, 165.0), (91.0, 165.0)]
        else:
            bounds = [(10.0, 85.0), (10.0, 85.0), (10.0, 85.0)]

        # Stage 1: Differential Evolution
        if verbose: print("  [1단계] Differential Evolution 전역 탐색...")
        best_val, best_x = np.inf, None
        try:
            res_de = differential_evolution(
                self._obj_ne, bounds,
                args=(r_target, g_target, rho_star_target, geom_type, 0.5),
                seed=42, maxiter=400, tol=1e-8,
                popsize=18, mutation=(0.5, 1.5), recombination=0.7
            )
            if res_de.fun < best_val:
                best_val, best_x = res_de.fun, res_de.x
        except Exception as e:
            if verbose: print(f"    [경고] {e}")

        # Stage 2: Nelder-Mead
        if verbose: print("  [2단계] Nelder-Mead 정밀화...")
        try:
            if best_x is not None:
                res_nm = minimize(self._obj_ne, best_x, method='Nelder-Mead',
                                   args=(r_target, g_target, rho_star_target, geom_type, 0.5),
                                   options={'xatol': 0.02, 'fatol': 1e-9, 'maxiter': 2000})
                if res_nm.fun < best_val:
                    best_val, best_x = res_nm.fun, res_nm.x
        except Exception:
            pass

        if best_x is None:
            if verbose: print("  [오류] 최적화 실패")
            return None

        a1, a3, a4 = float(best_x[0]), float(best_x[1]), float(best_x[2])
        alphas_opt = [a1, a1, a3, a4]
        A_val, B_val = compute_AB(alphas_opt)
        cos_star, rho_star_actual = compute_fixed_point(A_val, B_val)
        delta = 2*a1 + a3 + a4 - 360.0
        cls, _, cnum = classify_AB(A_val, B_val)
        final = build_and_analyze(alphas_opt, rho_star_actual or rho_star_target,
                                   self.sigma, 2, self.num_periods, self.lengths)

        result = {
            "type": "non_euclidean", "geom_type": geom_type,
            "alphas_deg": alphas_opt, "alpha1_deg": a1,
            "alpha3_deg": a3, "alpha4_deg": a4, "delta_deg": delta,
            "r_target": r_target, "g_target": g_target,
            "rho_star_target": rho_star_target,
            "rho_star_actual": rho_star_actual,
            "r_achieved": final["helix"]["r"] if (final and final["helix"]) else None,
            "g_achieved": final["helix"]["g"] if (final and final["helix"]) else None,
            "A": A_val, "B": B_val,
            "residual": best_val, "dyn_class": cls,
            "tessellator": final["tessellator"] if final else None,
            "full_result": final,
        }

        if verbose:
            print(f"\n  ✅ 완료!")
            print(f"     α = ({a1:.2f}°, {a1:.2f}°, {a3:.2f}°, {a4:.2f}°)")
            print(f"     δ = {delta:.2f}°  (Σα={2*a1+a3+a4:.1f}°)")
            print(f"     A={A_val:.4f},  B={B_val:.4f}")
            if rho_star_actual:
                print(f"     ρ*: 목표={rho_star_target:.1f}° → 달성={rho_star_actual:.2f}°")
            if final and final["helix"]:
                print(f"     r: {r_target:.4f} → {result['r_achieved']:.4f}")
                print(f"     g: {g_target:.4f} → {result['g_achieved']:.4f}")
            print(f"     최적화 이력: {len(self.convergence_history.get('ne',[]))} 호출")

        self.results["non_euclidean"] = result
        return result


# =============================================================================
#  Part 7: 수렴 추적 시각화 (신규)
# =============================================================================

def plot_convergence_history(designer, mode='ff', save_path=None):
    """
    역설계 최적화 수렴 이력을 설계공간에 시각화.

    - 각 최적화 호출에서의 (params, residual) 이력 플롯
    - residual의 log-scale 수렴 그래프
    """
    hist = designer.convergence_history.get(mode, [])
    if not hist:
        print(f"[경고] {mode} 모드의 수렴 이력이 없습니다.")
        return None

    params_arr = np.array([h[0] for h in hist])
    resids = np.array([h[1] for h in hist])

    n_params = params_arr.shape[1]
    fig, axes = plt.subplots(1, n_params + 1, figsize=(5*(n_params+1), 4))
    fig.suptitle(f"Inverse Design Convergence — {'Flat-foldable' if mode=='ff' else 'Non-Euclidean'} mode",
                 fontsize=11, fontweight='bold')

    # 각 파라미터의 탐색 경로
    param_labels_ff = [r'$\theta_0$ [deg]', r'$\theta_1$ [deg]']
    param_labels_ne = [r'$\alpha_1$ [deg]', r'$\alpha_3$ [deg]', r'$\alpha_4$ [deg]']
    plabels = param_labels_ff if mode == 'ff' else param_labels_ne

    for k in range(n_params):
        ax = axes[k]
        sc = ax.scatter(range(len(resids)), params_arr[:, k],
                        c=np.log10(np.clip(resids, 1e-10, 1e6)),
                        cmap='plasma_r', s=8, alpha=0.5)
        ax.set_xlabel('Optimization call #')
        ax.set_ylabel(plabels[k] if k < len(plabels) else f'param[{k}]')
        ax.set_title(f'{plabels[k] if k < len(plabels) else f"param[{k}]"} search path')
        ax.grid(True, alpha=0.3)

    # 잔차 수렴 그래프
    ax_r = axes[-1]
    # 누적 최솟값 (monotone decreasing)
    best_so_far = np.minimum.accumulate(resids)
    ax_r.semilogy(resids, '.', alpha=0.3, color='gray', ms=3, label='each call')
    ax_r.semilogy(best_so_far, '-', color='red', lw=2.0, label='best so far')
    ax_r.set_xlabel('Optimization call #')
    ax_r.set_ylabel('Residual (log scale)')
    ax_r.set_title('Residual Convergence')
    ax_r.legend(fontsize=8)
    ax_r.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [저장] {save_path}")
    return fig


# =============================================================================
#  Part 8: 민감도 분석 (신규)
# =============================================================================

def sensitivity_analysis(result, delta_pct=5.0, num_periods=10):
    """
    최적해 주변 국소 민감도 분석.

    각 설계 파라미터를 ±δ% 섭동 시 (r, g)의 변화를 계산.
    → 어느 파라미터가 헬릭스 형상에 가장 민감한가?

    Parameters
    ----------
    result   : 역설계 결과 dict
    delta_pct: 섭동 크기 [%]

    Returns
    -------
    sensitivity_table : list of dict
    """
    rtype = result.get("type", "")
    alphas_opt = result["alphas_deg"]
    rho0 = (result.get("rho0_deg") or
            result.get("rho_star_actual") or
            result.get("rho_star_target") or -90.0)

    baseline = build_and_analyze(alphas_opt, rho0, num_periods=num_periods)
    if baseline is None or baseline["helix"] is None:
        print("[민감도] 기준 계산 실패")
        return []

    r0 = baseline["helix"]["r"]
    g0 = baseline["helix"]["g"]

    if rtype == "flat_foldable":
        param_names = ['θ₀', 'θ₁']
        param_indices = [0, 1]  # t0, t1
        def make_alphas(idx, val):
            t = list(alphas_opt)
            t[idx] = val
            if idx == 0:
                t[2] = 180.0 - val
            elif idx == 1:
                t[3] = 180.0 - val
            return t
    else:
        param_names = ['α₁(=α₂)', 'α₃', 'α₄']
        param_indices = [0, 2, 3]
        def make_alphas(idx, val):
            t = list(alphas_opt)
            t[idx] = val
            if idx == 0:
                t[1] = val  # α₁=α₂ 조건 유지
            return t

    table = []
    for name, idx in zip(param_names, param_indices):
        v0 = alphas_opt[idx]
        delta = v0 * delta_pct / 100.0
        sens_r, sens_g = [], []

        for sign in [+1, -1]:
            v_new = v0 + sign * delta
            a_new = make_alphas(idx, v_new)
            res = build_and_analyze(a_new, rho0, num_periods=num_periods)
            if res and res["helix"]:
                dr = (res["helix"]["r"] - r0) / delta * sign
                dg = (res["helix"]["g"] - g0) / delta * sign
                sens_r.append(dr)
                sens_g.append(dg)

        if sens_r:
            table.append({
                "param": name,
                "value": v0,
                "dr_dparam": float(np.mean(sens_r)),
                "dg_dparam": float(np.mean(sens_g)),
                "|dr/r0|_per_%": abs(float(np.mean(sens_r))) * delta_pct / 100.0 / (r0 + 1e-8),
                "|dg/g0|_per_%": abs(float(np.mean(sens_g))) * delta_pct / 100.0 / (abs(g0) + 1e-8),
            })

    print(f"\n  ── 민감도 분석 (섭동 ±{delta_pct}%) ──────────────────")
    print(f"  기준값: r₀={r0:.4f},  g₀={g0:.4f}")
    print(f"  {'파라미터':<12} {'기준값':>8} {'∂r/∂α':>10} {'∂g/∂α':>10}"
          f" {'|Δr|/r₀ per%':>14} {'|Δg|/g₀ per%':>14}")
    print("  " + "-"*70)
    for row in table:
        print(f"  {row['param']:<12} {row['value']:>8.2f} {row['dr_dparam']:>10.4f}"
              f" {row['dg_dparam']:>10.4f} {row['|dr/r0|_per_%']:>14.4f}"
              f" {row['|dg/g0|_per_%']:>14.4f}")
    return table


# =============================================================================
#  Part 9: 3D 렌더링 유틸
# =============================================================================

def render_strip_3d(tess, rho0_deg, ax, palette, alpha=0.78):
    """Strip 3D 렌더링 (판넬 + 백본)"""
    if tess is None:
        return
    if len(tess.global_faces) == 0:
        tess.compute_strip_kinematics(np.deg2rad(rho0_deg))

    faces_3d, colors_3d = [], []
    for ui, unit_faces in enumerate(tess.global_faces):
        for fi, tri in enumerate(unit_faces):
            faces_3d.append(tri)
            colors_3d.append(palette[fi % len(palette)])

    if faces_3d:
        poly = Poly3DCollection(faces_3d, alpha=alpha, linewidth=0.3, edgecolor='#555')
        poly.set_facecolor(colors_3d)
        ax.add_collection3d(poly)

    crease_res = tess._build_crease_lines()
    backbone = crease_res[0] if (crease_res and crease_res[0] is not None) else None
    if backbone is not None and len(backbone) > 1:
        ax.plot(backbone[:, 0], backbone[:, 1], backbone[:, 2],
                '-', color='crimson', lw=2.2, zorder=5)

    all_pts = np.array([v for gv in tess.global_vertices for v in gv])
    if len(all_pts) > 0:
        center = (all_pts.max(0) + all_pts.min(0)) / 2
        rng = max((all_pts.max(0) - all_pts.min(0)).max(), 1.0) * 0.58
        ax.set_xlim(center[0]-rng, center[0]+rng)
        ax.set_ylim(center[1]-rng, center[1]+rng)
        ax.set_zlim(center[2]-rng, center[2]+rng)
    ax.set_xlabel('X', fontsize=7); ax.set_ylabel('Y', fontsize=7)
    ax.set_zlabel('Z', fontsize=7)


# =============================================================================
#  Part 10: 종합 대시보드 — v2 버전
# =============================================================================

def plot_dashboard_v2(ff_result, ne_result, designer=None,
                      save_path="helix_inverse_design_v2.png"):
    """
    v2 종합 대시보드.

    Layout (3행 4열):
      Row 0: 3D FF Helix | cos-space map (FF) | 3D NE Helix | cos-space map (NE)
      Row 1: FF ρ-cobweb  | (A,B) 위상도      | NE ρ-cobweb  | 설계 공간 (r vs g)
      Row 2: Sector (FF)  | Sector (NE)        | Kinematic portrait | 요약
    """
    fig = plt.figure(figsize=(22, 15))
    fig.patch.set_facecolor(_C_BG)
    fig.suptitle(
        "Helical Origami Inverse Design Dashboard v2\n"
        r"Flat-foldable (Class I-2)  vs  Non-Euclidean ($\alpha_1=\alpha_2$ symmetry)",
        fontsize=14, fontweight='bold', y=0.993,
        bbox=dict(facecolor='#EFF6FF', edgecolor='#2563EB', boxstyle='round,pad=0.3')
    )

    gs = mgs.GridSpec(3, 4, figure=fig, hspace=0.54, wspace=0.40,
                      top=0.955, bottom=0.04, left=0.04, right=0.97)

    # ── Row 0: 3D + cos-공간 맵 ───────────────────────────────────────────────
    ax3d_ff  = fig.add_subplot(gs[0, 0], projection='3d')
    ax_cos_ff = fig.add_subplot(gs[0, 1])
    ax3d_ne  = fig.add_subplot(gs[0, 2], projection='3d')
    ax_cos_ne = fig.add_subplot(gs[0, 3])

    if ff_result and ff_result.get("tessellator"):
        t0, t1 = ff_result["t0_deg"], ff_result["t1_deg"]
        render_strip_3d(ff_result["tessellator"], ff_result["rho0_deg"],
                        ax3d_ff, _PAL_FF)
        ax3d_ff.set_title(
            rf"[FF] $\theta$=({t0:.0f}°,{t1:.0f}°,{180-t0:.0f}°,{180-t1:.0f}°)\n"
            rf"$\rho_0$={ff_result['rho0_deg']:.0f}°  $r$={ff_result.get('r_achieved',0):.3f}",
            fontsize=8.5, color='#1e3a5f', fontweight='bold'
        )

    if ff_result:
        A_ff = ff_result.get("A") or 1.0
        B_ff = ff_result.get("B") or 0.0
        # cos-공간 맵에서 rho0의 cos를 x0으로
        x0_ff = np.cos(np.radians(ff_result.get("rho0_deg", -90)))
        plot_cosspace_linear_map(A_ff, B_ff, ax_cos_ff,
                                  x0=x0_ff, color=_C_FF, label='(FF)')
        ax_cos_ff.set_title(
            rf"$\cos$-space Linear Map (FF)\n$A$={A_ff:.5f} $\approx$ 1.0  — every $x_0$ is a fixed point",
            fontsize=8.5, color='#1e3a5f', fontweight='bold'
        )

    if ne_result and ne_result.get("tessellator"):
        a1 = ne_result["alpha1_deg"]
        a3, a4 = ne_result["alpha3_deg"], ne_result["alpha4_deg"]
        rho0_ne = ne_result.get("rho_star_actual") or 60.0
        render_strip_3d(ne_result["tessellator"], rho0_ne, ax3d_ne, _PAL_NE)
        ax3d_ne.set_title(
            rf"[NE] $\alpha$=({a1:.0f}°,{a1:.0f}°,{a3:.0f}°,{a4:.0f}°)\n"
            rf"$\rho^*$={rho0_ne:.1f}°  $r$={ne_result.get('r_achieved',0):.3f}",
            fontsize=8.5, color='#7c2d12', fontweight='bold'
        )

    if ne_result:
        A_ne = ne_result.get("A")
        B_ne = ne_result.get("B")
        if A_ne and B_ne:
            cos_star, rho_star = compute_fixed_point(A_ne, B_ne)
            x0_ne = cos_star if (cos_star is not None) else 0.0
            plot_cosspace_linear_map(float(A_ne), float(B_ne), ax_cos_ne,
                                      x0=x0_ne * 0.5, color=_C_NE, label='(NE)')
            ax_cos_ne.set_title(
                rf"$\cos$-space Linear Map (NE)\n$A$={float(A_ne):.4f}  — single fixed point $x^*$",
                fontsize=8.5, color='#7c2d12', fontweight='bold'
            )

    # ── Row 1: ρ-cobweb | (A,B) 위상도 | NE ρ-cobweb | r-g 설계공간 ────────
    ax_pcob_ff = fig.add_subplot(gs[1, 0])
    ax_ab      = fig.add_subplot(gs[1, 1])
    ax_pcob_ne = fig.add_subplot(gs[1, 2])
    ax_rg      = fig.add_subplot(gs[1, 3])

    if ff_result:
        A_ff = ff_result.get("A") or 1.0
        B_ff = ff_result.get("B") or 0.0
        plot_rho_space_cobweb(A_ff, B_ff, ax_pcob_ff,
                               rho0_deg=ff_result.get("rho0_deg", -90) * 0.7,
                               color=_C_FF, label='(FF)')
        ax_pcob_ff.set_title(r"$\rho$-space cobweb (FF)" + "\n" + r"appears nonlinear — compare with $\cos$-space",
                              fontsize=8.5, color='#1e3a5f', fontweight='bold')

    # (A,B) 위상도 — 두 설계점 강조
    highlight = []
    if ff_result and ff_result.get("A"):
        highlight.append((float(ff_result["A"]), float(ff_result["B"]),
                          "FF", _C_FF))
    if ne_result and ne_result.get("A"):
        highlight.append((float(ne_result["A"]), float(ne_result["B"]),
                          "NE", _C_NE))
    plot_AB_phase_diagram(ax_ab, highlight_pts=highlight if highlight else None)

    if ne_result:
        A_ne = ne_result.get("A")
        B_ne = ne_result.get("B")
        if A_ne and B_ne:
            rho_s = ne_result.get("rho_star_actual") or 60.0
            plot_rho_space_cobweb(float(A_ne), float(B_ne), ax_pcob_ne,
                                   rho0_deg=rho_s * 0.5,
                                   color=_C_NE, label='(NE)')
            ax_pcob_ne.set_title(r"$\rho$-space cobweb (NE)" + "\n" + "converging cobweb — stable fixed point",
                                  fontsize=8.5, color='#7c2d12', fontweight='bold')

    # r vs g 설계공간 (FF 궤적 + NE 고정점)
    if ff_result and ff_result.get("alphas_deg"):
        rho_scan = np.linspace(-155, -5, 28)
        rs_ff, gs_ff = [], []
        for rho0 in rho_scan:
            tmp = build_and_analyze(ff_result["alphas_deg"], rho0, num_periods=10)
            if tmp and tmp["helix"]:
                rs_ff.append(tmp["helix"]["r"])
                gs_ff.append(tmp["helix"]["g"])
            else:
                rs_ff.append(np.nan); gs_ff.append(np.nan)
        vm = ~np.isnan(rs_ff)
        ax_rg.plot(np.array(rs_ff)[vm], np.array(gs_ff)[vm],
                   'b-o', ms=3.5, lw=1.8, alpha=0.8, color=_C_FF, label='FF kinematic path')
        if ff_result.get("r_achieved"):
            ax_rg.plot(ff_result["r_achieved"], ff_result["g_achieved"],
                       '*', color=_C_FF, ms=16, zorder=7, label=r'FF inverse design')

    if ne_result and ne_result.get("r_achieved"):
        ax_rg.plot(ne_result["r_achieved"], ne_result["g_achieved"],
                   'D', color=_C_NE, ms=12, zorder=7,
                   label=rf'NE inverse design ($\rho^*$={ne_result.get("rho_star_actual",0):.0f}°)')

    if ff_result:
        ax_rg.plot(ff_result["r_target"], ff_result["g_target"],
                   'k+', ms=18, mew=3, zorder=8, label=r'target $(r^*,\,g^*)$')

    ax_rg.set_xlabel(r'Helix radius  $r$', fontsize=9)
    ax_rg.set_ylabel(r'Normalized pitch  $g$', fontsize=9)
    ax_rg.set_title(r'Design Space: $r$ vs $g$' + '\n' + 'FF continuous path  vs  NE fixed point', fontsize=9, fontweight='bold')
    ax_rg.legend(fontsize=7.5); ax_rg.grid(True, alpha=0.3)

    # ── Row 2: Sector pies | Kinematic portrait | 요약 ────────────────────────
    ax_pie_ff = fig.add_subplot(gs[2, 0])
    ax_pie_ne = fig.add_subplot(gs[2, 1])
    ax_kp     = fig.add_subplot(gs[2, 2])
    ax_sum    = fig.add_subplot(gs[2, 3])

    # Sector pies
    def _sector_pie(alphas, ax, title):
        labels = [rf'$\alpha_1$\n{alphas[0]:.1f}°', rf'$\alpha_2$\n{alphas[1]:.1f}°',
                  rf'$\alpha_3$\n{alphas[2]:.1f}°', rf'$\alpha_4$\n{alphas[3]:.1f}°']
        clrs = ['#BFDBFE', '#A7F3D0', '#FCA5A5', '#D1D5DB']
        wedges, texts, autotexts = ax.pie(
            [abs(a) for a in alphas], labels=labels, colors=clrs,
            explode=[0.05]*4, autopct='%1.0f%%', startangle=90,
            textprops={'fontsize': 7.5}
        )
        t = sum(alphas)
        g = "Euc" if np.isclose(t, 360) else ("Hyp" if t > 360 else "Ell")
        ax.set_title(f"{title}" + rf"\n$\Sigma\alpha$={t:.0f}° ({g})", fontsize=8, fontweight='bold')

    if ff_result and ff_result.get("alphas_deg"):
        _sector_pie(ff_result["alphas_deg"], ax_pie_ff, "Sector (FF)")
    if ne_result and ne_result.get("alphas_deg"):
        _sector_pie(ne_result["alphas_deg"], ax_pie_ne,
                    rf"Sector (NE)\n$\delta$={ne_result['delta_deg']:.1f}°")

    # Kinematic portrait (r vs ρ₀)
    if ff_result and ff_result.get("alphas_deg"):
        rho_scan2 = np.linspace(-155, -5, 22)
        rs2 = []
        for rho0 in rho_scan2:
            tmp = build_and_analyze(ff_result["alphas_deg"], rho0, num_periods=10)
            rs2.append(tmp["helix"]["r"] if (tmp and tmp["helix"]) else np.nan)
        vm2 = ~np.isnan(rs2)
        ax_kp.plot(rho_scan2[vm2], np.array(rs2)[vm2], 'o-',
                   color=_C_FF, ms=4, lw=2, label=r'FF: $r(\rho_0)$')
        ax_kp.axhline(ff_result["r_target"], color='navy', ls='--', lw=1.5, alpha=0.7, label=r'$r^*$ target')
        if ff_result.get("rho0_deg"):
            ax_kp.axvline(ff_result["rho0_deg"], color='green', ls=':', lw=1.5, alpha=0.7,
                          label=rf'$\rho_0$={ff_result["rho0_deg"]:.0f}° (chosen)')
    if ne_result and ne_result.get("r_achieved"):
        ax_kp.axhline(ne_result["r_achieved"], color=_C_NE, ls='-.', lw=2.0,
                      label=rf'NE: $r$={ne_result["r_achieved"]:.3f} (fixed)')
    ax_kp.set_xlabel(r'$\rho_0$ [deg]', fontsize=9)
    ax_kp.set_ylabel(r'Helix radius  $r$', fontsize=9)
    ax_kp.set_title(r'Kinematic Portrait: $r$ vs $\rho_0$' + '\n' + '(FF: continuous,  NE: fixed)',
                    fontsize=9, fontweight='bold')
    ax_kp.legend(fontsize=7.5); ax_kp.grid(True, alpha=0.3)

    # 요약 텍스트
    ax_sum.axis('off')
    lines = ["━━ 역설계 결과 요약 (v2) ━━\n"]
    if ff_result:
        t0, t1 = ff_result["t0_deg"], ff_result["t1_deg"]
        r_a = ff_result.get("r_achieved") or float('nan')
        g_a = ff_result.get("g_achieved") or float('nan')
        A_, B_ = float(ff_result.get("A", 1)), float(ff_result.get("B", 0))
        lines += [
            "▶ Flat-foldable [Class I-2]",
            f"  θ=({t0:.1f}°,{t1:.1f}°,{180-t0:.1f}°,{180-t1:.1f}°)",
            f"  A={A_:.6f}  B={B_:.6f}",
            f"  ρ₀={ff_result['rho0_deg']:.0f}° ← 운동학적 선택",
            f"  r: {ff_result['r_target']:.3f}→{r_a:.3f}",
            f"  g: {ff_result['g_target']:.3f}→{g_a:.3f}",
            f"  잔차: {ff_result['residual']:.2e}\n",
        ]
    if ne_result:
        a1 = ne_result["alpha1_deg"]
        a3, a4 = ne_result["alpha3_deg"], ne_result["alpha4_deg"]
        r_a = ne_result.get("r_achieved") or float('nan')
        g_a = ne_result.get("g_achieved") or float('nan')
        rho_s = ne_result.get("rho_star_actual") or float('nan')
        A_, B_ = float(ne_result.get("A", 0)), float(ne_result.get("B", 0))
        lines += [
            f"▶ Non-Euclidean [{ne_result.get('geom_type','')}]",
            f"  α=({a1:.1f}°,{a1:.1f}°,{a3:.1f}°,{a4:.1f}°)",
            f"  δ={ne_result['delta_deg']:.2f}° (각도 결손)",
            f"  A={A_:.4f}  B={B_:.4f}",
            f"  ρ*={rho_s:.1f}° ← 기하학적 선택",
            f"  r: {ne_result['r_target']:.3f}→{r_a:.3f}",
            f"  g: {ne_result['g_target']:.3f}→{g_a:.3f}",
            f"  잔차: {ne_result['residual']:.2e}\n",
        ]
    lines += [
        "━━ 핵심 차이 ━━",
        "FF: cos-map이 항등사상 (기울기=1)",
        "    어떤 ρ₀도 헬릭스 형상 → '접는 스프링'",
        "NE: cos-map에 수렴 고정점 존재",
        "    ρ₀→ρ* 수렴 → 'geometry에 잠긴 형상'",
    ]
    ax_sum.text(0.02, 0.98, "\n".join(lines),
                transform=ax_sum.transAxes, fontsize=8.2,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#F0F9FF',
                          edgecolor='#2563EB', alpha=0.9))

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=_C_BG)
    print(f"\n  [저장] {save_path}")
    return fig


# =============================================================================
#  Part 11: 대화형 슬라이더 (신규)
# =============================================================================

def launch_interactive_explorer(alphas_ff_deg, rho0_init=-90.0):
    """
    Flat-foldable strip 대화형 ρ₀ 슬라이더.

    - 슬라이더로 ρ₀를 바꾸면 → 3D 헬릭스 실시간 업데이트
    - cos-공간 맵의 x₀ 위치도 함께 이동
    - 현재 (r, g, A, B) 실시간 표시

    사용법:
        alphas = [70, 20, 110, 160]  # FF 조건
        launch_interactive_explorer(alphas, rho0_init=-90)
    """
    A_val, B_val = compute_AB(alphas_ff_deg)
    print(f"\n  대화형 탐색기 시작 — θ={alphas_ff_deg}")
    print(f"  A={A_val:.4f},  B={B_val:.4f}")
    print(f"  (슬라이더로 ρ₀를 바꾸면 3D 헬릭스가 실시간 업데이트됩니다)\n")

    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor(_C_BG)
    fig.suptitle(r"Flat-foldable Helix Interactive Explorer" + "\n" + r"Explore kinematic states via $\rho_0$ slider",
                 fontsize=11, fontweight='bold')
    plt.subplots_adjust(bottom=0.18, hspace=0.4, wspace=0.35)

    ax_3d  = fig.add_subplot(1, 3, 1, projection='3d')
    ax_cos = fig.add_subplot(1, 3, 2)
    ax_rg  = fig.add_subplot(1, 3, 3)

    ax_sl  = plt.axes([0.15, 0.06, 0.70, 0.03])
    slider = Slider(ax_sl, r'$\rho_0$ [deg]', -170, -5, valinit=rho0_init, color=_C_FF)

    # cos-공간 맵은 변하지 않으므로 한 번만 그림
    plot_cosspace_linear_map(A_val, B_val, ax_cos, x0=None, color=_C_FF)
    fp_marker, = ax_cos.plot([], [], 'o', color=_C_CW, ms=10, zorder=9,
                              label=r'current $x_0=\cos(\rho_0)$')
    ax_cos.legend(fontsize=7.5)

    # r-g 궤적 사전 계산
    rho_scan = np.linspace(-160, -5, 30)
    rs_traj, gs_traj = [], []
    print("  (r,g) 궤적 사전 계산 중...")
    for rho0 in rho_scan:
        tmp = build_and_analyze(alphas_ff_deg, rho0, num_periods=10)
        if tmp and tmp["helix"]:
            rs_traj.append(tmp["helix"]["r"])
            gs_traj.append(tmp["helix"]["g"])
        else:
            rs_traj.append(np.nan)
            gs_traj.append(np.nan)
    vm = ~np.isnan(rs_traj)
    ax_rg.plot(np.array(rs_traj)[vm], np.array(gs_traj)[vm],
               '-', color=_C_FF, lw=2, alpha=0.6, label='FF path')
    cur_pt, = ax_rg.plot([], [], '*', color='red', ms=14, zorder=7, label='current state')
    ax_rg.set_xlabel(r'$r$  (helix radius)'); ax_rg.set_ylabel(r'$g$  (normalized pitch)')
    ax_rg.set_title(r'$(r,g)$ phase trajectory', fontsize=9, fontweight='bold')
    ax_rg.legend(fontsize=8); ax_rg.grid(True, alpha=0.3)

    # 3D 초기 렌더링
    poly_obj  = Poly3DCollection([], alpha=0.8, linewidth=0.4, edgecolor='#555')
    ax_3d.add_collection3d(poly_obj)
    line_bb,  = ax_3d.plot([], [], [], 'r-', lw=2.2, zorder=5, label='backbone')
    info_text = ax_3d.text2D(0.02, 0.96, '', transform=ax_3d.transAxes,
                              fontsize=8, va='top',
                              bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))

    current_tess = [None]

    def update(val=None):
        rho0 = slider.val
        res = build_and_analyze(alphas_ff_deg, rho0, num_periods=15)

        ax_3d.cla()
        ax_3d.set_xlabel('X', fontsize=7); ax_3d.set_ylabel('Y', fontsize=7)
        ax_3d.set_zlabel('Z', fontsize=7)

        if res and res["tessellator"] and len(res["tessellator"].global_faces) > 0:
            tess = res["tessellator"]
            # 3D 렌더링
            faces_3d, colors_3d = [], []
            for ui, uf in enumerate(tess.global_faces):
                for fi, tri in enumerate(uf):
                    faces_3d.append(tri)
                    colors_3d.append(_PAL_FF[fi % len(_PAL_FF)])

            poly = Poly3DCollection(faces_3d, alpha=0.8, linewidth=0.3, edgecolor='#555')
            poly.set_facecolor(colors_3d)
            ax_3d.add_collection3d(poly)

            cr = tess._build_crease_lines()
            if cr[0] is not None:
                bb = cr[0]
                ax_3d.plot(bb[:,0], bb[:,1], bb[:,2], 'r-', lw=2.2)

            all_pts = np.array([v for gv in tess.global_vertices for v in gv])
            center  = (all_pts.max(0)+all_pts.min(0))/2
            rng     = max((all_pts.max(0)-all_pts.min(0)).max(), 1.0)*0.6
            ax_3d.set_xlim(center[0]-rng, center[0]+rng)
            ax_3d.set_ylim(center[1]-rng, center[1]+rng)
            ax_3d.set_zlim(center[2]-rng, center[2]+rng)

            h = res["helix"]
            if h:
                ax_3d.set_title(
                    rf"$\rho_0$={rho0:.1f}°\n$r$={h['r']:.4f},  $g$={h['g']:.4f}",
                    fontsize=9, fontweight='bold'
                )
                # r-g 현재점
                cur_pt.set_data([h["r"]], [h["g"]])

        # cos-공간 x₀ 마커 업데이트
        x0_cur = np.cos(np.radians(rho0))
        fp_marker.set_data([x0_cur], [x0_cur])
        fp_marker.set_label(rf'$x_0=\cos$(${rho0:.0f}$°)={x0_cur:.3f}')
        ax_cos.legend(fontsize=7.5)

        fig.canvas.draw_idle()

    slider.on_changed(update)
    update()

    ax_rg.legend(fontsize=8)
    plt.show()
    return fig


# =============================================================================
#  Part 12: 메인 데모 함수
# =============================================================================

def demo_forward_analysis():
    """순설계 분석 + A/B 위상도"""
    print("\n" + "="*65)
    print("  [순설계 분석] sector angles → A,B,ρ*,헬릭스 파라미터")
    print("="*65)

    examples = [
        {"name": "FF Class I-2 (Miura계열)",
         "alphas": [70, 20, 110, 160], "rho0": -140},
        {"name": "NE bird's foot (쌍곡, α₁=α₂)",
         "alphas": [112.5, 112.5, 90, 135], "rho0": -80},
        {"name": "Class II (수렴, |A|<1)",
         "alphas": [95, 60, 85, 120], "rho0": -80},
        {"name": "Class III (발산, |A|>1)",
         "alphas": [60, 95, 120, 85], "rho0": -80},
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(r'Forward Analysis: $(A,B)$ Phase Diagram & Dynamics Classification', fontsize=11, fontweight='bold')

    highlight_pts = []
    for ex in examples:
        A_val, B_val = compute_AB(ex["alphas"])
        cls, stable, cnum = classify_AB(A_val, B_val)
        cos_star, rho_star = compute_fixed_point(A_val, B_val)
        geom, delta = get_geom_type(ex["alphas"])
        ff = is_kawasaki(ex["alphas"])

        print(f"\n  ▶ {ex['name']}")
        print(f"    α = {ex['alphas']}  (Σα={sum(ex['alphas'])}°, {geom})")
        print(f"    A={A_val:.4f},  B={B_val:.4f}")
        print(f"    동역학: {cls}")
        if rho_star: print(f"    고정점: ρ*={rho_star:.2f}°  (cos ρ*={cos_star:.4f})")
        print(f"    Kawasaki(FF): {'✓' if ff else '✗'}")

        res = build_and_analyze(ex["alphas"], ex["rho0"], num_periods=15)
        if res and res["helix"]:
            h = res["helix"]
            print(f"    나선: r={h['r']:.4f},  g={h['g']:.4f}")
            print(f"    유효 유닛: {res['valid_units']}")

        clr = [_C_FF, _C_NE, 'green', 'purple'][len(highlight_pts)]
        if A_val and B_val:
            highlight_pts.append((float(A_val), float(B_val), ex["name"][:12], clr))

        # cos-공간 맵
    plot_AB_phase_diagram(axes[0], highlight_pts=highlight_pts)

    # cos-공간 비교
    for ex, clr in zip(examples[:2], [_C_FF, _C_NE]):
        A_val, B_val = compute_AB(ex["alphas"])
        x0 = np.cos(np.radians(ex["rho0"]))
        plot_cosspace_linear_map(A_val, B_val, axes[1], x0=x0, n_cobweb=15,
                                  color=clr, label=ex["name"][:15], show_stability=False)
    axes[1].set_title(r'$\cos$-space Linear Map Comparison' + '\n' + '(FF vs NE)', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig("forward_analysis_v2.png", dpi=150, bbox_inches='tight')
    print(f"\n  [저장] forward_analysis_v2.png")
    return fig


def run_full_demo(r_target=2.5, g_target=0.30,
                  rho0_ff=-95.0, rho_star_ne=70.0):
    """전체 역설계 데모: FF + NE 동시 실행 후 v2 대시보드"""
    print("\n" + "="*65)
    print("  헬리컬 오리가미 역설계 비교 데모 v2")
    print(f"  목표: r*={r_target:.3f},  g*={g_target:.3f}")
    print("="*65)

    designer = HelixInverseDesigner(sigma=1, num_periods=15, lengths=(1, 1, 1))

    ff_result = designer.inverse_flat_foldable(
        r_target, g_target, rho0_deg=rho0_ff)
    ne_result = designer.inverse_non_euclidean(
        r_target, g_target, rho_star_target=rho_star_ne,
        geom_type="hyperbolic")

    # 결과 비교표
    print("\n" + "="*65)
    print(f"  {'항목':<22} {'Flat-foldable':>18} {'Non-Euclidean':>18}")
    print("-"*65)
    rows = [
        ("달성 r",   ff_result.get("r_achieved"), ne_result.get("r_achieved")),
        ("달성 g",   ff_result.get("g_achieved"), ne_result.get("g_achieved")),
        ("A 계수",   ff_result.get("A"),           ne_result.get("A")),
        ("B 계수",   ff_result.get("B"),           ne_result.get("B")),
        ("잔차",     ff_result.get("residual"),    ne_result.get("residual")),
    ]
    for name, v1, v2 in rows:
        f1 = f"{v1:.5f}" if isinstance(v1, float) else str(v1)
        f2 = f"{v2:.5f}" if isinstance(v2, float) else str(v2)
        print(f"  {name:<22} {f1:>18} {f2:>18}")

    # 민감도 분석
    if ff_result:
        print("\n  [FF 민감도 분석]")
        sensitivity_analysis(ff_result, delta_pct=5.0, num_periods=10)

    if ne_result:
        print("\n  [NE 민감도 분석]")
        sensitivity_analysis(ne_result, delta_pct=5.0, num_periods=10)

    # 시각화
    print("\n  대시보드 생성 중...")
    fig1 = plot_dashboard_v2(ff_result, ne_result, designer,
                              save_path="helix_inverse_design_v2.png")

    # 수렴 이력
    if designer.convergence_history.get("ff"):
        fig2 = plot_convergence_history(designer, mode='ff',
                                         save_path="convergence_ff.png")
    if designer.convergence_history.get("ne"):
        fig3 = plot_convergence_history(designer, mode='ne',
                                         save_path="convergence_ne.png")

    print("\n  생성 파일:")
    print("    helix_inverse_design_v2.png  (종합 대시보드)")
    print("    convergence_ff.png           (FF 수렴 추적)")
    print("    convergence_ne.png           (NE 수렴 추적)")

    return designer, ff_result, ne_result


# =============================================================================
#  Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="헬리컬 오리가미 역설계 관찰 도구 v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
모드:
  (기본)              전체 역설계 + v2 대시보드
  --forward           순설계 분석 + A/B 위상도
  --scan-ff           FF 설계공간 heatmap 스캔
  --scan-ne           NE 설계공간 heatmap 스캔
  --interactive       FF 대화형 슬라이더 탐색기

예시:
  python helix_inverse_design_v2.py
  python helix_inverse_design_v2.py --forward
  python helix_inverse_design_v2.py --scan-ff --rho0 -90
  python helix_inverse_design_v2.py --interactive --alphas 70 20 110 160
  python helix_inverse_design_v2.py --target-r 2.5 --target-g 0.3
        """
    )
    parser.add_argument('--forward',     action='store_true', help='순설계 분석 + 위상도')
    parser.add_argument('--scan-ff',     action='store_true', help='FF 설계공간 스캔')
    parser.add_argument('--scan-ne',     action='store_true', help='NE 설계공간 스캔')
    parser.add_argument('--interactive', action='store_true', help='대화형 슬라이더')
    parser.add_argument('--target-r',    type=float, default=2.5,   help='목표 r*')
    parser.add_argument('--target-g',    type=float, default=0.30,  help='목표 g*')
    parser.add_argument('--rho0',        type=float, default=-95.0, help='FF ρ₀ [deg]')
    parser.add_argument('--rho-star',    type=float, default=70.0,  help='NE ρ* [deg]')
    parser.add_argument('--alpha1',      type=float, default=100.0, help='NE 스캔 α₁')
    parser.add_argument('--alphas',      type=float, nargs=4,
                        default=[70, 20, 110, 160],
                        help='대화형 탐색기용 sector angles 4개')
    parser.add_argument('--grid',        type=int,   default=18,    help='스캔 격자 수')
    parser.add_argument('--no-show',     action='store_true',       help='화면 출력 생략')

    args = parser.parse_args()

    if args.forward:
        demo_forward_analysis()

    elif args.scan_ff:
        scan = scan_ff_design_space(rho0_deg=args.rho0, n_grid=args.grid)
        plot_ff_scan_dashboard(scan, args.rho0, save_path="ff_design_space.png")

    elif args.scan_ne:
        scan = scan_ne_design_space(alpha1_deg=args.alpha1, n_grid=args.grid)
        plot_ne_scan_dashboard(scan, save_path="ne_design_space.png")

    elif args.interactive:
        launch_interactive_explorer(args.alphas, rho0_init=args.rho0)

    else:
        designer, ff_res, ne_res = run_full_demo(
            r_target=args.target_r,
            g_target=args.target_g,
            rho0_ff=args.rho0,
            rho_star_ne=args.rho_star
        )

    if not args.no_show:
        plt.show()