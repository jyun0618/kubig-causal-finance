"""RQ2: 직전 8분기 예상 상회 비율(HBR)에 따라 실적 발표 반응(CAR[0,+1])이 다른가.

공통 파이프라인이 저장한 output/{industry}/events.csv를 읽는 후속 분석이다.
SUE·CAR·HBR·표본 정의는 다시 계산하지 않는다. 공통 열에 없는 변수만 여기서 만든다.
  - 일치(Meet): 실제 EPS = 컨센서스. 메인에서는 제외한다 (공통 코드는 하회로 센다)
  - Beat Streak: 직전 발표부터 거꾸로 센 연속 상회 횟수 (최대 8). HBR과 같은 과거 발표 집합을 쓴다
  - pre_car: CAR[-5,-1], log_dvol: t−60~t−21 평균 거래대금의 로그 (시가총액 대신 쓰는 규모 대리변수)
    → data/raw/us/prices.csv 캐시가 필요하다. 없으면 price_controls=False로 두 변수 없이 추정한다

모형 (HBR 효과를 상회·하회별로 바로 읽도록 Beat·Miss와 각각 상호작용):
  CAR ~ Beat + S_beat + S_miss + HBR×Beat + HBR×Miss + S_beat×HBR + S_miss×HBR + 통제 | industry^quarter
  S_beat = 서프라이즈 × Beat, S_miss = 서프라이즈 × Miss (원값 SUE이면 SUE⁺, SUE⁻와 같다)
  HBR×Beat → H2-A1, S_beat×HBR → H2-A2, HBR×Miss → H2-B, S_miss×HBR → H2-B 보조
"""

import contextlib
import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

from src.did_us.events import build_market
from src.did_us.rq3 import load_events
from src.did_us.settings import INDUSTRIES, OUT_DIR, RAW_DIR, Params

EPS_TOL = 1e-9
FE_MAIN = "industry^quarter"
CONTROLS = ("hist_sd", "pre_car", "log_dvol")

# 결과표에 뽑을 가설별 항: (가설, 조절변수 이름 → 항 이름)
HYPOTHESES = [
    ("H2-A1 상회 시 수준효과", "{m}_beat"),
    ("H2-A2 상회 시 기울기효과", "Sb_{m}"),
    ("H2-B 하회 시 수준효과", "{m}_miss"),
    ("H2-B 보조: 하회 시 기울기효과", "Sm_{m}"),
]


# ---------------------------------------------------------------- 변수 만들기
def add_streak(ev, hist_q=8):
    """각 발표 직전의 연속 상회 횟수. 공통 _history와 같은 규칙(같은 회사, 반응일 순, beat 결측 제외, 최근 8개)."""
    ev = ev.copy()
    ev["beat_streak"] = np.nan
    ev["beat_cnt_check"] = np.nan
    for _, g in ev.groupby("ticker"):
        g = g.sort_values("t0i")
        beat = g.beat.values
        for j, i in enumerate(g.index):
            prev = beat[:j]
            pb = prev[~np.isnan(prev)][-hist_q:]
            run = 0
            for b in pb[::-1]:
                if b != 1:
                    break
                run += 1
            ev.loc[i, "beat_streak"] = run
            ev.loc[i, "beat_cnt_check"] = pb.sum() if len(pb) else np.nan
    return ev


def add_price_controls(ev, prices_path=None):
    """pre_car = CAR[-5,-1], log_dvol = log(t−60~t−21 평균 거래대금). 공통 파이프라인과 같은 Market 행렬을 쓴다."""
    path = Path(prices_path or RAW_DIR / "prices.csv")
    if not path.exists():
        raise FileNotFoundError(f"{path} 없음. 공통 파이프라인(prepare)을 한 번 실행해 주가 캐시를 만들거나 price_controls=False로 실행")
    M = build_market(pd.read_csv(path, parse_dates=["date"]), Params())
    AR = M.ar()
    dvol = M.close * np.exp(M.logv)
    T = len(M.cal)
    ev = ev.copy()
    pre, ldv, car_chk = [], [], []
    for r in ev.itertuples():
        c, t = M.col.get(r.ticker), int(r.t0i)
        if c is None or t - 60 < 0 or t + 1 >= T:
            pre.append(np.nan), ldv.append(np.nan), car_chk.append(np.nan)
            continue
        pre.append(AR[t - 5:t, c].sum())  # 결측이 있으면 NaN으로 남아 회귀에서 빠진다
        ldv.append(np.log(np.nanmean(dvol[t - 60:t - 20, c])))  # t−60 ~ t−21
        car_chk.append(AR[t, c] + AR[t + 1, c])
    ev["pre_car"], ev["log_dvol"], ev["car01_check"] = pre, ldv, car_chk
    return ev


def build_sample(ev, score="sue", meet="exclude", require_full=True):
    """메인 표본: 공통 in_main_sample & 직전 상회 기록 8개 & (기본) 일치 제외."""
    d = ev[ev.in_main_sample.astype(str).str.lower().eq("true")].copy()
    if require_full:
        d = d[d.n_beat_hist >= 8]
    diff = d.eps_actual - d.eps_consensus
    d["cat"] = np.select([diff > EPS_TOL, diff < -EPS_TOL], ["beat", "miss"], "meet")
    if meet == "exclude":
        d = d[d.cat != "meet"]
    elif meet != "as_miss":
        raise ValueError("meet must be exclude or as_miss")
    d["Beat"] = (d.cat == "beat").astype(float)
    d["Miss"] = 1.0 - d.Beat  # as_miss이면 일치가 Miss에 들어간다
    d["S"] = pd.to_numeric(d[score], errors="coerce")
    d["CAR"] = d.car01
    d["HBR"] = d.beat_rate_8q
    d["Streak"] = d.beat_streak / 8
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=["S", "CAR", "HBR", "Streak"])
    d["S_beat"], d["S_miss"] = d.S * d.Beat, d.S * d.Miss
    d["hbr_bin"] = pd.cut(d.beat_cnt_8q, [-1, 6, 7, 8], labels=["0-6", "7", "8"]).astype(str)
    d["HBR7"], d["HBR8"] = (d.hbr_bin == "7") * 1.0, (d.hbr_bin == "8") * 1.0
    for m in ("HBR", "Streak"):  # 연속 조절변수는 회귀 표본 평균으로 중심화
        d[m] = d[m] - d[m].mean()
    return d.reset_index(drop=True)


def _moderate(d, m):
    """조절변수 m의 네 상호작용 항을 만든다."""
    d[f"{m}_beat"], d[f"{m}_miss"] = d[m] * d.Beat, d[m] * d.Miss
    d[f"Sb_{m}"], d[f"Sm_{m}"] = d.S_beat * d[m], d.S_miss * d[m]
    return [f"{m}_beat", f"{m}_miss", f"Sb_{m}", f"Sm_{m}"]


# ---------------------------------------------------------------- 추정
def fit(d, moderators=(), fe=FE_MAIN, controls=CONTROLS):
    d = d.copy()
    terms = ["Beat", "S_beat", "S_miss"]
    for m in moderators:
        terms += _moderate(d, m)
    ctrl = [c for c in controls if c in d and d[c].notna().any()]
    d = d.dropna(subset=ctrl)
    fml = "CAR ~ " + " + ".join(terms + ctrl) + f" | {fe}"
    # numpy 2.0 + macOS Accelerate에서 matmul이 가짜 overflow 경고를 낸다. 일반 최소제곱으로 다시 계산해 계수 일치를 확인함
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*encountered in matmul")
        f = pf.feols(fml, data=d, vcov={"CRV1": "ticker"})
    tb = f.tidy().reset_index()
    tb.columns = ["term", "coef", "se", "t", "p", "ci_low", "ci_high"]
    tb["n_events"], tb["n_firms"] = len(d), d.ticker.nunique()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        f.summary()
    return tb, buf.getvalue(), fml


def specs(ev, price_controls=True):
    """(이름, 표본, 조절변수, 고정효과, 설명). 메인 표본을 공유하는 모형끼리 비교 가능하다."""
    base = build_sample(ev)
    ctrl = CONTROLS if price_controls else ("hist_sd",)
    return ctrl, [
        ("baseline", base, (), FE_MAIN, "기준 모형: HBR 미포함 (원값 SUE 기준 RQ1)"),
        ("main", base, ("HBR",), FE_MAIN, "메인: 연속 HBR"),
        ("rob_zpast", build_sample(ev, score="z_past"), ("HBR",), FE_MAIN, "강건성: 서프라이즈를 z_past로"),
        ("rob_bins", base, ("HBR7", "HBR8"), FE_MAIN, "강건성: HBR 구간형 (기준 0~6회)"),
        ("rob_firmfe", base, ("HBR",), FE_MAIN + " + ticker", "강건성: 기업 FE 추가"),
        ("rob_meet_as_miss", build_sample(ev, meet="as_miss"), ("HBR",), FE_MAIN, "강건성: 일치를 하회에 포함"),
        ("rob_streak", base, ("Streak",), FE_MAIN, "H2-C: HBR 대신 StreakNorm"),
        ("rob_hbr_streak", base, ("HBR", "Streak"), FE_MAIN, "H2-C: HBR과 StreakNorm 동시 투입"),
    ]


def key_table(results):
    rows = []
    for (name, desc), g in results.groupby(["model", "desc"], sort=False):
        g = g.set_index("term")
        mods = [m for m in ("HBR", "HBR7", "HBR8", "Streak") if f"{m}_beat" in g.index]
        for m in mods:
            for hyp, pat in HYPOTHESES:
                t = pat.format(m=m)
                r = g.loc[t]
                rows.append(dict(model=name, desc=desc, moderator=m, hypothesis=hyp, term=t, coef=r.coef, se=r.se,
                                 p=r.p, ci_low=r.ci_low, ci_high=r.ci_high, n_events=r.n_events, n_firms=r.n_firms))
        if not mods:  # 기준 모형: 기본 반응
            for t in ("Beat", "S_beat", "S_miss"):
                r = g.loc[t]
                rows.append(dict(model=name, desc=desc, moderator="", hypothesis="기본 반응", term=t, coef=r.coef, se=r.se,
                                 p=r.p, ci_low=r.ci_low, ci_high=r.ci_high, n_events=r.n_events, n_firms=r.n_firms))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 실행
def run_rq2(root=OUT_DIR, industries=INDUSTRIES, price_controls=True):
    root = Path(root)
    ev = add_streak(load_events(root, industries))
    bad = (ev.beat_cnt_check - ev.beat_cnt_8q).abs() > 1e-9
    if bad.any():
        raise ValueError(f"Beat Streak 계산의 과거 발표 집합이 공통 HBR과 다름: {int(bad.sum())}건")
    if price_controls:
        ev = add_price_controls(ev)
        m = ev.in_main_sample.astype(str).str.lower().eq("true")
        gap = (ev.loc[m, "car01_check"] - ev.loc[m, "car01"]).abs().max()
        if not gap < 1e-6:
            raise ValueError(f"주가 캐시로 다시 계산한 CAR[0,+1]이 events.csv와 다름 (최대 차이 {gap})")

    ctrl, sp = specs(ev, price_controls)
    dest = root / "rq2"
    dest.mkdir(parents=True, exist_ok=True)
    tabs, texts = [], []
    for name, d, mods, fe, desc in sp:
        tb, txt, fml = fit(d, mods, fe, ctrl)
        tabs.append(tb.assign(model=name, desc=desc))
        texts.append(f"## {name}: {desc}\n{fml}\n{txt}")
    results = pd.concat(tabs, ignore_index=True)
    key = key_table(results)

    base = sp[0][1]
    full = build_sample(ev, meet="as_miss")  # 일치 포함 표본 (셀 집계용, cat에는 meet가 그대로 남아 있다)
    cells = (full.groupby(["industry", "hbr_bin", "cat"]).size().unstack("cat", fill_value=0)
             .reindex(columns=["beat", "meet", "miss"], fill_value=0).reset_index())
    sc = (base[(base.beat_cnt_8q == 8) & (base.cat == "miss")]
          [["eid", "ticker", "industry", "quarter", "t0", "sue", "z_past", "beat_streak", "car01"]
           + (["pre_car"] if "pre_car" in base else [])]
          .sort_values("car01"))

    base.to_csv(dest / "rq2_sample.csv", index=False)
    results.to_csv(dest / "regression_results.csv", index=False)
    key.to_csv(dest / "key_results.csv", index=False)
    cells.to_csv(dest / "cell_counts.csv", index=False)
    sc.to_csv(dest / "sc_candidates.csv", index=False)
    (dest / "models.txt").write_text("\n\n".join(texts), encoding="utf-8")
    return key, cells, sc
