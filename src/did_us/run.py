"""산업별 실행: 메인 분석(4장) → 강건성·플라시보(5장) → 결과 저장(6장).

    ctx = prepare()                      # 세 산업 데이터 + 이벤트 테이블
    res = run_industry(ctx, "insurance")  # output/insurance/ 에 저장
    combine([res, ...])                   # output/combined/ 에 저장
"""

import warnings

import numpy as np
import pandas as pd

from src.did_us import estimate as es
from src.did_us.data import load_data
from src.did_us.events import build_events, build_market, find_controls, main_events
from src.did_us.settings import COVID_FIRMS, INDUSTRIES, MARKET, OUT_DIR, ROBUST_EXCLUDE_PERIODS, SECTOR_ETF, Params

warnings.filterwarnings("ignore")


def prepare(P: Params = None, refresh=False):
    P = P or Params()
    uni, earn, px = load_data(INDUSTRIES, P, refresh)
    M = build_market(px, P)
    ev = build_events(earn, M, P)
    return dict(P=P, uni=uni, earn=earn, M=M, ev=ev)


# ---------------------------------------------------------------- 대표 숫자 뽑기
def _row(spec, analysis, metric, tb, var, k):
    r = es.pick(tb, var, k)
    if r is None:
        return None
    return dict(spec=spec, analysis=analysis, metric=metric, k=k, coef=r.coef, se=r.se,
                ci_low=r.ci_low, ci_high=r.ci_high, p=r.p, n_events=int(tb.n_events.iloc[0]))


def _did_rows(ctx, tr, spec, P, pool=None, h=None, min_ctrl=None, bench=MARKET, shift=0, outcomes=("absar", "abvol")):
    M = ctx["M"]
    ctrl = find_controls(tr, ctx["ev"], ctx["uni"], M, P, pool=pool, h=h, bench=bench)
    rows = []
    for o in outcomes:
        pn = es.did_panel(tr, ctrl, M, o, P.win, min_ctrl or P.min_ctrl, shift=shift, bench=bench)
        if pn.empty:
            continue
        tb = es.es_did(pn, P.win, P.ref_bin)
        rows.append(_row(spec, "발표 여부", "|AR| (%p)" if o == "absar" else "비정상 log 거래량", tb, "D", 0))
    return rows


def _cont_rows(ctx, tr, spec, P, score="z_past", bench=MARKET, fe="eid + subgroup^k + quarter^k", shift=0, win=None):
    win = win or P.win
    d = tr[tr[score].notna()].copy()
    cp = es.car_panel(d, ctx["M"], win, shift=shift, bench=bench)
    tb = es.es_event(cp, [score], win, fe=fe)
    return [_row(spec, "연속처치", "점수 1당 CAR[0,+1] (%p)", tb, score, 1)]


def _g3_rows(ctx, tr, spec, P, gcol="group", bench=MARKET, win=None):
    win = win or P.win
    cp = es.car_panel(tr[tr[gcol].notna()], ctx["M"], win, bench=bench)
    cp = cp.assign(up=(cp[gcol] == "up") * 1.0, down=(cp[gcol] == "down") * 1.0)
    tb = es.es_event(cp, ["up", "down"], win)
    return [_row(spec, "3그룹", "커짐 vs 예상 CAR[0,+1] (%p)", tb, "up", 1),
            _row(spec, "3그룹", "작아짐 vs 예상 CAR[0,+1] (%p)", tb, "down", 1)]


def _all_rows(ctx, tr, spec, P, **kw):
    return _did_rows(ctx, tr, spec, P, **{k: v for k, v in kw.items() if k in ("bench",)}) + \
        _cont_rows(ctx, tr, spec, P, **kw) + _g3_rows(ctx, tr, spec, P, **kw)


# ---------------------------------------------------------------- 메인 분석
def main_analyses(ctx, ind, P):
    M, ev = ctx["M"], ctx["ev"]
    tr = main_events(ev, ind, P.win, len(M.cal))
    ctrl = find_controls(tr, ev, ctx["uni"], M, P)
    tr["n_ctrl"] = [len(c) for c in ctrl]
    out = {"treated": tr}
    out["did_absar"] = es.es_did(es.did_panel(tr, ctrl, M, "absar", P.win, P.min_ctrl), P.win, P.ref_bin)
    out["did_abvol"] = es.es_did(es.did_panel(tr, ctrl, M, "abvol", P.win, P.min_ctrl), P.win, P.ref_bin)

    cp = es.car_panel(tr, M, P.win)
    cp["up"], cp["down"] = (cp.group == "up") * 1.0, (cp.group == "down") * 1.0
    out["cont"] = es.es_event(cp, ["z_past"], P.win)
    out["group3"] = es.es_event(cp, ["up", "down"], P.win)

    # 4-4 과거 기록: 연속 상회(S) 여부와 상회율(연속)에 따라 점수 효과가 다른가
    s = cp[cp.streak_group.notna()].copy()
    s["S"] = (s.streak_group == "streak") * 1.0
    s["zS"] = s.z_past * s.S
    out["streak"] = es.es_event(s, ["z_past", "zS", "S"], P.win)
    s["R"] = s.beat_rate_8q - s.beat_rate_8q.mean()
    s["zR"] = s.z_past * s.R
    out["beatrate"] = es.es_event(s, ["z_past", "zR", "R"], P.win)
    cell = tr[tr.streak_group.notna() & tr.beat.notna()]
    out["streak_cells"] = (cell.assign(beat=np.where(cell.beat == 1, "beat", "miss"))
                           .groupby(["streak_group", "beat"]).car01.agg(["count", "mean", "std"]).reset_index())
    fm = cp[cp.first_miss]
    out["first_miss"] = fm.groupby("k").car.agg(["count", "mean"]).reset_index()
    return out


# ---------------------------------------------------------------- 강건성 (통일안 5장)
def robustness(ctx, ind, P, tr):
    ev = ctx["ev"]
    etf = SECTOR_ETF[ind]
    R = []
    add = lambda rows: R.extend([r for r in rows if r is not None])

    # 5-1 정의·모형
    for sc, lab in [("z_pct_past", "컨센서스 기준 서프라이즈(%)"), ("z_full", "전체기간 표준화(기존 보험)"),
                    ("z_mkt", "시장 기준 점수"), ("z_wins", "1·99% 윈저라이즈")]:
        add(_cont_rows(ctx, tr, lab, P, score=sc))
    add(_g3_rows(ctx, tr, "3그룹 기준 ±1SD", P, gcol="group_1sd"))
    add(_did_rows(ctx, tr, f"업종 ETF({etf}) 기준", P, bench=etf, outcomes=("absar",)))
    add(_cont_rows(ctx, tr, f"업종 ETF({etf}) 기준", P, bench=etf))
    add(_g3_rows(ctx, tr, f"업종 ETF({etf}) 기준", P, bench=etf))
    add(_cont_rows(ctx, tr, "FE: 이벤트 + 분기×k만", P, fe="eid + quarter^k"))
    add(_cont_rows(ctx, tr, "FE: 이벤트 + 하위그룹×k만", P, fe="eid + subgroup^k"))
    add(_cont_rows(ctx, tr, "FE: 이벤트 + k만", P, fe="eid + k"))
    # 교차확인 B: DiD 안에서 처치×점수
    ctrl = find_controls(tr, ev, ctx["uni"], ctx["M"], P)
    pb = es.did_panel(tr, ctrl, ctx["M"], "car", P.win, P.min_ctrl, z="z_past")
    add([_row("교차확인 B: DiD 안 처치×점수", "연속처치", "점수 1당 CAR[0,+1] (%p)", es.did_b(pb, P.win), "zD", 1)])
    # 교차확인: ±10일 안 '예상' 이벤트와 매칭 (보험 질문 A)
    for g, m in [("up", "커짐 vs 예상 CAR[0,+1] (%p)"), ("down", "작아짐 vs 예상 CAR[0,+1] (%p)")]:
        add([_row("교차확인: ±10일 '예상' 이벤트 매칭", "3그룹", m, es.es_match(es.match_panel(tr, ctx["M"], P.win, g), P.win), "D", 1)])
    add(_did_rows(ctx, tr, "대조군: 다른 하위그룹만", P, pool="other_subgroup"))
    add(_did_rows(ctx, tr, "대조군 최소 2개", P, min_ctrl=2))

    # 데이터 점검 플래그 제외
    for f, lab in [("flag_eps", "EPS 의심 제외"), ("flag_date", "날짜 의심 제외"), ("flag_mkt_shock", "시장 급변일 낀 이벤트 제외"),
                   ("flag_mixed_timing", "시각 혼재 기업 제외"), ("flag_confound", "|CAR[0,+1]|>10%p 제외"), ("robust_exclude", "경계 교란 이벤트 제외")]:
        if tr[f].any():
            add(_all_rows(ctx, tr[~tr[f]], lab, P))
    for a, b, lab in ROBUST_EXCLUDE_PERIODS.get(ind, []):
        add(_all_rows(ctx, tr[~tr.announce_date.between(a, b)], f"{lab} 제외", P))
    add(_cont_rows(ctx, tr, "분석 창 ±5", P, win=5))
    add(_g3_rows(ctx, tr, "분석 창 ±5", P, win=5))
    for tm in ("BMO", "AMC"):
        d = tr[tr.timing.eq(tm) if tm == "AMC" else tr.timing.ne("AMC")]
        if len(d) >= 30:
            add(_cont_rows(ctx, d, f"{tm} 발표만", P))
    for sg, d in tr.groupby("subgroup"):
        if len(d) >= 30:
            add(_cont_rows(ctx, d, f"하위그룹: {sg}", P, fe="eid + quarter^k"))

    # 5-2 대조군 부족
    add(_did_rows(ctx, tr, "대조군 규칙 ±7 (k=0만 해석)", P, h=7))
    add(_did_rows(ctx, tr, "대조군 규칙 ±5 (k=0만 해석)", P, h=5))
    add(_did_rows(ctx, tr, "대조군 풀: 세 산업 합침", P, pool="all"))
    add(_cont_rows(ctx, tr[tr.n_ctrl > 0], "대조군 있는 이벤트만", P))
    add(_g3_rows(ctx, tr[tr.n_ctrl > 0], "대조군 있는 이벤트만", P))

    # 5-3 코로나 시기 비교 기준
    for y in ("2023-01-01", "2024-01-01"):
        add(_all_rows(ctx, tr[tr.announce_date >= y], f"{y[:4]}년 이후 이벤트만", P))
    add(_cont_rows(ctx, tr, "코로나 분기(2020) 제외 점수", P, score="z_nocovid"))
    t2 = tr.copy()
    t2["group"] = pd.cut(t2.z_nocovid, [-np.inf, -P.z_cut, P.z_cut, np.inf], labels=["down", "expected", "up"]).astype(object)
    add(_g3_rows(ctx, t2, "코로나 분기(2020) 제외 점수", P))
    cp = es.car_panel(tr, ctx["M"], P.win)
    cp["C"] = (cp.n_hist_covid >= 4) * 1.0
    cp["zC"] = cp.z_past * cp.C
    add([_row("코로나 상호작용: 점수×(직전 기록 절반 이상 코로나)", "연속처치", "추가 기울기 CAR[0,+1] (%p)",
              es.es_event(cp, ["z_past", "zC", "C"], P.win), "zC", 1)])
    if ind in COVID_FIRMS:
        add(_all_rows(ctx, tr[~tr.ticker.isin(COVID_FIRMS[ind])], "코로나 제품 기업 제외", P))
    return pd.DataFrame(R)


def placebo(ctx, ind, P, tr):
    R = []
    for s in sorted(set((P.placebo_shift,) + tuple(P.placebo_grid))):
        rows = _did_rows(ctx, tr, f"−{s}거래일", P, shift=s) + _cont_rows(ctx, tr, f"−{s}거래일", P, shift=s)
        R.extend([dict(r, shift=s) for r in rows if r is not None])
    return pd.DataFrame(R)


def control_balance(tr):
    """대조군이 있는 이벤트 vs 없는 이벤트의 특성 비교 (5-2 ②)."""
    t = tr.assign(has_ctrl=np.where(tr.n_ctrl > 0, "대조군 있음", "대조군 없음"), amc=(tr.timing == "AMC") * 1.0)
    # 시즌 내 발표 순서: 같은 분기 이벤트 중 몇 번째 (0 = 첫 발표, 1 = 마지막)
    t["season_rank"] = t.groupby("quarter").t0i.rank(pct=True)
    g = t.groupby("has_ctrl").agg(n=("eid", "size"), z_mean=("z_past", "mean"), z_absmean=("z_past", lambda s: s.abs().mean()),
                                  amc_share=("amc", "mean"), season_rank=("season_rank", "mean"), car01_abs=("car01", lambda s: s.abs().mean()))
    sub = pd.crosstab(t.has_ctrl, t.subgroup, normalize="index").add_prefix("share_")
    return g.join(sub).reset_index()


def covid_share(ev, ind):
    m = ev[(ev.industry == ind) & (ev.period == "main") & ev.z_past.notna()]
    return (m.assign(year=m.announce_date.dt.year, covid_share=m.n_hist_covid / m.n_hist)
            .groupby("year").agg(n=("eid", "size"), covid_share=("covid_share", "mean"), hist_sd_median=("hist_sd", "median"))
            .reset_index())


# ---------------------------------------------------------------- 요약 · 저장
def team_summary(ind, out):
    rows = []
    for key, analysis, metric, var, k in [
        ("did_absar", "발표 여부", "반응일 |AR| 차이 (%p)", "D", 0),
        ("did_abvol", "발표 여부", "반응일 비정상 log 거래량 차이", "D", 0),
        ("cont", "연속처치", "점수 1당 CAR[0,+1] (%p)", "z_past", 1),
        ("group3", "3그룹", "커짐 vs 예상 CAR[0,+1] (%p)", "up", 1),
        ("group3", "3그룹", "작아짐 vs 예상 CAR[0,+1] (%p)", "down", 1),
        ("streak", "과거 기록", "연속 상회 회사의 점수 기울기 차이 CAR[0,+1] (%p)", "zS", 1),
        ("beatrate", "과거 기록", "상회율 +1(0→100%)당 점수 기울기 변화 CAR[0,+1] (%p)", "zR", 1),
    ]:
        r = _row("메인", analysis, metric, out[key], var, k)
        if r:
            rows.append(r)
    df = pd.DataFrame(rows).drop(columns="spec")
    v = df[df.metric.str.contains("거래량")].iloc[0]
    df = pd.concat([df, pd.DataFrame([dict(analysis="발표 여부", metric="반응일 거래량 (배)", k=0, coef=np.exp(v.coef),
                                           ci_low=np.exp(v.ci_low), ci_high=np.exp(v.ci_high), n_events=v.n_events)])])
    tr = out["treated"]
    df = pd.concat([df, pd.DataFrame([
        dict(analysis="표본", metric="분석 이벤트 수", coef=len(tr)),
        dict(analysis="표본", metric="대조군 확보율 (%)", coef=(tr.n_ctrl > 0).mean() * 100),
        dict(analysis="표본", metric="이벤트당 평균 대조군 수", coef=tr.n_ctrl.mean()),
    ])])
    df.insert(0, "industry", ind)
    return df.reset_index(drop=True)


def _plot(tb_list, labels, title, path, ylabel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    for f in ("AppleGothic", "Malgun Gothic", "NanumGothic", "Noto Sans CJK KR"):
        if f in names:
            plt.rcParams["font.family"] = f
            break
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, ((tb, var), lab) in enumerate(zip(tb_list, labels)):
        d = tb[tb["var"] == var].sort_values("k")
        x = d.k + (i - (len(tb_list) - 1) / 2) * 0.15
        ax.errorbar(x, d.coef, yerr=[d.coef - d.ci_low, d.ci_high - d.coef], fmt="o-", ms=3, capsize=2, label=lab)
    ax.axhline(0, color="gray", lw=0.8)
    ax.axvline(-0.5, color="gray", lw=0.8, ls="--")
    ks = sorted(tb_list[0][0].k.unique().tolist() + [-1])
    ax.set_xticks(range(ks[0], ks[-1] + 1, 2))
    ax.set_xlabel("반응일 기준 거래일 (k)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if len(labels) > 1:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save(ind, out, ev):
    d = OUT_DIR / ind
    (d / "figures").mkdir(parents=True, exist_ok=True)
    e = ev[ev.industry == ind].merge(out["treated"][["eid", "n_ctrl"]], on="eid", how="left")
    e["in_main_sample"] = e.eid.isin(out["treated"].eid)
    e.to_csv(d / "events.csv", index=False)
    pd.concat([out["did_absar"].assign(outcome="absar"), out["did_abvol"].assign(outcome="abvol")]).to_csv(d / "did_results.csv", index=False)
    pd.concat([out[k].assign(model=k) for k in ("cont", "group3", "streak", "beatrate")]).to_csv(d / "sue_results.csv", index=False)
    for k in ("robustness", "placebo", "control_balance", "covid_share", "streak_cells", "first_miss", "team_summary"):
        if k in out:
            out[k].to_csv(d / f"{k}.csv", index=False)
    f = d / "figures"
    _plot([(out["did_absar"], "D")], ["|AR|"], f"[{ind}] 발표 여부: |AR| 차이", f / "did_absar.png", "%p")
    _plot([(out["did_abvol"], "D")], ["거래량"], f"[{ind}] 발표 여부: 비정상 log 거래량 차이", f / "did_abvol.png", "log")
    _plot([(out["cont"], "z_past")], ["점수 1당"], f"[{ind}] 연속처치: 점수 1당 CAR", f / "cont_car.png", "%p")
    _plot([(out["group3"], "up"), (out["group3"], "down")], ["커짐 vs 예상", "작아짐 vs 예상"], f"[{ind}] 3그룹 CAR", f / "group3_car.png", "%p")
    _plot([(out["streak"], "z_past"), (out["streak"], "zS")], ["점수 1당 (연속 상회 아님)", "연속 상회 추가 기울기"],
          f"[{ind}] 과거 기록별 점수 효과", f / "streak_car.png", "%p")


def run_industry(ctx, ind, robust=True, run_placebo=True, write=True):
    P = ctx["P"]
    out = main_analyses(ctx, ind, P)
    tr = out["treated"]
    if robust:
        out["robustness"] = robustness(ctx, ind, P, tr)
        out["control_balance"] = control_balance(tr)
        out["covid_share"] = covid_share(ctx["ev"], ind)
    if run_placebo:
        out["placebo"] = placebo(ctx, ind, P, tr)
    out["team_summary"] = team_summary(ind, out)
    if write:
        save(ind, out, ctx["ev"])
    return out


def combine(results: dict):
    d = OUT_DIR / "combined"
    (d / "figures").mkdir(parents=True, exist_ok=True)
    ts = pd.concat([r["team_summary"] for r in results.values()], ignore_index=True)
    ts.to_csv(d / "sector_comparison.csv", index=False)
    rb = [r["robustness"].assign(industry=i) for i, r in results.items() if "robustness" in r]
    if rb:
        pd.concat(rb, ignore_index=True).to_csv(d / "robustness_all.csv", index=False)
    return ts
