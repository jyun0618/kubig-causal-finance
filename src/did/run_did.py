"""실적발표 DiD 파일럿: 통제군 매칭 + 이중차분 추정 + 검증.

실행: .venv/bin/python -m src.did.run_did --industry finance   (build_event_windows 이후)
업종(sector)별로 설계 A·B를 돌린다. 사건이 있는 기업이 MIN_FIRMS곳 미만인 업종은 건너뛴다.
출력 (data/<산업군>/results/):
  did_summary.csv       설계·업종·결과변수별 추정치 요약
  did_event_study.csv   상대일별 이벤트스터디 계수
  did_events.csv        사건 단위 DiD 값 (재분석용)
  did_balance.csv       매칭 전후 공변량 균형

설계 A (발표 시점): 처치 = 어떤 분기에 발표한 기업, 통제 = 같은 업종·같은 분기에서 처치 t0보다
  GAP_DAYS 거래일 이상 뒤에 발표하는 기업(not-yet-treated). 규모·변동성·SUE가 가까운 K_CONTROLS곳을 고른다
  (SUE가 비슷한 기업끼리 비교해 '실적 내용'이 아니라 '발표 여부'의 효과만 남기려는 의도).
  결과변수는 같은 달력일의 수익률이라 시장·업종 공통 충격이 차분으로 상쇄된다.
설계 B (서프라이즈 강도): 같은 업종·분기에서 SUE 상위(또는 하위) 1/3 = 처치, 중간 1/3 = 통제.
  t0 차이가 B_WINDOW 거래일 이내인 통제군 중 규모·변동성이 가까운 K_CONTROLS곳을 고른다.
  각자 발표일 기준의 시장모형 초과수익률(AR)로 비교한다 (발표 시점이 달라 달력일 상쇄가 안 되므로).

DiD 추정: 사건 단위로 [(처치 post−pre) − (통제 post−pre 평균)]을 구해 평균낸다.
  pre = 상대일 -5..-1, post = 처치의 반응일(0..reaction_days-1)
추론: 기업·분기 단위 군집 부호반전 wild bootstrap p값 (군집 수가 적어 정확 열거 또는 몬테카를로).
"""

import itertools

import numpy as np
import pandas as pd

from src.common import ROOT, cli_industry

MIN_FIRMS = 4

PRE, POST = 5, 3
EST_START, EST_END = -65, -6
K_CONTROLS = 2
GAP_DAYS = 4
B_WINDOW = 3
MIN_GROUP = 6
PLACEBO_SHIFT = -10
BOOT_DRAWS = 20000
OUTCOMES = ("ret", "abs_ret", "abn_vol")
RELS = np.arange(-PRE, POST + 1)


class Data:
    def __init__(self, industry):
        prices = pd.read_csv(industry.path("prices"), parse_dates=["date"], dtype={"ticker": str})
        self.days = pd.DatetimeIndex(sorted(prices["date"].unique()))
        wide = lambda col: prices.pivot(index="date", columns="ticker", values=col).reindex(self.days)
        self.ret = {t: v.to_numpy() for t, v in wide("ret").items()}
        self.logvol = {t: np.log(v.to_numpy()) for t, v in wide("volume").items()}
        self.mcap = {t: v.to_numpy() for t, v in wide("market_cap").items()}
        index = pd.read_csv(industry.path("index"), parse_dates=["date"]).set_index("date")
        market_ret = {m: index[f"{m.lower()}_ret"].reindex(self.days).to_numpy() for m in industry.universe["market"].unique()}
        self.mkt = {t: market_ret[m] for t, m in zip(industry.universe["ticker"], industry.universe["market"])}

        ev = pd.read_csv(industry.path("events"), parse_dates=["t0_date"], dtype={"ticker": str})
        ev["i"] = self.days.searchsorted(ev["t0_date"])
        self.ev = ev


def window(i):
    return slice(i + EST_START, i + EST_END + 1)


def features(D, ticker, i):
    """i 시점의 규모(직전일 log 시총)와 변동성(추정 구간 일수익률 표준편차)."""
    cap = D.mcap[ticker][i - 1]
    return {"log_mcap": np.log(cap) if cap > 0 else np.nan, "vol60": np.nanstd(D.ret[ticker][window(i)], ddof=1)}


def path(D, ticker, i, est_i, shift=0, alpha=None, beta=None):
    """상대일 -5..+3의 결과변수 경로. alpha가 주어지면 시장모형 초과수익률을 쓴다."""
    idx = i + shift + RELS
    r = D.ret[ticker][idx]
    if alpha is not None:
        r = r - alpha - beta * D.mkt[ticker][idx]
    base = np.nanmean(D.logvol[ticker][window(est_i)])
    return {"ret": r, "abs_ret": np.abs(r), "abn_vol": D.logvol[ticker][idx] - base}


def valid(p):
    return all(np.isfinite(v).all() for v in p.values())


def did(treated, controls, reaction_days):
    """사건 하나의 DiD와 이벤트스터디(상대일 -1 기준) 값을 계산한다."""
    pre, post = RELS < 0, (RELS >= 0) & (RELS < reaction_days)
    base = RELS == -1
    out = {}
    for o in OUTCOMES:
        diff = lambda p: p[o][post].mean() - p[o][pre].mean()
        out[f"treat_{o}"] = diff(treated)
        out[f"ctrl_{o}"] = np.mean([diff(c) for c in controls])
        out[f"did_{o}"] = out[f"treat_{o}"] - out[f"ctrl_{o}"]
        early, late = (RELS >= -5) & (RELS <= -3), (RELS >= -2) & (RELS <= -1)
        slope = lambda p: p[o][late].mean() - p[o][early].mean()
        out[f"pretrend_{o}"] = slope(treated) - np.mean([slope(c) for c in controls])
    for k, rel in enumerate(RELS):
        out[f"es_{rel}"] = ((treated["ret"][k] - treated["ret"][base][0])
                            - np.mean([c["ret"][k] - c["ret"][base][0] for c in controls]))
    return out


def standardized_distance(target, cands, cols, sd):
    return np.sqrt(sum(((cands[c] - target[c]) / sd[c]) ** 2 for c in cols))


def design_a(D, sector):
    ev = D.ev[(D.ev["sector"] == sector) & D.ev["sue"].notna()]
    cols, rows, bal = ["log_mcap", "vol60", "sue"], [], []
    feat_all = pd.DataFrame([{**features(D, e.ticker, e.i), "sue": e.sue} for e in ev.itertuples()])
    sd = feat_all.std()
    for e in ev.itertuples():
        pool = ev[(ev["fiscal_q"] == e.fiscal_q) & (ev["ticker"] != e.ticker) & (ev["i"] >= e.i + GAP_DAYS)]
        if pool.empty:
            continue
        t_feat = {**features(D, e.ticker, e.i), "sue": e.sue}
        cand = pd.DataFrame([{**features(D, c.ticker, e.i), "sue": c.sue, "ticker": c.ticker} for c in pool.itertuples()])
        cand["dist"] = standardized_distance(t_feat, cand, cols, sd)
        cand["path_ok"] = [valid(path(D, c, e.i, e.i)) for c in cand["ticker"]]
        cand = cand[cand["path_ok"] & cand[cols].notna().all(axis=1)].sort_values("dist")
        pick = cand.head(K_CONTROLS)
        t_path = path(D, e.ticker, e.i, e.i)
        if pick.empty or not valid(t_path):
            continue
        res = did(t_path, [path(D, c, e.i, e.i) for c in pick["ticker"]], e.reaction_days)
        plc = did(path(D, e.ticker, e.i, e.i, PLACEBO_SHIFT),
                  [path(D, c, e.i, e.i, PLACEBO_SHIFT) for c in pick["ticker"]], e.reaction_days)
        rows.append({"design": "A", "sector": sector, "ticker": e.ticker, "fiscal_q": e.fiscal_q,
                     "timing": e.timing, "sue": e.sue, "n_controls": len(pick),
                     **res, **{f"placebo_{k}": v for k, v in plc.items() if k.startswith("did_")}})
        for tag, group in (("matched", pick), ("all_candidates", cand)):
            bal.append({"design": "A", "sector": sector, "sample": tag,
                        **{c: t_feat[c] - group[c].mean() for c in cols},
                        **{f"{c}_t": t_feat[c] for c in cols}})
    return pd.DataFrame(rows), pd.DataFrame(bal)


def design_b(D, sector, contrast):
    ev = D.ev[(D.ev["sector"] == sector) & D.ev["sue"].notna()]
    cols, rows, bal = ["log_mcap", "vol60"], [], []
    feat = {r.Index: features(D, r.ticker, r.i) for r in ev.itertuples()}
    sd = pd.DataFrame(feat.values()).std()
    for q, g in ev.groupby("fiscal_q"):
        if len(g) < MIN_GROUP:
            continue
        rank, n = g["sue"].rank(method="first"), len(g)
        third = n // 3
        treat = g[rank > n - third] if contrast == "high" else g[rank <= third]
        mid = g[(rank > third) & (rank <= n - third)]
        for e in treat.itertuples():
            pool = mid[(mid["i"] - e.i).abs() <= B_WINDOW]
            if pool.empty:
                continue
            cand = pd.DataFrame([{**feat[c.Index], "row": c} for c in pool.itertuples()])
            cand["dist"] = standardized_distance(feat[e.Index], cand, cols, sd)
            cand = cand[cand[cols].notna().all(axis=1)].sort_values("dist")

            def ar_path(r, shift=0):
                return path(D, r.ticker, r.i, r.i, shift, r.alpha, r.beta)

            pick = cand.head(K_CONTROLS)["row"].tolist()
            if not pick or not valid(ar_path(e)) or not all(valid(ar_path(c)) for c in pick):
                continue
            res = did(ar_path(e), [ar_path(c) for c in pick], e.reaction_days)
            plc = did(ar_path(e, PLACEBO_SHIFT), [ar_path(c, PLACEBO_SHIFT) for c in pick], e.reaction_days)
            rows.append({"design": f"B_{contrast}", "sector": sector, "ticker": e.ticker, "fiscal_q": e.fiscal_q,
                         "timing": e.timing, "sue": e.sue, "n_controls": len(pick),
                         **res, **{f"placebo_{k}": v for k, v in plc.items() if k.startswith("did_")}})
            for tag, group in (("matched", cand.head(K_CONTROLS)), ("all_candidates", cand)):
                bal.append({"design": f"B_{contrast}", "sector": sector, "sample": tag,
                            **{c: feat[e.Index][c] - group[c].mean() for c in cols},
                            **{f"{c}_t": feat[e.Index][c] for c in cols}})
    return pd.DataFrame(rows), pd.DataFrame(bal)


def cluster_se(values, clusters):
    resid = pd.Series(values - values.mean())
    return float(np.sqrt((resid.groupby(np.asarray(clusters)).sum() ** 2).sum()) / len(values))


def wild_p(values, clusters, seed=0):
    """군집 단위 부호반전 wild bootstrap p값. 군집이 14개 이하면 모든 조합을 열거한다."""
    codes, uniq = pd.factorize(pd.Series(clusters))
    g = len(uniq)
    if g <= 14:
        signs = np.array(list(itertools.product([-1, 1], repeat=g)))
    else:
        signs = np.random.default_rng(seed).choice([-1, 1], size=(BOOT_DRAWS, g))
    stat = (signs[:, codes] * values).mean(axis=1)
    return float((np.abs(stat) >= abs(values.mean()) - 1e-12).mean())


def summarize(df, label):
    rows = []
    for o in OUTCOMES:
        v, plc, pt = (df[f"did_{o}"].to_numpy(), df[f"placebo_did_{o}"].to_numpy(), df[f"pretrend_{o}"].to_numpy())
        rows.append({
            "design": label[0], "sector": label[1], "outcome": o, "n_events": len(df),
            "n_firms": df["ticker"].nunique(), "n_quarters": df["fiscal_q"].nunique(),
            "treat_change_pct": df[f"treat_{o}"].mean() * 100, "ctrl_change_pct": df[f"ctrl_{o}"].mean() * 100,
            "did_pct": v.mean() * 100, "se_firm": cluster_se(v, df["ticker"]) * 100,
            "p_firm": wild_p(v, df["ticker"]), "p_quarter": wild_p(v, df["fiscal_q"]),
            "placebo_pct": plc.mean() * 100, "placebo_p_quarter": wild_p(plc, df["fiscal_q"]),
            "pretrend_pct": pt.mean() * 100, "pretrend_p_quarter": wild_p(pt, df["fiscal_q"]),
        })
    return rows


def event_study(df, label):
    rows = []
    for rel in RELS:
        v = df[f"es_{rel}"].to_numpy()
        rows.append({"design": label[0], "sector": label[1], "rel_day": int(rel), "coef_pct": v.mean() * 100,
                     "se_firm_pct": cluster_se(v, df["ticker"]) * 100})
    return rows


def balance_table(bal, label):
    rows = []
    for tag, g in bal.groupby("sample"):
        for c in [c for c in bal.columns if c in ("log_mcap", "vol60", "sue")]:
            sd_t = g[f"{c}_t"].std()
            rows.append({"design": label[0], "sector": label[1], "sample": tag, "variable": c,
                         "smd": g[c].mean() / sd_t if sd_t > 0 else np.nan, "n": len(g)})
    return rows


def main(industry):
    D = Data(industry)
    out_dir = industry.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    n_firms = D.ev.groupby("sector")["ticker"].nunique()
    sectors = sorted(n_firms[n_firms >= MIN_FIRMS].index)
    skipped = sorted(set(n_firms.index) - set(sectors))
    if skipped:
        print(f"기업 수가 {MIN_FIRMS}곳 미만이라 건너뛴 업종: {', '.join(skipped)}")
    if not sectors:
        raise SystemExit("DiD를 돌릴 수 있는 업종이 없습니다 (업종당 사건이 있는 기업이 4곳 이상 필요).")
    runs = [(("A", s), design_a(D, s)) for s in sectors]
    runs += [((f"B_{c}", s), design_b(D, s, c)) for s in sectors for c in ("high", "low")]

    summary, es, events, balance = [], [], [], []
    for label, (df, bal) in runs:
        if df.empty:
            continue
        summary += summarize(df, label)
        es += event_study(df, label)
        balance += balance_table(bal, label)
        events.append(df)

    if not events:
        raise SystemExit("추정 가능한 사건이 없습니다. 매칭 조건(통제군 후보)이나 SUE 결측을 확인하세요.")
    pd.DataFrame(summary).to_csv(out_dir / "did_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(es).to_csv(out_dir / "did_event_study.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(balance).to_csv(out_dir / "did_balance.csv", index=False, encoding="utf-8-sig")
    pd.concat(events).to_csv(out_dir / "did_events.csv", index=False, encoding="utf-8-sig")
    print(f"저장: {out_dir.relative_to(ROOT)}/ (설계 {len(runs)}개 중 추정 {len(events)}개)")


if __name__ == "__main__":
    main(cli_industry(__doc__.splitlines()[0]))
