"""패널 만들기와 이벤트스터디 추정 (통일안 4장).

- did_panel + es_did: 발표 여부 스택 DiD (|AR|, 비정상 log 거래량). FE = 스택×회사 + 스택×k, 회사 클러스터
- car_panel + es_event: 발표 이벤트만 쓰는 CAR 이벤트스터디 (연속처치 · 3그룹 · 과거 기록). FE = 이벤트 + 하위그룹×k + 분기×k
- did_b: DiD 안에서 처치×점수 (교차확인 B)
- match_panel: ±10일 안 '예상' 이벤트와 매칭 (보험 질문 A 방식, 교차확인)
"""

import numpy as np
import pandas as pd
import pyfixest as pf

from src.did_us.settings import MARKET


def _kname(k):
    return f"m{-k}" if k < 0 else f"p{k}"


def _window(M, anchor, win):
    idx = anchor + np.arange(-win, win + 1)
    return idx if idx[0] >= 0 and idx[-1] < len(M.cal) else None


def did_panel(treated, ctrl_lists, M, outcome, win, min_ctrl=1, shift=0, bench=MARKET, z=None):
    """스택 패널. outcome: absar | abvol | car. shift>0이면 반응일을 그만큼 앞으로 옮긴다(플라시보)."""
    AR = M.ar(bench)
    ks = np.arange(-win, win + 1)
    out = []
    for i, (r, ctrls) in enumerate(zip(treated.itertuples(), ctrl_lists)):
        anchor = r.t0i - shift
        idx = _window(M, anchor, win)
        if idx is None or len(ctrls) < min_ctrl:
            continue
        firms = [r.ticker] + list(ctrls)
        cols = [M.col[f] for f in firms]
        if outcome == "absar":
            Y = np.abs(AR[np.ix_(idx, cols)])
        elif outcome == "abvol":
            Y = M.logv[np.ix_(idx, cols)] - M.vbase[anchor, cols]
        else:
            a = AR[np.ix_(idx, cols)]
            c = np.cumsum(a, axis=0)
            Y = c - c[win - 1]
        ok = ~np.isnan(Y).any(axis=0)
        if not ok[0] or ok[1:].sum() < min_ctrl:
            continue
        for j in np.where(ok)[0]:
            d = pd.DataFrame({"stack": r.eid, "firm": firms[j], "k": ks, "D": int(j == 0), "y": Y[:, j]})
            if z is not None:
                d["zD"] = getattr(r, z) if j == 0 else 0.0
            out.append(d)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def car_panel(ev, M, win, shift=0, bench=MARKET):
    """발표 이벤트만의 CAR 패널. CAR은 k=−1에서 0."""
    AR = M.ar(bench)
    ks = np.arange(-win, win + 1)
    out = []
    for r in ev.itertuples():
        idx = _window(M, r.t0i - shift, win)
        if idx is None:
            continue
        a = AR[idx, M.col[r.ticker]]
        if np.isnan(a).any():
            continue
        c = np.cumsum(a)
        out.append(pd.DataFrame({"eid": r.eid, "k": ks, "car": c - c[win - 1]}))
    p = pd.concat(out, ignore_index=True)
    return p.merge(ev.drop(columns=["k"], errors="ignore"), on="eid", how="left")


def _fit(df, yvar, terms, ks, fe, cluster="firm"):
    d = df.copy()
    names = []
    for t in terms:
        for k in ks:
            n = f"{t}_{_kname(k)}"
            d[n] = (d.k == k) * d[t]
            names.append(n)
    fml = f"{yvar} ~ " + " + ".join(names) + f" | {fe}"
    fit = pf.feols(fml, data=d, vcov={"CRV1": cluster})
    tb = fit.tidy().reset_index().rename(columns={"Coefficient": "term"})
    tb.columns = ["term", "coef", "se", "t", "p", "ci_low", "ci_high"]
    tb["var"] = tb.term.str.rsplit("_", n=1).str[0]
    tb["k"] = tb.term.str.rsplit("_", n=1).str[1].map(lambda s: -int(s[1:]) if s[0] == "m" else int(s[1:]))
    return tb.drop(columns="term")


def es_did(panel, win, ref=(-10, -6)):
    ks = [k for k in range(-win, win + 1) if not ref[0] <= k <= ref[1]]
    tb = _fit(panel, "y", ["D"], ks, "stack^firm + stack^k")
    tb["n_events"] = panel["stack"].nunique()
    tb["n_obs_firms"] = panel[["stack", "firm"]].drop_duplicates().shape[0]
    return tb


def es_event(panel, terms, win, fe="eid + subgroup^k + quarter^k", yvar="car"):
    ks = [k for k in range(-win, win + 1) if k != -1]
    d = panel.rename(columns={"ticker": "firm"})
    tb = _fit(d, yvar, terms, ks, fe)
    tb["n_events"] = d.eid.nunique()
    return tb


def did_b(panel_car, win):
    """교차확인 B: 스택 패널에서 CAR ~ D×k + (D×점수)×k."""
    ks = [k for k in range(-win, win + 1) if k != -1]
    tb = _fit(panel_car, "y", ["D", "zD"], ks, "stack^firm + stack^k")
    tb["n_events"] = panel_car["stack"].nunique()
    return tb


def match_panel(ev, M, win, group, h=10, bench=MARKET):
    """보험 질문 A: 커짐(또는 작아짐) 이벤트 vs 처치 반응일 ±h 거래일 안 다른 회사의 '예상' 이벤트. 각자 자기 t0에 정렬."""
    AR = M.ar(bench)
    ks = np.arange(-win, win + 1)
    exp = ev[ev.group == "expected"]
    out = []
    for r in ev[ev.group == group].itertuples():
        c = exp[(exp.ticker != r.ticker) & ((exp.t0i - r.t0i).abs() <= h)]
        rows = [(r, 1)] + [(x, 0) for x in c.itertuples()]
        if len(rows) < 2:
            continue
        for x, d in rows:
            idx = _window(M, x.t0i, win)
            if idx is None:
                continue
            a = AR[idx, M.col[x.ticker]]
            if np.isnan(a).any():
                continue
            cs = np.cumsum(a)
            out.append(pd.DataFrame({"stack": r.eid, "firm": x.ticker, "eid": x.eid, "k": ks, "D": d, "y": cs - cs[win - 1]}))
    return pd.concat(out, ignore_index=True)


def es_match(panel, win):
    ks = [k for k in range(-win, win + 1) if k != -1]
    d = panel.assign(se=panel["stack"] + "|" + panel.eid)
    tb = _fit(d, "y", ["D"], ks, "se + stack^k")
    tb["n_events"] = panel.loc[panel.D == 1, "stack"].nunique()
    return tb


def pick(tb, var, k):
    r = tb[(tb["var"] == var) & (tb.k == k)]
    return r.iloc[0] if len(r) else None
