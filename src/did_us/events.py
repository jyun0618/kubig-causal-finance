"""이벤트 테이블: 발표 1건 = 1행. 이후 모든 분석이 이 표를 공유한다.

열 (통일안 2장)
  반응일: t0, t0i(거래일 위치), timing
  서프라이즈: sue, sue_pct, z_past(메인), z_pct_past, z_full, z_mkt, z_nocovid, z_wins, group, group_1sd
  과거 기록: n_hist, n_hist_covid, hist_sd(예측 난이도), beat, beat_cnt_8q, beat_rate_8q, streak_group, first_miss
  점검 플래그: flag_eps, flag_date, flag_confound, flag_mkt_shock, flag_mixed_timing, flag_time_unknown
  기타: period(hist / main / after), quarter, excluded(교란 제외), robust_exclude(경계 이벤트)
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.did_us.data import read_config
from src.did_us.settings import MARKET, SECTOR_ETF, Params


@dataclass
class Market:
    """거래일 × 종목 행렬. 패널을 만들 때 numpy 인덱스로 빠르게 꺼내 쓴다."""

    cal: pd.DatetimeIndex
    col: dict
    close: np.ndarray
    ret: np.ndarray  # 일별 수익률 (%), 수정종가 기준
    logv: np.ndarray
    vbase: np.ndarray  # 각 거래일 t의 t−60 ~ t−21 평균 log 거래량

    def ar(self, bench=MARKET):
        return self.ret - self.ret[:, [self.col[bench]]]


def build_market(px, P: Params):
    cal = pd.DatetimeIndex(sorted(px.loc[px.ticker == MARKET, "date"].unique()))
    wide = {c: px.pivot(index="date", columns="ticker", values=c).reindex(cal) for c in ("close", "adjclose", "volume")}
    tick = list(wide["close"].columns)
    ret = wide["adjclose"].pct_change(fill_method=None) * 100
    logv = np.log(wide["volume"].where(wide["volume"] > 0))
    lo, hi = P.vol_base
    vbase = logv.rolling(lo - hi + 1, min_periods=int(0.75 * (lo - hi + 1))).mean().shift(hi)
    return Market(cal, {t: i for i, t in enumerate(tick)}, wide["close"].to_numpy(), ret.to_numpy(), logv.to_numpy(), vbase.to_numpy())


def _reaction_index(e, M: Market):
    pos = M.cal.searchsorted(e.announce_date.values)  # 발표일 이후 첫 거래일
    on_day = np.array([p < len(M.cal) and M.cal[p] == d for p, d in zip(pos, e.announce_date)])
    pos = np.where((e.timing.values == "AMC") & on_day, pos + 1, pos)
    ov = e.t0_override.notna().values
    pos[ov] = M.cal.searchsorted(e.t0_override[ov].values)
    return pos


def _history(e, P: Params):
    """발표마다 그 이전 발표(최대 8분기)의 SUE·상회 기록. 미래 정보는 쓰지 않는다."""
    out = {k: np.full(len(e), np.nan) for k in
           ("hist_mean", "hist_sd", "n_hist", "n_hist_covid", "pct_mean", "pct_sd", "n_pct",
            "nc_mean", "nc_sd", "n_nc", "beat_cnt_8q", "n_beat_hist")}
    covid_end, nc_start = pd.Timestamp(P.covid_end), pd.Timestamp(P.nocovid_hist_start)
    for _, g in e.groupby("ticker"):
        g = g.sort_values("t0i")
        idx, sue, pct, beat, day = g.index.values, g.sue.values, g.sue_pct_c.values, g.beat.values, g.announce_date.values
        for j in range(len(g)):
            prev = np.arange(j)
            ps = prev[~np.isnan(sue[prev])][-P.hist_q:]
            i = idx[j]
            out["n_hist"][i] = len(ps)
            out["n_hist_covid"][i] = (day[ps] < covid_end).sum()
            if len(ps) >= 2:
                out["hist_mean"][i], out["hist_sd"][i] = sue[ps].mean(), sue[ps].std(ddof=1)
            pp = prev[~np.isnan(pct[prev])][-P.hist_q:]
            out["n_pct"][i] = len(pp)
            if len(pp) >= 2:
                out["pct_mean"][i], out["pct_sd"][i] = pct[pp].mean(), pct[pp].std(ddof=1)
            pn = prev[~np.isnan(sue[prev]) & (day[prev] >= nc_start)][-P.hist_q:]
            out["n_nc"][i] = len(pn)
            if len(pn) >= 2:
                out["nc_mean"][i], out["nc_sd"][i] = sue[pn].mean(), sue[pn].std(ddof=1)
            pb = prev[~np.isnan(beat[prev])][-P.hist_q:]
            out["n_beat_hist"][i] = len(pb)
            out["beat_cnt_8q"][i] = beat[pb].sum() if len(pb) else np.nan
    return pd.DataFrame(out, index=e.index)


def _score(x, mean, sd, n, floor, P: Params):
    z = (x - mean) / np.maximum(sd, floor)
    z = np.where(n >= P.hist_min, z, np.nan)
    return np.clip(z, -P.score_clip, P.score_clip), z


def build_events(earn, M: Market, P: Params):
    e = earn.copy().reset_index(drop=True)
    e["t0i"] = _reaction_index(e, M)
    e = e[e.t0i < len(M.cal)].reset_index(drop=True)
    e["t0"] = M.cal[e.t0i.values]
    T = len(M.cal)
    cols = np.array([M.col[t] for t in e.ticker])
    t0i = e.t0i.values

    def at(mat, off):
        i = t0i + off
        ok = (i >= 0) & (i < T)
        v = np.full(len(e), np.nan)
        v[ok] = mat[i[ok], cols[ok]]
        return v

    # 서프라이즈 원값
    e["p_t0m2"] = at(M.close, -2)  # 분할만 조정한 종가
    diff = e.eps_actual - e.eps_consensus
    e["sue"] = diff / e.p_t0m2 * 100
    e["sue_pct"] = diff / e.eps_consensus.abs() * 100
    e["sue_pct_c"] = e.sue_pct.clip(-100, 100)
    e["beat"] = np.where(diff.notna(), (diff > 0).astype(float), np.nan)

    e["period"] = np.select([e.announce_date < P.event_start, e.announce_date <= P.event_end], ["hist", "main"], "after")
    e["quarter"] = e.t0.dt.to_period("Q").astype(str)
    e = pd.concat([e, _history(e, P)], axis=1)

    # 회사 기준 점수 (메인 + 비교용 정의). SD 하한은 산업 안에서 정한다
    for ind, g in e.groupby("industry"):
        i = g.index
        fl = np.nanpercentile(g.loc[g.n_hist >= P.hist_min, "hist_sd"], P.sd_floor_pct)
        e.loc[i, "z_past"], raw = _score(g.sue, g.hist_mean, g.hist_sd, g.n_hist, fl, P)
        e.loc[i, "z_past_raw"] = raw
        fp = np.nanpercentile(g.loc[g.n_pct >= P.hist_min, "pct_sd"], P.sd_floor_pct)
        e.loc[i, "z_pct_past"] = _score(g.sue_pct_c, g.pct_mean, g.pct_sd, g.n_pct, fp, P)[0]
        e.loc[i, "z_nocovid"] = _score(g.sue, g.nc_mean, g.nc_sd, g.n_nc, fl, P)[0]
        fm = g.groupby("ticker").sue.agg(["mean", "std"])
        e.loc[i, "z_full"] = np.clip((g.sue - g.ticker.map(fm["mean"])) / np.maximum(g.ticker.map(fm["std"]), fl), -P.score_clip, P.score_clip)
        m = g[g.period == "main"].sue
        e.loc[i, "z_mkt"] = np.clip((g.sue - m.mean()) / m.std(), -P.score_clip, P.score_clip)
        r = e.loc[i, "z_past_raw"]
        lo, hi = np.nanpercentile(r[g.period == "main"], [1, 99])
        e.loc[i, "z_wins"] = r.clip(lo, hi)
        e.loc[i, "sd_floor"] = fl

    e["group"] = pd.cut(e.z_past, [-np.inf, -P.z_cut, P.z_cut, np.inf], labels=["down", "expected", "up"]).astype(object)
    e["group_1sd"] = pd.cut(e.z_past, [-np.inf, -1, 1, np.inf], labels=["down", "expected", "up"]).astype(object)

    # 과거 기록 (핵심 질문)
    e["beat_rate_8q"] = np.where(e.n_beat_hist >= P.hist_min, e.beat_cnt_8q / e.n_beat_hist, np.nan)
    e["streak_group"] = np.select(
        [e.beat_rate_8q.isna(), e.beat_rate_8q >= P.streak_hi / P.hist_q, e.beat_rate_8q <= P.streak_lo / P.hist_q],
        [None, "streak", "occasional"], "mid")
    e["first_miss"] = (e.streak_group == "streak") & (e.beat == 0)
    e["predictability_sd"] = e.hist_sd  # 예측 난이도 대체 지표: 직전 8분기 SUE의 SD

    # 반응일 주변 AR (SPY 기준)과 점검 플래그
    AR = M.ar()
    for k in (-1, 0, 1):
        e[f"ar_{k}"] = at(AR, k)
    e["car01"] = e.ar_0 + e.ar_1
    side = np.maximum(e["ar_-1"].abs(), e.ar_1.abs())
    e["flag_date"] = (side >= 2 * e.ar_0.abs()) & (side >= 3)
    e["flag_eps"] = (e.sue.abs() > 1.5) & (e.sue_pct.abs() > 75)
    e["flag_confound"] = e.car01.abs() > 10
    spy = np.abs(M.ret[:, M.col[MARKET]])
    shock_day = np.nan_to_num(spy) >= P.shock
    cs = np.concatenate([[0], np.cumsum(shock_day)])
    lo_i, hi_i = np.clip(t0i - P.win, 0, T), np.clip(t0i + P.win + 1, 0, T)
    e["flag_mkt_shock"] = (cs[hi_i] - cs[lo_i]) > 0
    e["flag_time_unknown"] = (e.earnings_dt.dt.hour == 0) & (e.earnings_dt.dt.minute == 0)
    mn = e[e.period == "main"].groupby("ticker").timing.agg(lambda s: min((s == "AMC").mean(), (s != "AMC").mean()))
    e["flag_mixed_timing"] = e.ticker.map(mn).fillna(0) >= 0.25

    # 교란 이벤트 (뉴스 확인 결과)
    ex = read_config("exclude_events.csv")
    ex["announce_date"] = pd.to_datetime(ex.announce_date)
    key = e.ticker + e.announce_date.dt.strftime("%Y-%m-%d")
    exk = ex.ticker + ex.announce_date.dt.strftime("%Y-%m-%d")
    e["excluded"] = key.isin(exk[ex.main_exclude == 1])
    e["robust_exclude"] = key.isin(exk[ex.main_exclude == 0])

    e["eid"] = e.ticker + "_" + e.t0.dt.strftime("%Y%m%d")
    keep = [c for c in e.columns if not c.startswith(("pct_", "nc_", "n_pct", "n_nc", "hist_mean"))]
    return e[keep]


def main_events(ev, industry, win=10, T=None):
    """메인 분석 표본: 분석 기간 · 점수 계산 가능 · 교란 제외 · 창이 데이터 안에 있음."""
    m = (ev.industry == industry) & (ev.period == "main") & ev.z_past.notna() & ~ev.excluded
    if T is not None:
        m &= (ev.t0i - 70 >= 0) & (ev.t0i + win < T)
    return ev[m].reset_index(drop=True)


def find_controls(treated, ev_all, uni, M: Market, P: Params, pool=None, h=None, bench=MARKET):
    """처치 반응일 ±h 거래일 안에 자기 반응일이 없는 회사. 자기 발표 판정은 2020~2026-10 전체 기록으로 한다."""
    pool, h = pool or P.ctrl_pool, P.ctrl_h if h is None else h
    own = {t: np.sort(g.t0i.values) for t, g in ev_all.groupby("ticker")}
    sub = uni.set_index("ticker").subgroup
    ind = uni.set_index("ticker").industry
    AR = M.ar(bench)
    T = len(M.cal)
    out = []
    for r in treated.itertuples():
        if pool == "all":
            cand = uni.ticker
        elif pool == "other_subgroup":
            cand = uni.ticker[(uni.industry == r.industry) & (uni.subgroup != r.subgroup)]
        else:
            cand = uni.ticker[uni.industry == r.industry]
        ctrls = []
        lo, hi = r.t0i - P.win, r.t0i + P.win + 1
        for t in cand:
            if t == r.ticker or t not in own:
                continue
            o = own[t]
            j = np.searchsorted(o, r.t0i - h)
            if j < len(o) and o[j] <= r.t0i + h:
                continue
            c = M.col[t]
            if lo < 0 or hi > T or np.isnan(AR[lo:hi, c]).any() or np.isnan(M.vbase[r.t0i, c]):
                continue
            ctrls.append(t)
        out.append(ctrls)
    return out
