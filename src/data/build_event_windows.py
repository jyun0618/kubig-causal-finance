"""실적발표별 사건 창 데이터(초과수익률, 비정상 거래량)를 만든다.

실행: .venv/bin/python src/data/build_event_windows.py
  (collect_calendar → collect_filing_times → collect_earnings → collect_prices 이후)
출력:
  data/events/fin_events.csv        발표 1건당 1행 (t0, 처치 변수 SUE, 누적초과수익률 등)
  data/events/fin_event_panel.csv   발표×상대일(-5..+3) 일별 수익률·초과수익률·비정상 거래량

사건일(t0)과 반응 구간 규칙 (공시 접수 시각 기준):
- 장후(15:30 이후): t0 = 발표일 다음 거래일, 반응일 1일
- 그 외(장전·장중·마감직전·시각 미확인): t0 = 발표일, 반응일 2일(t0, t0+1)
- 발표일이 휴장일이면 t0는 그 다음 첫 거래일
장중·장전 공시도 2일 창을 쓰는 이유: 접수 시간대와 무관하게 다음 날 변동성이 더 컸다
(변동성은 장후 건만 t0 하루에 몰림). 은행이 공시 뒤 저녁 IR에서 주주환원책 등을 추가
발표하기 때문으로 추정한다(미검증). car_0(1일)과 car_0_1(2일)을 함께 남겨 비교할 수 있다

초과수익률은 시장모형으로 구한다: AR = r - (alpha + beta * r_KOSPI)
- 추정 구간 [-65, -6] 거래일에서 OLS로 alpha, beta를 추정한다 (유효 관측 MIN_EST_OBS개 이상)
비정상 거래량 = log(거래량) - 추정 구간 평균 log(거래량)
"""

import numpy as np
import pandas as pd

from collect_calendar import ROOT

OUT_DIR = ROOT / "data" / "events"
PRE, POST = 5, 3
EST_START, EST_END = -65, -6
MIN_EST_OBS = 40
SINGLE_DAY_TIMINGS = ("장후",)


def load_inputs():
    prices = pd.read_csv(ROOT / "data/prices/fin_prices.csv", parse_dates=["date"], dtype={"ticker": str})
    kospi = pd.read_csv(ROOT / "data/prices/kospi.csv", parse_dates=["date"]).set_index("date")
    sue = pd.read_csv(ROOT / "data/earnings/fin_sue.csv", parse_dates=["announce_date"], dtype={"ticker": str})
    return prices, kospi, sue


def locate_t0(announce_date, timing, days):
    """t0의 거래일 위치와 반응일 수를 돌려준다."""
    i = days.searchsorted(announce_date)  # 발표일 이후 첫 거래일
    on_trading_day = i < len(days) and days[i] == announce_date
    if timing == "장후" and on_trading_day:
        i += 1
    return i, (1 if timing in SINGLE_DAY_TIMINGS else 2)


def fit_market_model(ret, mkt):
    ok = ret.notna() & mkt.notna()
    if ok.sum() < MIN_EST_OBS:
        return None
    x, y = mkt[ok].to_numpy(), ret[ok].to_numpy()
    beta = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
    return y.mean() - beta * x.mean(), beta, int(ok.sum())


def main():
    prices, kospi, sue = load_inputs()
    days = pd.DatetimeIndex(sorted(prices["date"].unique()))
    mkt = kospi["kospi_ret"].reindex(days)

    by_ticker = {t: g.set_index("date").reindex(days) for t, g in prices.groupby("ticker")}
    events = sue[sue["announce_date"] >= "2021-01-01"].sort_values(["announce_date", "ticker"])

    event_rows, panel_rows, dropped = [], [], {"추정 구간 부족": 0, "창 내 결측": 0, "기간 밖": 0}
    for ev in events.itertuples():
        i, reaction_days = locate_t0(ev.announce_date, ev.timing, days)
        if i + POST >= len(days) or i + EST_START < 0:
            dropped["기간 밖"] += 1
            continue

        g = by_ticker[ev.ticker]
        est = slice(i + EST_START, i + EST_END + 1)
        fit = fit_market_model(g["ret"].iloc[est], mkt.iloc[est])
        if fit is None:
            dropped["추정 구간 부족"] += 1
            continue
        alpha, beta, n_est = fit

        win = slice(i - PRE, i + POST + 1)
        ret, volume = g["ret"].iloc[win], g["volume"].iloc[win]
        if ret.isna().any() or mkt.iloc[win].isna().any() or volume.isna().any():
            dropped["창 내 결측"] += 1
            continue

        ar = ret.to_numpy() - (alpha + beta * mkt.iloc[win].to_numpy())
        abn_vol = np.log(volume.to_numpy()) - np.log(g["volume"].iloc[est]).mean()
        rel = np.arange(-PRE, POST + 1)
        for k in range(len(rel)):
            panel_rows.append((ev.ticker, ev.fiscal_q, int(rel[k]), days[i + rel[k]], ret.iloc[k],
                               mkt.iloc[win].iloc[k], ar[k], abn_vol[k]))

        at = {r: ar[k] for k, r in enumerate(rel)}
        event_rows.append({
            "ticker": ev.ticker, "name": ev.name, "sector": ev.sector, "fiscal_q": ev.fiscal_q,
            "announce_date": ev.announce_date.date(), "filing_time": ev.filing_time, "timing": ev.timing,
            "t0_date": days[i].date(), "reaction_days": reaction_days,
            "sue": ev.sue, "yoy_growth": ev.yoy_growth,
            "log_mcap": np.log(g["market_cap"].iloc[i - 1]) if pd.notna(g["market_cap"].iloc[i - 1]) else np.nan,
            "alpha": alpha, "beta": beta, "n_est": n_est,
            "car_pre": sum(at[r] for r in range(-PRE, 0)),
            "car_react": sum(at[r] for r in range(reaction_days)),
            "car_0": at[0],
            "car_0_1": at[0] + at[1],
            "car_0_3": sum(at[r] for r in range(0, POST + 1)),
            "abn_vol_react": float(np.mean(abn_vol[PRE:PRE + reaction_days])),
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(event_rows).to_csv(OUT_DIR / "fin_events.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(panel_rows, columns=["ticker", "fiscal_q", "rel_day", "date", "ret", "mkt_ret", "ar", "abn_logvol"]
                 ).to_csv(OUT_DIR / "fin_event_panel.csv", index=False, encoding="utf-8-sig")
    print(f"저장: data/events/ (사건 {len(event_rows)}건 / 대상 {len(events)}건)")
    print("제외:", {k: v for k, v in dropped.items() if v})


if __name__ == "__main__":
    main()
