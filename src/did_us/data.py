"""데이터 수집(yfinance)과 보정(config/did/*_overrides.csv).

수집 결과는 data/raw/us/ 에 캐시한다. refresh=True 로 다시 받는다.
산업이 달라도 이 단계를 지나면 같은 열 이름을 가진다.
  실적: ticker, industry, subgroup, earnings_dt, announce_date, timing, eps_actual, eps_consensus, eps_source, t0_override
  주가: date, ticker, close(분할 조정), adjclose(분할+배당 조정), volume
"""

import time

import numpy as np
import pandas as pd

from src.did_us.settings import CONFIG_DIR, MARKET, RAW_DIR, SECTOR_ETF, Params


def load_universe(industries=None):
    u = pd.read_csv(CONFIG_DIR / "universe.csv", comment="#")
    if industries is not None:
        u = u[u.industry.isin(industries)]
    return u.reset_index(drop=True)


def read_config(name):
    return pd.read_csv(CONFIG_DIR / name, comment="#")


def fetch_earnings(tickers, refresh=False, limit=100):
    """yfinance 실적 발표 기록. limit=100이면 70개사 모두 2020년 이전까지 나온다."""
    import yfinance as yf

    out_dir = RAW_DIR / "earnings"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for t in tickers:
        f = out_dir / f"{t}.csv"
        if refresh or not f.exists():
            df = None
            for _ in range(3):
                try:
                    df = yf.Ticker(t).get_earnings_dates(limit=limit)
                    break
                except Exception:
                    time.sleep(3)
            if df is None or df.empty:
                raise RuntimeError(f"{t}: 실적 기록을 받지 못했다")
            df.reset_index().to_csv(f, index=False)
            time.sleep(0.3)
        d = pd.read_csv(f)
        d["ticker"] = t
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def fetch_prices(tickers, P: Params, refresh=False):
    """일별 종가(분할 조정) · 수정종가(분할+배당 조정) · 거래량."""
    import yfinance as yf

    f = RAW_DIR / "prices.csv"
    need = sorted(set(tickers))
    if f.exists() and not refresh:
        px = pd.read_csv(f, parse_dates=["date"])
        if set(need) <= set(px.ticker) and px.date.min() <= pd.Timestamp(P.price_start) + pd.Timedelta(days=7):
            return px[px.ticker.isin(need)].reset_index(drop=True)
    end = (pd.Timestamp(P.price_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(need, start=P.price_start, end=end, auto_adjust=False, progress=False)
    parts = [raw[c].stack().rename(n) for c, n in [("Close", "close"), ("Adj Close", "adjclose"), ("Volume", "volume")]]
    px = pd.concat(parts, axis=1).reset_index()
    px.columns = ["date", "ticker", "close", "adjclose", "volume"]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    px.to_csv(f, index=False)
    return px


def clean_earnings(raw, universe, P: Params):
    e = raw.rename(columns={"Earnings Date": "earnings_dt", "EPS Estimate": "eps_consensus", "Reported EPS": "eps_actual"})
    e["earnings_dt"] = pd.to_datetime(e.earnings_dt, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    e = e.dropna(subset=["eps_actual"])
    e["announce_date"] = e.earnings_dt.dt.normalize()
    e = e[(e.announce_date >= P.earn_start) & (e.announce_date <= P.price_end)]
    e = e.sort_values(["ticker", "earnings_dt"]).drop_duplicates(["ticker", "announce_date"])
    e = e.merge(universe[["ticker", "industry", "subgroup"]], on="ticker")
    e["eps_source"] = "yfinance"

    # 1) GAAP 혼입 보정: SEC 보도자료의 조정 EPS로 바꾼다 (값이 비어 있으면 결측)
    ov = read_config("eps_overrides.csv")
    ov["announce_date"] = pd.to_datetime(ov.announce_date)
    e = e.merge(ov[["ticker", "announce_date", "eps_adjusted"]], on=["ticker", "announce_date"], how="left", indicator=True)
    hit = e._merge == "both"
    e.loc[hit, "eps_actual"] = e.loc[hit, "eps_adjusted"]
    e.loc[hit, "eps_source"] = "SEC 보도자료 조정 EPS"
    e = e.drop(columns=["eps_adjusted", "_merge"])

    # 2) 발표 시각 → BMO / AMC / 장중
    h = e.earnings_dt.dt.hour + e.earnings_dt.dt.minute / 60
    e["timing"] = np.select([h >= 16, h < 9.5], ["AMC", "BMO"], "INTRA")

    # 3) 발표일·시각 보정: SEC 8-K 접수 시각으로 확인한 반응일
    e["t0_override"] = pd.NaT
    t0o = read_config("t0_overrides.csv")
    for r in t0o.itertuples():
        m = (e.ticker == r.ticker) & (e.announce_date == pd.Timestamp(r.yf_datetime).normalize())
        e.loc[m, "earnings_dt"] = pd.Timestamp(r.announce_datetime)
        e.loc[m, "announce_date"] = pd.Timestamp(r.announce_datetime).normalize()
        e.loc[m, "timing"] = r.timing
        e.loc[m, "t0_override"] = pd.Timestamp(r.t0)
    return e.sort_values(["ticker", "announce_date"]).reset_index(drop=True)


def load_data(industries=None, P: Params = None, refresh=False):
    """회사 목록 · 실적 · 주가를 한 번에 읽는다. 대조군 풀을 세 산업으로 넓히는 강건성 때문에 보통 세 산업을 같이 읽는다."""
    P = P or Params()
    uni = load_universe(industries)
    raw = fetch_earnings(uni.ticker.tolist(), refresh=refresh)
    earn = clean_earnings(raw, uni, P)
    etfs = [SECTOR_ETF[i] for i in uni.industry.unique()]
    px = fetch_prices(uni.ticker.tolist() + [MARKET] + etfs, P, refresh=refresh)
    return uni, earn, px
