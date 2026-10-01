"""분석 대상 17개 종목과 KOSPI 지수의 일별 주가·거래량·시가총액을 수집한다.

실행: .venv/bin/python src/data/collect_prices.py
출력:
  data/prices/fin_prices.csv   종목×일 (수정주가, 거래량, 수익률, 시가총액)
  data/prices/kospi.csv        KOSPI 지수 일별 종가와 수익률

출처 조합:
- 종목 수정주가·거래량: pykrx (권리락·배당 등을 반영한 수정주가)
- 시가총액: yfinance 종가 × 발행주식수. 한국거래소 데이터 포털(pykrx 시가총액 경로)이
  점검 중이라 대체했다. 주식 분할이 없다는 가정이며, 근사치라 규모 매칭용으로만 쓴다
- KOSPI 지수: yfinance (^KS11), 빠진 거래일은 FinanceDataReader로 보완
"""

import time

import numpy as np
import FinanceDataReader as fdr
import pandas as pd
import yfinance as yf
from pykrx import stock

from collect_calendar import ROOT, UNIVERSE

OUT_DIR = ROOT / "data" / "prices"
START, END = "20201001", "20260930"  # 2021Q1 발표 전 60거래일 추정 구간을 확보한다


def fetch_adjusted_prices(ticker):
    df = stock.get_market_ohlcv(START, END, ticker, adjusted=True)
    if df.empty:
        raise RuntimeError(f"{ticker}: pykrx 수정주가가 비어 있습니다")
    df = df.rename(columns={"종가": "close_adj", "거래량": "volume"})[["close_adj", "volume"]]
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df


def fetch_market_cap(ticker):
    """yfinance 종가 × 발행주식수(변동일 기준 이력을 앞으로 채움)."""
    yf_ticker = yf.Ticker(f"{ticker}.KS")
    close = yf_ticker.history(start="2020-10-01", end="2026-10-01", auto_adjust=False)["Close"]
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    try:
        shares = yf_ticker.get_shares_full(start="2020-01-01", end="2026-10-01")
    except Exception:
        shares = None
    if shares is None or len(shares) == 0:
        return pd.Series(np.nan, index=close.index, name="market_cap")
    shares.index = pd.to_datetime(shares.index).tz_localize(None).normalize()
    shares = shares[~shares.index.duplicated(keep="last")].sort_index()
    shares = shares.reindex(close.index.union(shares.index)).ffill().reindex(close.index)
    return (close * shares).rename("market_cap")


def fetch_kospi():
    k = yf.download("^KS11", start="2020-10-01", end="2026-10-02", progress=False, auto_adjust=False)
    close = k["Close"].squeeze().dropna()
    close.index = pd.to_datetime(close.index)
    # yfinance에 빠진 거래일은 FinanceDataReader(~2026-09-17까지)로 보완한다
    fdr_close = fdr.DataReader("KS11", "2020-10-01", "2026-09-17")["Close"]
    close = close.combine_first(fdr_close).sort_index()
    out = close.rename("kospi").to_frame()
    out.index.name = "date"
    out["kospi_ret"] = np.log(out["kospi"]).diff()
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    frames = []
    for ticker, (name, sector) in UNIVERSE.items():
        prices = fetch_adjusted_prices(ticker)
        cap = fetch_market_cap(ticker)
        df = prices.join(cap, how="left")
        df["ret"] = np.log(df["close_adj"]).diff()
        df.insert(0, "sector", sector)
        df.insert(0, "name", name)
        df.insert(0, "ticker", ticker)
        frames.append(df.reset_index())
        print(f"{name}: {len(df)}일, {df.index.min().date()} ~ {df.index.max().date()}, "
              f"시가총액 {'있음' if df['market_cap'].notna().any() else '없음'}")
        time.sleep(0.3)

    prices = pd.concat(frames, ignore_index=True)
    prices.to_csv(OUT_DIR / "fin_prices.csv", index=False, encoding="utf-8-sig")

    kospi = fetch_kospi()
    kospi.to_csv(OUT_DIR / "kospi.csv", encoding="utf-8-sig")
    print(f"\n저장: data/prices/fin_prices.csv ({len(prices)}행), data/prices/kospi.csv ({len(kospi)}행)")


if __name__ == "__main__":
    main()
