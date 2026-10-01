"""산업군 종목들과 시장 지수의 일별 주가·거래량·시가총액을 수집한다.

실행: .venv/bin/python -m src.data.collect_prices --industry finance
입력: config/industries/<산업군>.csv
출력:
  data/<산업군>/prices/prices.csv        종목×일 (수정주가, 거래량, 수익률, 시가총액)
  data/<산업군>/prices/market_index.csv  KOSPI(·KOSDAQ) 지수 일별 종가와 수익률

출처 조합:
- 종목 수정주가·거래량: pykrx (권리락·배당 등을 반영한 수정주가)
- 시가총액: yfinance 종가 × 발행주식수. 한국거래소 데이터 포털(pykrx 시가총액 경로)이
  점검 중이라 대체했다. 주식 분할이 없다는 가정이며, 근사치라 규모 매칭용으로만 쓴다
- 시장 지수: yfinance (^KS11, ^KQ11), 빠진 거래일은 FinanceDataReader로 보완
  KOSDAQ 종목은 KOSDAQ 지수를 벤치마크로 쓴다 (설정 파일의 market 컬럼)
"""

import time

import numpy as np
import FinanceDataReader as fdr
import pandas as pd
import yfinance as yf
from pykrx import stock

from src.common import PRICE_END, PRICE_START, ROOT, cli_industry

YF_SUFFIX = {"KOSPI": ".KS", "KOSDAQ": ".KQ"}
INDEX_CODES = {"KOSPI": ("^KS11", "KS11"), "KOSDAQ": ("^KQ11", "KQ11")}


def iso(yyyymmdd, days=0):
    return (pd.to_datetime(yyyymmdd) + pd.Timedelta(days=days)).strftime("%Y-%m-%d")


def fetch_adjusted_prices(ticker):
    df = stock.get_market_ohlcv(PRICE_START, PRICE_END, ticker, adjusted=True)
    if df.empty:
        raise RuntimeError(f"{ticker}: pykrx 수정주가가 비어 있습니다 (종목코드나 상장 기간을 확인하세요)")
    df = df.rename(columns={"종가": "close_adj", "거래량": "volume"})[["close_adj", "volume"]]
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df


def fetch_market_cap(ticker, market):
    """yfinance 종가 × 발행주식수(변동일 기준 이력을 앞으로 채움)."""
    yf_ticker = yf.Ticker(f"{ticker}{YF_SUFFIX[market]}")
    close = yf_ticker.history(start=iso(PRICE_START), end=iso(PRICE_END, 1), auto_adjust=False)["Close"]
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    try:
        shares = yf_ticker.get_shares_full(start="2020-01-01", end=iso(PRICE_END, 1))
    except Exception:
        shares = None
    if shares is None or len(shares) == 0:
        return pd.Series(np.nan, index=close.index, name="market_cap")
    shares.index = pd.to_datetime(shares.index).tz_localize(None).normalize()
    shares = shares[~shares.index.duplicated(keep="last")].sort_index()
    shares = shares.reindex(close.index.union(shares.index)).ffill().reindex(close.index)
    return (close * shares).rename("market_cap")


def fetch_index(market):
    yf_code, fdr_code = INDEX_CODES[market]
    k = yf.download(yf_code, start=iso(PRICE_START), end=iso(PRICE_END, 2), progress=False, auto_adjust=False)
    close = k["Close"].squeeze().dropna()
    close.index = pd.to_datetime(close.index)
    # yfinance에 빠진 거래일은 FinanceDataReader로 보완한다
    try:
        fdr_close = fdr.DataReader(fdr_code, iso(PRICE_START), iso(PRICE_END))["Close"]
        close = close.combine_first(fdr_close).sort_index()
    except Exception:
        pass
    name = market.lower()
    out = close.rename(name).to_frame()
    out.index.name = "date"
    out[f"{name}_ret"] = np.log(out[name]).diff()
    return out


def main(industry):
    out_dir = industry.path("prices").parent
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for ticker, name, sector, market in industry.rows():
        prices = fetch_adjusted_prices(ticker)
        cap = fetch_market_cap(ticker, market)
        df = prices.join(cap, how="left")
        df["ret"] = np.log(df["close_adj"]).diff()
        df.insert(0, "market", market)
        df.insert(0, "sector", sector)
        df.insert(0, "name", name)
        df.insert(0, "ticker", ticker)
        frames.append(df.reset_index())
        print(f"{name}: {len(df)}일, {df.index.min().date()} ~ {df.index.max().date()}, "
              f"시가총액 {'있음' if df['market_cap'].notna().any() else '없음'}")
        time.sleep(0.3)

    prices = pd.concat(frames, ignore_index=True)
    prices.to_csv(industry.path("prices"), index=False, encoding="utf-8-sig")

    index = pd.concat([fetch_index(m) for m in sorted(industry.universe["market"].unique())], axis=1)
    index.to_csv(industry.path("index"), encoding="utf-8-sig")
    print(f"\n저장: {industry.path('prices').relative_to(ROOT)} ({len(prices)}행), "
          f"{industry.path('index').relative_to(ROOT)} ({len(index)}행)")


if __name__ == "__main__":
    main(cli_industry(__doc__.splitlines()[0]))
