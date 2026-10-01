"""파이프라인 공통 설정: 산업군 설정 파일 읽기, 산출물 경로 규칙, 수집 기간.

산업군마다 config/industries/<이름>.csv 에 종목 목록을 두고, 산출물은 data/<이름>/ 아래에 쌓인다.
모든 스크립트는 `--industry <이름>`을 받는다 (기본값: finance).
"""

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config" / "industries"
RAW_DIR = ROOT / "data" / "raw"  # 산업군 공통 캐시 (공시 원문, KIND 응답, DART 고유번호)

# 수집 기간. 바꾸면 모든 산업군에 적용된다
DART_BGN_DE = "20190101"  # 공시 목록 시작일. SUE 과거값(직전 8개 분기)을 확보하려고 분석 시작보다 2년 앞선다
DART_END_DE = "20260930"
FIRST_QUARTER = "2019Q1"
PRICE_START, PRICE_END = "20201001", "20260930"  # 분석 시작 직전 60거래일 이상을 확보한다
ANALYSIS_START = "2021-01-01"  # 이 날짜 이후 발표만 사건으로 분석한다

MARKETS = ("KOSPI", "KOSDAQ")

# 산출물 종류 → (하위 폴더, 파일명)
FILES = {
    "calendar": ("calendar", "earnings_calendar.csv"),
    "filing_times": ("calendar", "filing_times.csv"),
    "sue": ("earnings", "sue.csv"),
    "coverage": ("earnings", "parse_coverage.csv"),
    "prices": ("prices", "prices.csv"),
    "index": ("prices", "market_index.csv"),
    "events": ("events", "events.csv"),
    "panel": ("events", "event_panel.csv"),
}


class Industry:
    def __init__(self, name):
        self.name = name
        self.universe = self._load_universe()

    def _load_universe(self):
        path = CONFIG_DIR / f"{self.name}.csv"
        if not path.exists():
            available = sorted(p.stem for p in CONFIG_DIR.glob("*.csv") if not p.stem.startswith("_"))
            raise SystemExit(f"산업군 설정 파일이 없습니다: {path.relative_to(ROOT)}\n"
                             f"사용 가능한 산업군: {', '.join(available)}\n"
                             f"새로 만들려면 config/industries/_template.csv 를 복사하세요.")

        df = pd.read_csv(path, dtype=str, comment="#", skipinitialspace=True).fillna("")
        missing = {"ticker", "name", "sector"} - set(df.columns)
        if missing:
            raise SystemExit(f"{path.name}에 필수 컬럼이 없습니다: {', '.join(sorted(missing))}")
        for col in df.columns:
            df[col] = df[col].str.strip()
        if "market" not in df.columns:
            df["market"] = ""
        df["market"] = df["market"].str.upper().replace("", "KOSPI")
        df["ticker"] = df["ticker"].str.zfill(6)

        problems = []
        if df.empty:
            problems.append("종목이 하나도 없습니다")
        problems += [f"종목코드가 6자리 숫자가 아닙니다: {t}" for t in df["ticker"] if not (len(t) == 6 and t.isdigit())]
        problems += [f"중복된 종목코드: {t}" for t in df.loc[df["ticker"].duplicated(), "ticker"]]
        problems += [f"sector가 비어 있습니다: {t}" for t in df.loc[df["sector"] == "", "ticker"]]
        problems += [f"market은 KOSPI 또는 KOSDAQ이어야 합니다: {t} ({m})"
                     for t, m in zip(df["ticker"], df["market"]) if m not in MARKETS]
        if problems:
            raise SystemExit(f"{path.name} 설정 오류:\n  " + "\n  ".join(problems))
        return df

    @property
    def sectors(self):
        return sorted(self.universe["sector"].unique())

    def rows(self):
        """(ticker, name, sector, market) 튜플을 설정 파일 순서대로 돌려준다."""
        return list(self.universe[["ticker", "name", "sector", "market"]].itertuples(index=False, name=None))

    def market_of(self, ticker):
        return self.universe.set_index("ticker").loc[ticker, "market"]

    def path(self, kind):
        subdir, filename = FILES[kind]
        return ROOT / "data" / self.name / subdir / filename

    @property
    def results_dir(self):
        return ROOT / "data" / self.name / "results"

    def describe(self):
        counts = self.universe.groupby("sector").size().to_dict()
        return f"산업군 '{self.name}': {len(self.universe)}개 종목, 업종별 {counts}"


def load_industry(name):
    return Industry(name)


def cli_industry(description=None):
    """스크립트의 공통 명령행 인자를 읽어 Industry를 돌려준다."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--industry", default="finance",
                        help="config/industries/<이름>.csv 의 이름 (기본값: finance)")
    return load_industry(parser.parse_args().industry)
