"""OpenDART에서 산업군 종목들의 실적발표(잠정실적 공정공시) 캘린더를 수집한다.

실행: .venv/bin/python -m src.data.collect_calendar --industry finance
입력: config/industries/<산업군>.csv
출력: data/<산업군>/calendar/earnings_calendar.csv

- 실적발표일 = '영업(잠정)실적(공정공시)' 공시의 접수일
  (4분기는 '매출액또는손익구조...변경' 공시로만 내는 회사가 많아 함께 수집)
- 분기마다 1건만 남긴다: 회사 본체 공시를 자회사 공시보다 우선하고, 그중 가장 이른
  접수일을 쓴다. 시장이 실적 숫자를 처음 접하는 날이 사건일이기 때문이다
  (예: 메리츠는 4분기 손익구조 공시가 IR 잠정실적보다 약 2주 먼저 나온다)
- 정정공시([기재정정] 등)는 최초 발표가 아니므로 제외한다
- 접수 시각은 API에 없으므로 장중/장후 구분은 별도로 보완해야 한다
"""

import io
import os
import time
import zipfile
import xml.etree.ElementTree as ET

import pandas as pd
import requests
from dotenv import load_dotenv

from src.common import DART_BGN_DE, DART_END_DE, FIRST_QUARTER, RAW_DIR, ROOT, cli_industry

CORP_CODE_CACHE = RAW_DIR / "dart_corp_codes.csv"
BASE_URL = "https://opendart.fss.or.kr/api"

EARNINGS_KEYWORD = "영업(잠정)실적"
ANNUAL_KEYWORD = "매출액또는손익구조"
SUBSIDIARY_MARKER = "자회사의 주요경영사항"
CORRECTION_MARKERS = ("정정", "추가", "연장")


def get_api_key():
    load_dotenv(ROOT / ".env")
    key = os.getenv("DART_API_KEY")
    if not key:
        raise SystemExit(".env에 DART_API_KEY가 없습니다.")
    return key


def load_corp_codes(api_key):
    """종목코드 → DART 고유번호(corp_code) 매핑. 한 번 받으면 캐시한다."""
    if CORP_CODE_CACHE.exists():
        return pd.read_csv(CORP_CODE_CACHE, dtype=str)

    resp = requests.get(f"{BASE_URL}/corpCode.xml", params={"crtfc_key": api_key}, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        xml_bytes = zf.read(zf.namelist()[0])

    rows = []
    for item in ET.fromstring(xml_bytes).iter("list"):
        stock_code = (item.findtext("stock_code") or "").strip()
        if stock_code:
            rows.append({
                "corp_code": item.findtext("corp_code"),
                "corp_name": item.findtext("corp_name"),
                "stock_code": stock_code,
            })
    df = pd.DataFrame(rows)
    CORP_CODE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CORP_CODE_CACHE, index=False)
    return df


def fetch_disclosures(api_key, corp_code):
    """한 회사의 거래소공시(I) 목록을 전체 페이지 수집한다."""
    items, page = [], 1
    while True:
        params = {
            "crtfc_key": api_key,
            "corp_code": corp_code,
            "bgn_de": DART_BGN_DE,
            "end_de": DART_END_DE,
            "pblntf_ty": "I",
            "page_no": page,
            "page_count": 100,
        }
        data = requests.get(f"{BASE_URL}/list.json", params=params, timeout=30).json()
        status = data.get("status")
        if status == "013":  # 조회된 데이터 없음
            break
        if status != "000":
            raise RuntimeError(f"DART 오류 {status}: {data.get('message')}")
        items.extend(data["list"])
        if page >= int(data["total_page"]):
            break
        page += 1
        time.sleep(0.2)
    return items


def to_fiscal_quarter(date):
    """발표일로 대상 분기를 추정한다 (12월 결산 기준, 분기 종료 후 ~45일 내 발표)."""
    if date.month <= 3:
        return f"{date.year - 1}Q4"
    return f"{date.year}Q{(date.month - 1) // 3}"


def classify(report_nm, date):
    """실적발표 공시면 우선순위(낮을수록 우선)를, 아니면 None을 돌려준다."""
    if report_nm.startswith("[") and any(m in report_nm for m in CORRECTION_MARKERS):
        return None
    is_earnings = EARNINGS_KEYWORD in report_nm
    is_annual = ANNUAL_KEYWORD in report_nm and date.month <= 3
    if not (is_earnings or is_annual):
        return None
    return 1 if SUBSIDIARY_MARKER in report_nm else 0


def main(industry):
    api_key = get_api_key()
    corp_codes = load_corp_codes(api_key).set_index("stock_code")

    unknown = [t for t, *_ in industry.rows() if t not in corp_codes.index]
    if unknown:
        raise SystemExit(f"DART 고유번호 목록에 없는 종목코드: {', '.join(unknown)}\n"
                         "상장폐지 종목이거나 종목코드 오타입니다. 설정 파일을 확인하세요.")

    rows = []
    for ticker, name, sector, _ in industry.rows():
        corp_code = corp_codes.loc[ticker, "corp_code"]
        disclosures = fetch_disclosures(api_key, corp_code)
        n_before = len(rows)
        for d in disclosures:
            report_nm = d["report_nm"].strip()
            date = pd.to_datetime(d["rcept_dt"])
            priority = classify(report_nm, date)
            if priority is None:
                continue
            rows.append({
                "ticker": ticker,
                "name": name,
                "sector": sector,
                "fiscal_q": to_fiscal_quarter(date),
                "announce_date": date,
                "priority": priority,
                "report_nm": report_nm,
                "rcept_no": d["rcept_no"],
            })
        print(f"{name}: 공시 {len(disclosures)}건 중 실적 관련 {len(rows) - n_before}건")

    df = pd.DataFrame(rows)
    df = df[df["fiscal_q"] >= FIRST_QUARTER]
    df["n_candidates"] = df.groupby(["ticker", "fiscal_q"])["announce_date"].transform("nunique")
    df = (
        df.sort_values(["ticker", "fiscal_q", "priority", "announce_date", "rcept_no"])
        .drop_duplicates(["ticker", "fiscal_q"], keep="first")
        .sort_values(["sector", "ticker", "fiscal_q"])
    )
    df["announce_time"] = ""
    df["timing"] = "미확인"
    df["source"] = "opendart"
    df["announce_date"] = df["announce_date"].dt.strftime("%Y-%m-%d")

    cols = ["ticker", "name", "sector", "fiscal_q", "announce_date",
            "announce_time", "timing", "source", "report_nm", "rcept_no", "n_candidates"]
    out_path = industry.path("calendar")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df[cols].to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n저장: {out_path.relative_to(ROOT)} ({len(df)}행)")


if __name__ == "__main__":
    main(cli_industry(__doc__.splitlines()[0]))
