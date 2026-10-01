"""KIND(한국거래소 공시 시스템)에서 실적발표 공시의 실제 접수 시각을 수집한다.

실행: .venv/bin/python -m src.data.collect_filing_times --industry finance   (collect_calendar를 먼저 실행)
입력: data/<산업군>/calendar/earnings_calendar.csv
출력: data/<산업군>/calendar/filing_times.csv

- OpenDART API에는 접수 일자만 있고 시각이 없어서 KIND 공시 목록에서 시각을 가져온다
- 캘린더의 (종목, 접수일, 공시 제목)과 같은 KIND 공시를 찾고, 같은 제목이 여러 건이면 가장 이른 시각을 쓴다
- 공시 시각은 실적설명회(IR) 시각과 다를 수 있다. 예: KB금융 2025Q3는 IR이 16:00이지만
  공시는 15:22에 접수되어 장 마감 전이다. 시장이 숫자를 처음 본 시각은 공시 접수 시각이다
- timing: 09:00 이전 장전, 15:00 이전 장중, 15:00~15:30 마감직전, 15:30 이후 장후
  마감직전은 반응할 시간이 30분 이하라 장중으로 볼지 장후로 볼지 애매하므로,
  사건 창을 만들 때 발표일과 다음 거래일을 합친 2일 창으로 처리한다
"""

import re
import time

import pandas as pd
import requests

from src.common import RAW_DIR, ROOT, cli_industry

CACHE_DIR = RAW_DIR / "kind"

KIND_URL = "https://kind.krx.co.kr/disclosure/details.do"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://kind.krx.co.kr/disclosure/details.do?method=searchDetailsMain",
}
MARKET_OPEN, LATE_SESSION, MARKET_CLOSE = 9 * 60, 15 * 60, 15 * 60 + 30

ROW_RE = re.compile(r"<tr[^>]*>.*?</tr>", re.S)
TIME_RE = re.compile(r'<td class="txc">(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2})</td>')
TITLE_RE = re.compile(r"openDisclsViewer\('(\d+)',''\)\" title='([^']*)'")


def normalize(title):
    return re.sub(r"\s+", "", title)


def fetch_day(ticker, date):
    """한 종목의 하루치 KIND 공시 목록을 [(접수일시, 제목, KIND 접수번호)]로 돌려준다. 응답은 캐시한다."""
    cache = CACHE_DIR / f"{ticker}_{date}.html"
    if cache.exists():
        html = cache.read_text(encoding="utf-8")
    else:
        data = {
            "method": "searchDetailsSub", "currentPageSize": "100", "pageIndex": "1",
            "orderMode": "1", "orderStat": "D", "forward": "details_sub",
            "fromDate": date, "toDate": date, "repIsuSrtCd": f"A{ticker}",
        }
        resp = requests.post(KIND_URL, data=data, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        html = resp.text
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(html, encoding="utf-8")
        time.sleep(0.3)

    filings = []
    for tr in ROW_RE.findall(html):
        when, title = TIME_RE.search(tr), TITLE_RE.search(tr)
        if when and title:
            filings.append((f"{when.group(1)} {when.group(2)}", title.group(2), title.group(1)))
    return filings


def to_timing(clock):
    minutes = int(clock[:2]) * 60 + int(clock[3:])
    if minutes < MARKET_OPEN:
        return "장전"
    if minutes < LATE_SESSION:
        return "장중"
    return "마감직전" if minutes < MARKET_CLOSE else "장후"


def main(industry):
    calendar = pd.read_csv(industry.path("calendar"), dtype={"ticker": str, "rcept_no": str})
    records = []
    for row in calendar.itertuples():
        filings = fetch_day(row.ticker, row.announce_date)
        matches = sorted(f for f in filings if normalize(f[1]) == normalize(row.report_nm))
        filing_time = matches[0][0].split(" ")[1] if matches else ""
        records.append({
            "ticker": row.ticker, "name": row.name, "fiscal_q": row.fiscal_q,
            "announce_date": row.announce_date, "rcept_no": row.rcept_no,
            "filing_time": filing_time,
            "timing": to_timing(filing_time) if filing_time else "미확인",
            "n_kind_filings": len(filings), "n_matches": len(matches),
        })

    out = pd.DataFrame(records)
    out_path = industry.path("filing_times")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"저장: {out_path.relative_to(ROOT)} ({len(out)}행)")
    print(out["timing"].value_counts().to_string())


if __name__ == "__main__":
    main(cli_industry(__doc__.splitlines()[0]))
