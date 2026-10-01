"""실적발표 공시 원문에서 순이익과 발표 시각을 뽑아 SUE(표준화 어닝 서프라이즈)를 계산한다.

실행: .venv/bin/python -m src.data.collect_earnings --industry finance   (collect_calendar를 먼저 실행)
입력: data/<산업군>/calendar/earnings_calendar.csv, filing_times.csv(있으면)
출력: data/<산업군>/earnings/sue.csv            실적발표 1건당 1행
      data/<산업군>/earnings/parse_coverage.csv 종목별 순이익·SUE 추출 성공률

재무제표 API 대신 공시 원문을 쓰는 이유:
- 금융업은 2023년 3분기 이전 보고서가 재무제표 API(fnlttSinglAcntAll)에 없다
- 공시 원문에는 발표 당일 시장이 실제로 본 잠정치와 전년 동기 값이 함께 들어 있다

SUE 정의 (애널리스트 컨센서스 대신 전년 동기 실적을 기대치로 쓰는 seasonal random walk):
  delta_q = NI_q - NI_{q-4}          (둘 다 같은 공시에 적힌 3개월 값)
  SUE_q   = delta_q / std(그 회사의 직전 최대 8개 발표의 delta)
- 표준편차는 해당 발표 이전 값만 써서 미래 정보가 섞이지 않게 한다
- 직전 delta가 MIN_HISTORY개 미만이면 SUE를 비워 둔다
- 손익구조 변경 공시(4분기)는 연간 값만 있으므로, 같은 해 3분기 공시의 누계를 빼서 4분기 값을 만든다
"""

import io
import re
import time
import zipfile

import numpy as np
import pandas as pd
import requests

from src.common import ANALYSIS_START, RAW_DIR, ROOT, cli_industry
from src.data.collect_calendar import ANNUAL_KEYWORD, BASE_URL, get_api_key

DOC_CACHE_DIR = RAW_DIR / "dart_docs"
LOW_COVERAGE = 0.8  # 분석 기간 발표 중 순이익이 추출된 비율이 이보다 낮으면 경고한다

UNIT_TO_WON = {"천원": 1e3, "백만원": 1e6, "억원": 1e8, "십억원": 1e9, "조원": 1e12, "원": 1.0}
OWNERS_LABEL = "지배기업소유주지분순이익"
TOTAL_LABEL = "당기순이익"

MAX_HISTORY = 8
MIN_HISTORY = 4
MARKET_OPEN, MARKET_CLOSE = 9 * 60, 15 * 60 + 30

NUMBER_RE = re.compile(r"^[+-]?[\d,]+(\.\d+)?%?$")
# 2025년 3분기부터 서식이 바뀌어 증감률 뒤에 흑자적자전환여부 열이 추가됐다
NEW_FORMAT_MARKER = "흑자적자전환여부"


def load_tokens(api_key, rcept_no):
    """공시 원문(XML)을 받아 표의 셀 단위 문자열 목록으로 바꾼다. 원문은 파일로 캐시한다."""
    cache = DOC_CACHE_DIR / f"{rcept_no}.xml"
    if not cache.exists():
        resp = requests.get(f"{BASE_URL}/document.xml",
                            params={"crtfc_key": api_key, "rcept_no": rcept_no}, timeout=60)
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            raw = zf.read(zf.namelist()[0])
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(raw)
        time.sleep(0.15)

    raw = cache.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("cp949", errors="ignore")
    text = re.sub(r"<style.*?</style>", "", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", "|", text).replace("&nbsp;", " ")
    return [t.strip() for t in text.split("|") if t.strip()]


def is_cell_value(token):
    """표의 값 칸인지 판별한다 (숫자, '-', '흑자전환' 같은 표기)."""
    return (bool(NUMBER_RE.match(token)) or token in ("-", "%", "N/A")
            or "전환" in token or "적자" in token)


def to_number(token):
    return float(token.replace(",", "").rstrip("%")) if NUMBER_RE.match(token) else np.nan


def find_unit(tokens):
    match = re.search(r"단위\s*[:：]?\s*(천원|백만원|십억원|억원|조원|원)", " ".join(tokens))
    return UNIT_TO_WON[match.group(1)] if match else None


def parse_preliminary(tokens):
    """잠정실적 공시에서 (지배주주순이익, 전체 순이익) 각각의 당해·누계 값을 뽑는다.

    표의 각 항목은 [항목명, '당해실적', 값들, '누계실적', 값들] 순서다.
    값은 구 서식에서 [당기, 전기, 증감률, 전년동기, 증감률] 5개,
    신 서식에서 [당기, 전기, 증감률, 전환여부, 전년동기, 증감률, 전환여부] 7개다.
    """
    unit = find_unit(tokens)
    width, prev_year = (7, 4) if NEW_FORMAT_MARKER in tokens else (5, 3)
    # 값이 width개 이어지는 행을 모두 찾는다: (행 바로 앞 항목명, 값들)
    rows = []
    for i in range(1, len(tokens) - width):
        values = tokens[i + 1:i + 1 + width]
        if not is_cell_value(tokens[i]) and all(is_cell_value(v) for v in values):
            label = tokens[i - 1].replace(" ", "") if not is_cell_value(tokens[i - 1]) else None
            rows.append((label, values))
    if len(rows) < 2 or len(rows) % 2:
        return None

    # 당해·누계가 한 쌍이므로 2행씩 묶는다. 항목명은 당해 행 앞에만 있다
    blocks = [(rows[i][0], rows[i][1], rows[i + 1][1]) for i in range(0, len(rows), 2)]

    def pick(label, fallback_index):
        for name, quarter_values, cumulative_values in blocks:
            if name == label:
                return quarter_values, cumulative_values
        # 원문 글자가 깨진 공시는 항목명을 읽을 수 없어 표의 위치로 찾는다
        if all(name not in (OWNERS_LABEL, TOTAL_LABEL) for name, _, _ in blocks):
            if -len(blocks) <= fallback_index < len(blocks):
                return blocks[fallback_index][1], blocks[fallback_index][2]
        return None, None

    is_consolidated = len(blocks) >= 5
    labels_readable = any(name in (OWNERS_LABEL, TOTAL_LABEL) for name, _, _ in blocks)
    if unit is None and not labels_readable:
        unit = UNIT_TO_WON["백만원"]  # 글자가 깨져 단위를 읽을 수 없는 공시. 서식 기본 단위로 간주
    out = {"unit": unit}
    for key, label, index in (("owners", OWNERS_LABEL, 4), ("total", TOTAL_LABEL, 3)):
        if key == "owners" and not is_consolidated:
            continue
        quarter_values, cumulative_values = pick(label, index)
        if quarter_values is None:
            continue
        out[key] = {
            "ni": to_number(quarter_values[0]),
            "prior": to_number(quarter_values[1]),
            "ni_prev": to_number(quarter_values[prev_year]),
            "cum": to_number(cumulative_values[0]),
            "cum_prev": to_number(cumulative_values[prev_year]),
        }
    return out


def parse_annual(tokens):
    """손익구조 변경 공시에서 연간 순이익(당해, 직전)과 그 값이 지배주주 기준인지 여부를 뽑는다."""
    text = " ".join(tokens)
    unit = find_unit(tokens)
    for i, token in enumerate(tokens):
        if token.replace(" ", "") == "-당기순이익" and i + 2 < len(tokens):
            is_owners = bool(re.search(r"당기순이익.{0,3}은\s*지배", text))
            return {"unit": unit, "annual": to_number(tokens[i + 1]),
                    "annual_prev": to_number(tokens[i + 2]), "is_owners": is_owners}
    return None


def parse_ir_time(tokens):
    """'정보제공(예정)일시'에 적힌 시각을 'HH:MM'으로 돌려준다. 시각이 없으면 None."""
    for i, token in enumerate(tokens[:-1]):
        label = token.replace(" ", "")
        if label.startswith("정보제공(예정)일시") or label.startswith("정보제공(예정)시간"):
            value = tokens[i + 1]
            match = re.search(r"(\d{1,2}):(\d{2})", value)
            if match:
                return f"{int(match.group(1)):02d}:{match.group(2)}"
            match = re.search(r"(오전|오후)\s*(\d{1,2})시(?:\s*(\d{1,2})분)?", value)
            if match:
                hour = int(match.group(2)) % 12 + (12 if match.group(1) == "오후" else 0)
                return f"{hour:02d}:{int(match.group(3) or 0):02d}"
            return None
    return None


def to_timing(ir_time):
    if not ir_time:
        return "미확인"
    minutes = int(ir_time[:2]) * 60 + int(ir_time[3:])
    if minutes < MARKET_OPEN:
        return "장전"
    return "장중" if minutes < MARKET_CLOSE else "장후"


def build_events(api_key, calendar):
    """발표 1건마다 3개월 순이익(당기, 전년 동기)을 원 단위로 정리한다."""
    parsed = {}
    for row in calendar.itertuples():
        tokens = load_tokens(api_key, row.rcept_no)
        is_annual = ANNUAL_KEYWORD in row.report_nm
        parsed[(row.ticker, row.fiscal_q)] = (
            is_annual, parse_annual(tokens) if is_annual else parse_preliminary(tokens),
            parse_ir_time(tokens))

    records = []
    for row in calendar.itertuples():
        is_annual, data, ir_time = parsed[(row.ticker, row.fiscal_q)]
        ni = ni_prev = np.nan
        basis = None
        if data and data.get("unit"):
            annual = annual_prev = np.nan
            if not is_annual:
                basis = "owners" if "owners" in data else "total" if "total" in data else None
                if basis:
                    values = data[basis]
                    ni = values["ni"] * data["unit"]
                    ni_prev = values["ni_prev"] * data["unit"]
                    quarter = int(row.fiscal_q[-1])
                    # 우리금융처럼 4분기 공시에 연간 값만 적는 경우: 당기 칸이 연간, 전기 칸이 전년 연간
                    if (quarter == 4 and np.isnan(values["ni_prev"])
                            and np.isnan(values["cum"])):
                        annual, annual_prev = ni, values["prior"] * data["unit"]
                        ni = np.nan
                    # 당해실적 칸에 3개월 값 대신 누계를 적은 공시: 직전 분기 공시의 누계를 뺀다
                    elif quarter in (2, 3) and values["ni"] == values["cum"]:
                        before = parsed.get((row.ticker, f"{row.fiscal_q[:4]}Q{quarter - 1}"))
                        ni = ni_prev = np.nan
                        if before and not before[0] and before[1] and basis in before[1]:
                            scale = before[1]["unit"]
                            ni = values["cum"] * data["unit"] - before[1][basis]["cum"] * scale
                            ni_prev = (values["cum_prev"] * data["unit"]
                                       - before[1][basis]["cum_prev"] * scale)
                            basis += "_cum_minus_prev_q"
            else:
                basis = "owners" if data["is_owners"] else "total"
                annual = data["annual"] * data["unit"]
                annual_prev = data["annual_prev"] * data["unit"]
            if not np.isnan(annual):
                # 연간 값에서 같은 해 3분기 공시의 누계를 빼 4분기 3개월 값을 만든다
                q3 = parsed.get((row.ticker, row.fiscal_q[:4] + "Q3"))
                if q3 and not q3[0] and q3[1] and q3[1].get("unit") and basis in q3[1]:
                    ni = annual - q3[1][basis]["cum"] * q3[1]["unit"]
                    ni_prev = annual_prev - q3[1][basis]["cum_prev"] * q3[1]["unit"]
                    basis += "_annual_minus_q3"
        records.append({
            "ticker": row.ticker, "name": row.name, "sector": row.sector,
            "fiscal_q": row.fiscal_q, "announce_date": row.announce_date,
            "ir_time": ir_time or "", "timing": to_timing(ir_time),
            "net_income": ni, "net_income_prev_year": ni_prev,
            "basis": basis if not np.isnan(ni) else None, "rcept_no": row.rcept_no,
        })
    return pd.DataFrame(records)


def add_sue(events):
    events = events.sort_values(["ticker", "fiscal_q"]).reset_index(drop=True)
    events["delta"] = events["net_income"] - events["net_income_prev_year"]
    events["yoy_growth"] = events["delta"] / events["net_income_prev_year"].abs()
    history = events.groupby("ticker")["delta"].transform(
        lambda s: s.shift(1).rolling(MAX_HISTORY, min_periods=MIN_HISTORY).std())
    events["n_history"] = events.groupby("ticker")["delta"].transform(
        lambda s: s.shift(1).rolling(MAX_HISTORY, min_periods=1).count())
    events["delta_std"] = history
    events["sue"] = events["delta"] / events["delta_std"]
    return events


def coverage_report(events):
    """분석 기간 발표 중 순이익·SUE가 채워진 비율을 종목별로 정리한다."""
    recent = events[events["announce_date"] >= ANALYSIS_START]
    report = recent.groupby(["ticker", "name"]).agg(
        n_events=("rcept_no", "size"),
        net_income_ok=("net_income", lambda s: s.notna().mean()),
        sue_ok=("sue", lambda s: s.notna().mean()),
    ).reset_index()
    return report


def main(industry):
    api_key = get_api_key()
    calendar = pd.read_csv(industry.path("calendar"), dtype={"ticker": str, "rcept_no": str})
    events = add_sue(build_events(api_key, calendar))
    filing_times_path = industry.path("filing_times")
    if filing_times_path.exists():
        # 공시 접수 시각(KIND)이 IR 예정 시각보다 정확하므로 timing을 덮어쓴다
        filed = pd.read_csv(filing_times_path, dtype={"rcept_no": str})[["rcept_no", "filing_time", "timing"]]
        events = events.drop(columns="timing").merge(filed, on="rcept_no", how="left")
        events["timing"] = events["timing"].fillna("미확인")

    cols = ["ticker", "name", "sector", "fiscal_q", "announce_date", "ir_time", "filing_time", "timing",
            "net_income", "net_income_prev_year", "delta", "yoy_growth",
            "delta_std", "n_history", "sue", "basis", "rcept_no"]
    out_path = industry.path("sue")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    events[cols].to_csv(out_path, index=False, encoding="utf-8-sig")

    coverage = coverage_report(events)
    coverage.to_csv(industry.path("coverage"), index=False, encoding="utf-8-sig")

    print(f"저장: {out_path.relative_to(ROOT)} ({len(events)}행)")
    print(f"  순이익 추출 {events['net_income'].notna().sum()}행, "
          f"SUE 계산 {events['sue'].notna().sum()}행, "
          f"공시 시각 확인 {(events['timing'] != '미확인').sum()}행")
    low = coverage[coverage["net_income_ok"] < LOW_COVERAGE]
    if not low.empty:
        print(f"\n[경고] 분석 기간 순이익 추출률이 {LOW_COVERAGE:.0%} 미만인 종목 {len(low)}곳")
        print("  공시 서식이 금융업과 달라 파서가 값을 못 읽었을 수 있습니다. 해당 공시 원문을 확인하세요")
        print("  (원문 캐시: data/raw/dart_docs/<접수번호>.xml)")
        print(low.to_string(index=False, float_format=lambda v: f"{v:.0%}"))


if __name__ == "__main__":
    main(cli_industry(__doc__.splitlines()[0]))
