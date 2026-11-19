"""공통 파이프라인 설정. 산업에 따라 달라지는 값은 여기와 config/did/ 에만 둔다."""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config" / "did"
RAW_DIR = ROOT / "data" / "raw" / "us"  # yfinance 수집 캐시 (저장소에 올리지 않음)
OUT_DIR = ROOT / "output"

INDUSTRIES = ("insurance", "bio", "it")
MARKET = "SPY"  # 메인 벤치마크: AR = 회사 수익률 − SPY 수익률
SECTOR_ETF = {"insurance": "KIE", "bio": "XLV", "it": "XLK"}  # 강건성용 업종 ETF

# 강건성에서 빼 보는 산업별 기간 (발표일 기준)
ROBUST_EXCLUDE_PERIODS = {
    "insurance": [("2025-04-01", "2025-05-31", "2025Q1 실적 (LA 산불)")],
}

# 2020~2022년 실적이 코로나 백신·치료제·진단 매출에 크게 좌우된 기업 (강건성에서 제외해 본다, 담당자 확인 필요)
COVID_FIRMS = {"bio": ["PFE", "MRK", "LLY", "GILD", "REGN", "TMO", "DHR", "A"]}


@dataclass
class Params:
    # 기간
    price_start: str = "2019-12-01"
    price_end: str = "2026-10-31"  # 고정 날짜: 다시 돌려도 결과가 같도록
    earn_start: str = "2020-01-01"  # 직전 8분기 점수와 대조군 판정에 쓰는 실적 기록 시작
    event_start: str = "2022-01-01"  # 분석 이벤트 (발표일 기준)
    event_end: str = "2026-09-30"
    covid_end: str = "2022-01-01"  # 이 날짜 이전 발표 = 코로나 시기 기록
    nocovid_hist_start: str = "2021-01-01"  # 코로나 분기 제외 점수: 이 날짜 이후 발표만 기록으로 사용

    # 창 · 대조군
    win: int = 10  # 분석 창 [−win, +win] 거래일
    ctrl_h: int = 10  # 대조군: 처치 반응일 ±ctrl_h 거래일 안에 자기 반응일이 없어야 함
    min_ctrl: int = 1
    ctrl_pool: str = "industry"  # industry | other_subgroup | all (세 산업 합친 풀)

    # 서프라이즈 점수
    hist_q: int = 8
    hist_min: int = 4
    sd_floor_pct: float = 10.0  # SD 하한 = 산업 내 직전 8분기 SD의 하위 10% 값
    score_clip: float = 5.0
    z_cut: float = 0.5  # 커짐 / 예상 / 작아짐 기준

    # 과거 기록 (핵심 질문) — 회의에서 확정 전 임시값
    streak_hi: int = 7  # 직전 8분기 중 7회 이상 상회 = 연속 상회
    streak_lo: int = 4  # 4회 이하 = 가끔 상회

    # 결과변수
    vol_base: tuple = (60, 21)  # 비정상 거래량 기준구간 t−60 ~ t−21
    ref_bin: tuple = (-10, -6)  # |AR|·거래량 이벤트스터디 기준구간

    # 점검 · 플라시보
    shock: float = 3.0  # |SPY| ≥ 3% = 시장 급변일
    placebo_shift: int = 30
    placebo_grid: tuple = (25, 27, 29, 31, 33, 35, 37)
