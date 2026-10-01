"""산업군 하나에 대해 파이프라인 전체(수집 → 사건 창 → DiD)를 순서대로 실행한다.

실행:
  .venv/bin/python -m src.run_pipeline --industry finance            # 전체
  .venv/bin/python -m src.run_pipeline --industry finance --from earnings   # earnings 단계부터
  .venv/bin/python -m src.run_pipeline --industry finance --only did        # 한 단계만
  .venv/bin/python -m src.run_pipeline --list                              # 단계 목록
"""

import argparse
import importlib
import time

from src.common import load_industry

STEPS = [
    ("calendar", "src.data.collect_calendar", "실적발표 캘린더 수집 (OpenDART)"),
    ("filing_times", "src.data.collect_filing_times", "공시 접수 시각 수집 (KIND)"),
    ("earnings", "src.data.collect_earnings", "순이익 추출·SUE 계산 (공시 원문)"),
    ("prices", "src.data.collect_prices", "주가·거래량·시장 지수 수집"),
    ("events", "src.data.build_event_windows", "사건 창 데이터 생성"),
    ("did", "src.did.run_did", "통제군 매칭·DiD 추정·검증"),
]
STEP_NAMES = [name for name, _, _ in STEPS]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--industry", default="finance", help="config/industries/<이름>.csv 의 이름")
    parser.add_argument("--from", dest="start", choices=STEP_NAMES, help="이 단계부터 끝까지 실행")
    parser.add_argument("--only", choices=STEP_NAMES, help="이 단계만 실행")
    parser.add_argument("--list", action="store_true", help="단계 목록을 보고 종료")
    args = parser.parse_args()

    if args.list:
        for n, (name, _, desc) in enumerate(STEPS, 1):
            print(f"{n}. {name:<13} {desc}")
        return

    industry = load_industry(args.industry)
    print(industry.describe())

    if args.only:
        selected = [s for s in STEPS if s[0] == args.only]
    else:
        start = STEP_NAMES.index(args.start) if args.start else 0
        selected = STEPS[start:]

    for name, module, desc in selected:
        print(f"\n=== [{name}] {desc} ===")
        started = time.time()
        try:
            importlib.import_module(module).main(industry)
        except Exception:
            print(f"\n[{name}] 단계에서 실패했습니다. 원인을 고친 뒤 "
                  f"`--industry {industry.name} --from {name}` 으로 이어서 실행하세요.")
            raise
        print(f"({time.time() - started:.0f}초)")


if __name__ == "__main__":
    main()
