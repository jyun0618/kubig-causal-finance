"""RQ2: 공통 파이프라인의 events.csv로 HBR × 서프라이즈 회귀를 추정한다.

Usage: python rq2_pipeline.py
       python rq2_pipeline.py --no-price-controls   # data/raw/us/prices.csv 캐시가 없을 때
"""
import argparse

import pandas as pd

from src.did_us.rq2 import run_rq2


def main():
    p = argparse.ArgumentParser(description="RQ2 on shared IT/insurance/bio events")
    p.add_argument("--no-price-controls", action="store_true", help="pre_car·log_dvol 통제 없이 실행 (주가 캐시 불필요)")
    a = p.parse_args()
    key, cells, sc = run_rq2(price_controls=not a.no_price_controls)
    pd.set_option("display.width", 200)
    print("셀별 이벤트 수 (산업 × HBR 구간 × 상회/일치/하회):\n", cells.to_string(index=False))
    cols = ["model", "moderator", "hypothesis", "coef", "se", "p", "n_events"]
    print("\n가설별 핵심 계수:\n", key[cols].round(3).to_string(index=False))
    print(f"\nSynthetic Control 후보 (8분기 연속 상회 후 하회): {len(sc)}건")
    print("\nSaved to output/rq2/")


if __name__ == "__main__":
    main()
