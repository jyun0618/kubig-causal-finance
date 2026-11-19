"""Run RQ3 on the shared RQ2/DiD pipeline's saved events.csv files.

Usage: python rq3_pipeline.py
       python rq3_pipeline.py --include-nonpositive --allow-short-history
       python rq3_pipeline.py --score z_past
"""
import argparse
from src.did_us.rq3 import run_rq3


def main():
    p=argparse.ArgumentParser(description='RQ3 on shared IT/insurance/bio events')
    p.add_argument('--include-nonpositive',action='store_true',help='Use all SUE signs')
    p.add_argument('--allow-short-history',action='store_true',help='Use >=4 instead of >=8 prior records (shared pipeline minimum)')
    p.add_argument('--score',choices=['sue','z_past'],default='sue')
    a=p.parse_args()
    results,summary=run_rq3(positive_only=not a.include_nonpositive,
                            full_eight=not a.allow_short_history,score=a.score)
    key=results[(results.model=='rq3_main')&(results.term=='SUE_c:HBR_c:Stability_c')]
    print('RQ3 sample by industry:\n',summary.to_string(index=False))
    print('\nTriple interaction:\n',key[['coef','se','p','ci_low','ci_high','n_events','n_firms']].to_string(index=False))
    print('\nSaved to output/rq3/')

if __name__=='__main__':main()
