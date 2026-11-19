# RQ3 — 팀 공통 이벤트 파이프라인 연결

## 핵심 변경

기존 독립형 RQ3 스크립트에서 **공통 파이프라인의 `output/{it,insurance,bio}/events.csv`를 읽는 후속 분석**으로 변경했습니다. `src/did_us/events.py`가 계산한 SUE, HBR(`beat_rate_8q`), 과거 SUE 표준편차(`hist_sd`), SPY 기준 CAR[0,+1](`car01`) 및 `in_main_sample`을 그대로 사용합니다. SEC 보정·제외 이벤트·반응일 처리도 이 공통 이벤트 파일에 반영된 결과를 사용합니다.

## 실행

저장소 루트에서:

```bash
pip install -r requirements.txt
pip install statsmodels
python rq3_pipeline.py
```

현재 ZIP에는 공통 파이프라인 실행 결과인 세 산업 `events.csv`가 포함되어 있어 바로 RQ3를 재현할 수 있습니다. 공통 이벤트 파일을 갱신했다면 **먼저 팀 공통 파이프라인을 실행**하고, 이어서 `python rq3_pipeline.py`를 실행하세요. RQ3는 네트워크나 추가 수집 없이 저장된 이벤트 CSV를 이용합니다.

## 분석 정의

- 메인 표본: 공통 `in_main_sample=True` 이벤트 중 직전 SUE와 beat 기록이 **각각 8개 이상**, 현재 **원값 SUE > 0**인 이벤트. 직전 8개는 공통 코드가 계산한 과거 유효 발표 8개이며, 연속된 회계분기 8개인지 검증하지는 않습니다.
- `SUE`: 공통 원값 SUE (%), `HBR`: 직전 8개 예상 상회 비율, `VolSUE`: 직전 SUE 표준편차.
- `Stability`: `-(VolSUE - 표본평균)/표본표준편차` (높을수록 안정적).
- `CAR`: SPY 초과수익률의 CAR[0,+1] (%p).
- 각 연속변수는 회귀표본에서 평균 중심화.
- 메인 회귀: `CAR ~ SUE_c * HBR_c * Stability_c + 산업 FE + 반응분기 FE`, 회사(ticker) 클러스터 표준오차.
- 같은 표본의 RQ2 비교 회귀도 추정하지만, **팀 RQ2의 원래 이벤트스터디/DiD 모형과 동일한 모형은 아닙니다.**
- 산업별 이질성: 산업과 삼중 상호작용의 4중 상호작용을 추가한 탐색적 회귀.

## 결과물

`output/rq3/`에 `rq3_regression_sample.csv`, `regression_results.csv`, `sector_comparison.csv`, `sample_audit.csv`, `rq2_same_sample.txt`, `rq3_main.txt`, `rq3_industry_heterogeneity.txt` 저장.

## 추가 선택 사항

```bash
python rq3_pipeline.py --score z_past
python rq3_pipeline.py --include-nonpositive
python rq3_pipeline.py --allow-short-history
```

`--score z_past`는 공통 회사기준 표준화 점수로 SUE를 대체합니다. 이 경우 '긍정적'은 **평소보다 큰 서프라이즈(z_past>0)** 를 의미하며, 실제 EPS가 컨센서스를 상회했는지와 다릅니다. `--allow-short-history`는 공통 파이프라인 최소 기준인 4개 기록부터 포함하므로, 직전 8분기 완비 표본과 구별해서 해석해야 합니다.

## 해석상 주의

삼중 상호작용 계수는 HBR에 따른 SUE-CAR 기울기 차이가 Stability에 따라 어떻게 달라지는지 나타냅니다. 양수라면 '안정적일수록 HBR의 약화효과가 더 크다'는 음의 계수 가설과 반대 방향입니다. 관측 자료에 대한 회귀 관계이며, 추가 식별 가정 없이 인과효과로 단정할 수 없습니다. 공통 SUE·HBR·CAR 정의와 산업별 표본은 이전 독립형 RQ3와 달라 **이전 계수(+3.762)를 그대로 비교하거나 재사용하면 안 됩니다.**
