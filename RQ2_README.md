# RQ2 — 과거 예상 상회 이력(HBR)과 실적 발표 반응

설계 문서: 노션 「RQ2 구체화: 과거 예상 상회 이력과 시장 반응」, `plans/2026-11-19-rq2-analysis.md`

## 구조

공통 파이프라인의 `output/{it,insurance,bio}/events.csv`를 읽는 **후속 분석**입니다 (RQ3와 같은 방식). SUE·CAR[0,+1]·HBR(`beat_rate_8q`)·메인 표본(`in_main_sample`)은 다시 계산하지 않고 그대로 씁니다.

공통 열에 없는 변수만 `src/did_us/rq2.py`에서 만듭니다.

| 변수 | 정의 |
| --- | --- |
| 일치(Meet) | 실제 EPS = 컨센서스. 메인에서 제외 (공통 코드는 하회로 셈) |
| `beat_streak` | 직전 발표부터 거꾸로 센 연속 상회 횟수 (최대 8). HBR과 같은 과거 발표 집합 사용 |
| `pre_car` | CAR[-5,-1], SPY 기준 |
| `log_dvol` | t−60~t−21 평균 거래대금(종가×거래량)의 로그. 시가총액 대신 쓰는 규모 대리변수 |

실행 시 두 가지를 자동 검증합니다: ① 재계산한 직전 8개 상회 횟수 = 공통 `beat_cnt_8q` ② 주가 캐시로 재계산한 CAR[0,+1] = 공통 `car01`.

## 실행

```bash
python rq2_pipeline.py                      # data/raw/us/prices.csv 캐시 필요 (공통 파이프라인 prepare가 만듦)
python rq2_pipeline.py --no-price-controls  # 캐시가 없을 때: pre_car·log_dvol 없이 추정
```

## 모형

```
CAR ~ Beat + S_beat + S_miss + HBR×Beat + HBR×Miss + S_beat×HBR + S_miss×HBR
      + hist_sd + pre_car + log_dvol | 산업×분기 FE,  기업 클러스터 표준오차
```

- `S_beat = 서프라이즈 × Beat`, `S_miss = 서프라이즈 × Miss` (원값 SUE이면 SUE⁺, SUE⁻와 같음)
- HBR은 회귀 표본 평균으로 중심화. 범위 0~1이므로 계수는 "HBR 0 → 1"일 때의 차이 (1회 더 상회 = 계수 ÷ 8)

| 항 | 가설 |
| --- | --- |
| `HBR_beat` | H2-A1 상회 시 수준효과 (예상 −) |
| `Sb_HBR` | H2-A2 상회 시 기울기효과 (예상 −) |
| `HBR_miss` | H2-B 하회 시 수준효과 (예상 −) |
| `Sm_HBR` | H2-B 보조: 하회 시 기울기효과 |

| 모형 | 내용 |
| --- | --- |
| `baseline` | HBR 미포함 기준 모형 (원값 SUE 기준 RQ1) |
| `main` | 연속 HBR |
| `rob_zpast` | 서프라이즈를 z_past로 |
| `rob_bins` | HBR 구간형 0~6회(기준) / 7회 / 8회 |
| `rob_firmfe` | 기업 FE 추가 |
| `rob_meet_as_miss` | 일치를 하회에 포함 |
| `rob_streak` | HBR 대신 StreakNorm(= beat_streak ÷ 8) |
| `rob_hbr_streak` | HBR과 StreakNorm 동시 투입 (H2-C) |

## 결과물 (`output/rq2/`)

- `key_results.csv`: 모형 × 가설별 핵심 계수
- `regression_results.csv`: 모든 계수, `models.txt`: 모형별 요약
- `cell_counts.csv`: 산업 × HBR 구간 × 상회/일치/하회 이벤트 수
- `sc_candidates.csv`: 8분기 연속 상회 후 하회 이벤트 (Synthetic Control 사례 후보)
- `rq2_sample.csv`: 회귀 표본

## 해석상 주의

- 관측 자료의 회귀 관계입니다. HBR은 무작위로 배정되지 않으므로 HBR의 조절효과를 인과효과로 단정할 수 없습니다.
- `rob_zpast`는 서프라이즈 0의 의미가 "컨센서스 부합"이 아니라 "그 회사 평소 수준"이라, 수준효과 계수를 메인과 직접 비교할 수 없습니다.
- 8분기 연속 상회 후 하회 이벤트는 24건뿐이라 하회 쪽 계수는 불안정합니다.
- macOS에서 pyfixest 실행 시 나오는 `encountered in matmul` 경고는 numpy 2.0 + Accelerate의 가짜 경고로, 일반 최소제곱 재계산과 계수가 일치함을 확인하고 숨겼습니다.
