# 파이프라인 재현 가이드

보험·바이오·IT 세 산업의 실적발표 DiD 분석을 처음부터 다시 돌리는 방법입니다. 세 산업이 같은 코드와 같은 정의를 쓰고, 산업별로 다른 것은 `config/did/`의 CSV뿐입니다.

## 1. 준비

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

- Python 3.9라면 `pyfixest==0.25.4`로 설치한다. 그 이후 버전은 3.10 이상이 필요하다.
- API 키는 필요 없다. 데이터는 yfinance에서 받는다.
- Colab: 저장소를 클론하고 `pip install yfinance pyfixest`를 실행한 뒤, 노트북 첫 셀의 `ROOT`를 저장소 경로로 바꾼다.

## 2. 실행

[notebook/did_common.ipynb](../../notebook/did_common.ipynb)를 위에서부터 실행한다. 노트북 없이 돌리려면 프로젝트 루트에서 아래를 실행한다.

```python
from src.did_us.run import prepare, run_industry, combine
from src.did_us.settings import INDUSTRIES

ctx = prepare()                                          # 세 산업 데이터 + 이벤트 테이블
results = {i: run_industry(ctx, i) for i in INDUSTRIES}  # 산업당 1분 안팎
combine(results)                                         # 산업 비교표
```

- 처음 실행하면 yfinance에서 실적·주가를 받아 `data/raw/us/`에 저장한다. 이 폴더는 저장소에 올리지 않는다. 그다음부터는 저장된 파일을 읽는다.
- 새로 받으려면 `prepare(refresh=True)`(노트북은 `REFRESH = True`)로 실행한다. yfinance 값은 받는 시점에 따라 바뀔 수 있다(컨센서스 수정 등). 그래서 다시 받으면 결과가 조금 달라질 수 있다.

## 3. 설정 바꾸기

분석 기준값은 [src/did_us/settings.py](../../src/did_us/settings.py)의 `Params`에 있다. 기본값은 회의에서 확정한 통일안이다. 바꾸고 싶은 값만 넘긴다.

```python
P = Params(streak_hi=6, z_cut=1.0)
ctx = prepare(P)
```

| 값 | 기본 | 의미 |
|---|---|---|
| `event_start`, `event_end` | 2022-01-01 ~ 2026-09-30 | 분석 이벤트 기간 (발표일 기준) |
| `ctrl_h` | 10 | 대조군: 처치 반응일 ±10거래일 안에 자기 발표가 없는 회사 |
| `hist_q`, `hist_min` | 8, 4 | 회사 기준 점수: 직전 8분기 평균·SD. 4분기 미만이면 계산하지 않음 |
| `z_cut` | 0.5 | 커짐 / 예상 / 작아짐 기준 (±0.5SD) |
| `streak_hi`, `streak_lo` | 7, 4 | 연속 상회 / 가끔 상회 기준 (회의 확정 전 임시값) |

## 4. 산업별 보정 파일 (`config/did/`)

| 파일 | 내용 |
|---|---|
| `universe.csv` | 표본 기업과 하위그룹 (보험 23 · 바이오 17 · IT 30) |
| `eps_overrides.csv` | yfinance EPS에 GAAP 값이 섞인 이벤트를 SEC 보도자료의 조정 EPS로 교체 |
| `t0_overrides.csv` | yfinance 발표일·시각이 틀린 이벤트의 반응일 교정 |
| `exclude_events.csv` | 실적과 별개의 뉴스가 겹친 교란 이벤트 (메인 제외 / 강건성에서만 제외) |

모든 파일에서 `#`으로 시작하는 줄은 주석이다. 각 행의 근거(SEC 접수 시각, 뉴스 등)는 `note`, `evidence`, `reason` 컬럼에 적는다.

## 5. 결과물 (`output/`)

| 경로 | 내용 |
|---|---|
| `<산업>/events.csv` | 발표 1건 = 1행. 반응일·SUE·회사 기준 점수·3그룹·CAR. CAR·SC·Causal ML의 출발점 |
| `<산업>/did_results.csv` | 발표 여부 스택 DiD (\|AR\|·비정상 거래량) |
| `<산업>/sue_results.csv` | 연속처치·3그룹·과거 기록 분석 (CAR) |
| `<산업>/robustness.csv`, `placebo.csv` | 강건성·플라시보 |
| `<산업>/control_balance.csv`, `covid_share.csv` | 대조군 부족 점검, 코로나 시기 비중 점검 |
| `<산업>/streak_cells.csv`, `first_miss.csv` | 과거 기록별 평균 CAR, 연속 상회 후 첫 미달 |
| `<산업>/figures/` | 이벤트스터디 그림 |
| `<산업>/team_summary.csv` | 산업별 대표 숫자 요약 (표본·발표 여부·연속처치·3그룹·과거 기록) |
| `combined/sector_comparison.csv` | 산업 비교 대표 숫자 |

설계 결정과 그 근거는 [memory/pipeline-unification-decisions.md](../../memory/pipeline-unification-decisions.md)에 정리돼 있다.
