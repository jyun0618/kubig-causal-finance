# 파이프라인 재현 가이드

실적발표 DiD 파일럿을 산업군 하나에 대해 처음부터 끝까지 다시 돌리는 방법입니다. 종목 목록만 바꾸면 다른 산업군에도 같은 순서로 적용할 수 있습니다.

## 1. 준비

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env     # 열어서 DART_API_KEY= 뒤에 본인 키를 넣는다
```

- OpenDART API 키는 https://opendart.fss.or.kr 에서 무료로 발급받는다.
- `.env`는 `.gitignore`에 들어 있어 커밋되지 않는다. 키를 채팅이나 코드에 붙여넣지 않는다.
- 모든 명령은 **프로젝트 루트**에서 실행한다 (`-m` 방식이라 위치가 중요하다).

## 2. 산업군 설정

산업군마다 `config/industries/<이름>.csv` 하나를 둔다. 금융은 [config/industries/finance.csv](../../config/industries/finance.csv)에 있다. 새 산업군은 [_template.csv](../../config/industries/_template.csv)를 복사해서 만든다.

| 컬럼 | 설명 |
|---|---|
| `ticker` | 6자리 종목코드 |
| `name` | 표시용 이름 |
| `sector` | 같은 업종으로 묶는 단위. **매칭과 DiD가 이 값 안에서만** 이루어진다 |
| `market` | `KOSPI` 또는 `KOSDAQ`. 시장모형의 벤치마크 지수를 정한다. 비우면 KOSPI |

설정 때 알아 둘 점:
- 업종(`sector`)당 사건이 있는 기업이 **4곳 미만이면 DiD에서 자동 제외**된다. 6곳 이상을 권장한다.
- 설정 파일의 `#` 줄은 주석이다.
- 오류(종목코드 형식, 중복, 빈 sector 등)는 실행 첫 단계에서 바로 알려 준다.

## 3. 실행

```bash
.venv/bin/python -m src.run_pipeline --industry finance          # 전체
.venv/bin/python -m src.run_pipeline --industry finance --from earnings   # 중간 단계부터
.venv/bin/python -m src.run_pipeline --industry finance --only did        # 한 단계만
.venv/bin/python -m src.run_pipeline --list                      # 단계 목록
```

금융 17개 종목 기준 소요 시간은 캐시가 없을 때 5~10분 안팎이다(대부분 공시 원문과 KIND 응답을 내려받는 시간). 캐시가 있으면 1분 안에 끝난다.

각 단계는 개별로도 실행할 수 있다: `.venv/bin/python -m src.data.collect_calendar --industry finance`

## 4. 단계별 입력과 산출물

산출물은 `data/<산업군>/` 아래에 쌓인다. `data/raw/`는 산업군이 공유하는 캐시이고 git에 올라가지 않는다.

| # | 단계 | 하는 일 | 주요 산출물 |
|---|---|---|---|
| 1 | `calendar` | OpenDART 공시 목록에서 실적발표일(잠정실적 공정공시 접수일)을 모은다 | `calendar/earnings_calendar.csv` |
| 2 | `filing_times` | KIND에서 공시 접수 시각을 받아 장전/장중/마감직전/장후로 분류한다 | `calendar/filing_times.csv` |
| 3 | `earnings` | 공시 원문에서 순이익을 뽑아 SUE(전년 동기 기반 서프라이즈)를 계산한다 | `earnings/sue.csv`, `earnings/parse_coverage.csv` |
| 4 | `prices` | 수정주가, 거래량, 시가총액, 시장 지수를 받는다 | `prices/prices.csv`, `prices/market_index.csv` |
| 5 | `events` | 발표별 사건일(t0)을 정하고 시장모형 초과수익률·비정상 거래량을 계산한다 | `events/events.csv`, `events/event_panel.csv` |
| 6 | `did` | 통제군 매칭, DiD 추정, 평행추세·위약 검정 | `results/did_*.csv` |

### 결과 파일 읽는 법 (`results/did_summary.csv`)

| 컬럼 | 의미 |
|---|---|
| `design` | `A` = 발표 시점 기반 (not-yet-treated 통제), `B_high`/`B_low` = SUE 상위·하위 1/3 vs 중간 1/3 |
| `outcome` | `ret` 수익률(A는 달력일 수익률, B는 시장모형 초과수익률), `abs_ret` 절대값, `abn_vol` 비정상 거래량 |
| `did_pct` | 처치군이 통제군보다 더 변한 정도 (수익률은 %p, 거래량은 log×100) |
| `treat_change_pct`, `ctrl_change_pct` | DiD를 처치·통제 각각의 변화로 분해한 값. 통제군이 같이 움직이면 전이효과를 의심한다 |
| `p_firm`, `p_quarter` | 기업·분기 단위 군집 wild bootstrap p값. **둘 중 큰 쪽을 기준**으로 삼는다 |
| `placebo_pct`, `placebo_p_quarter` | 처치일을 10거래일 앞당긴 위약 검정. 0에 가깝고 p가 크면 통과 |
| `pretrend_pct`, `pretrend_p_quarter` | 사전 추세 검정(발표 전 −5~−3일 대비 −2~−1일의 차이). p가 작으면 평행추세 의심 |

그 외: `did_event_study.csv`는 상대일별 계수, `did_balance.csv`는 매칭 전후 공변량 균형(SMD), `did_events.csv`는 사건 단위 값이다.

## 5. 설계의 핵심 규칙

- **사건일(t0)**: 공시 접수 시각 기준. 장후(15:30 이후)는 다음 거래일이 t0이고 반응일 1일, 그 외는 발표일이 t0이고 반응일 2일(t0, t0+1)이다.
- **SUE**: `(당기 3개월 순이익 − 전년 동기) / 직전 최대 8개 발표 증감의 표준편차`. 애널리스트 컨센서스가 아니라 전년 동기를 기대치로 쓴다.
- **설계 A 통제군**: 같은 업종·같은 분기에서 처치 t0보다 4거래일 이상 뒤에 발표하는 기업 중, 규모·변동성·SUE가 가장 가까운 2곳.
- **설계 B 통제군**: 같은 업종·분기에서 SUE 중간 1/3이면서 t0 차이가 3거래일 이내인 기업 중 규모·변동성이 가장 가까운 2곳.
- 조정 가능한 값은 [src/common.py](../../src/common.py)(수집 기간)와 [src/did/run_did.py](../../src/did/run_did.py)(`K_CONTROLS`, `GAP_DAYS`, `B_WINDOW` 등) 상단의 상수다.

## 6. 알려진 제약

- **12월 결산 법인만 지원한다.** 분기 추정이 12월 결산을 가정한다.
- **공시 원문 파서는 금융업 공시 서식에 맞춰 만들었다.** 다른 산업군의 `영업(잠정)실적` 공시에서도 동작하는지는 **검증하지 않았다.** `earnings` 단계가 끝나면 종목별 순이익 추출률을 출력하고 80% 미만이면 경고한다. 새 산업군은 이 경고를 가장 먼저 확인한다 (원문은 `data/raw/dart_docs/<접수번호>.xml`에 캐시된다).
- **4분기 순이익은 오차가 크다.** 연간 값에서 3분기 누계를 빼서 만든 경우가 많고, 이 경우 `earnings/sue.csv`의 `basis` 컬럼에 `_annual_minus_q3`가 붙는다. 4분기를 뺀 결과로 강건성을 확인하는 것이 좋다.
- **시가총액은 근사치다.** yfinance 종가 × 발행주식수이고 주식 분할이 없다고 가정한다. 규모 매칭에만 쓴다. 한국거래소 데이터 포털이 점검 중일 때 pykrx 시가총액 경로가 비어서 이렇게 대체했다.
- **군집 수가 작다.** 기업이 7~9곳이라 p값이 거칠다. 다중검정도 감안해서 해석한다.
- **한 분기 안의 다른 기업 발표가 서로 영향을 줄 수 있다(SUTVA).** 통제군의 `ctrl_change_pct`로 대략 점검할 수 있지만, 확정적인 검정은 아니다.

## 7. 문제 해결

| 증상 | 확인할 것 |
|---|---|
| `.env에 DART_API_KEY가 없습니다` | `.env` 파일이 프로젝트 루트에 있고 키가 채워졌는지 |
| `DART 고유번호 목록에 없는 종목코드` | 종목코드 오타 또는 상장폐지 종목 |
| 주가 단계에서 `pykrx 수정주가가 비어 있습니다` | 종목코드, 상장 시점(2020-10 이후 상장이면 앞부분이 비는 것은 정상), 한국거래소 점검 여부 |
| `earnings` 단계 경고 | 공시 서식이 달라 파서가 값을 못 읽은 종목. 원문 캐시를 열어 표 구조를 확인 |
| 중간에 실패 | 오류를 고친 뒤 `--from <실패한 단계>`로 이어서 실행 (이전 단계 산출물은 그대로 쓴다) |
| 결과를 처음부터 다시 만들고 싶다 | `data/<산업군>/`를 지우고 다시 실행 (`data/raw/` 캐시는 지우지 않아도 된다) |
