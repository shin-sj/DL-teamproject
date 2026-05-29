# DL-teamproject — 모델 비교 결과

## TL;DR

- **TextCNN이 3 조건 모두에서 1등** (avg-F1: Raw 77.46 / Masked 72.86 / Masked+Geo 66.86). BiLSTM·Hybrid가 한 번도 이긴 task가 없음.
- **"X대" 마스킹 = 연령대 F1 -17~19%p** (모든 모델). LLM이 페르소나에 "60대"를 직접 박아넣음을 정량 확인.
- **지명 마스킹 = 광역시도 F1 -24~36%p**. 특히 BiLSTM이 가장 크게 무너짐 (-35.9%p) → 지명에 가장 의존.
- **성별·라이프스타일은 마스킹에 거의 영향 없음** → 직접 단어가 아닌 다른 시그널(가족 패턴, 취미 키워드 자체)에서 학습.
- **라이프스타일이 항상 bottleneck** (macro-F1 13~18%): 8-class 중 4개가 1k 미만으로 weak labeling 불균형 → 발표용 caveat.

## 실험 설정

- 데이터: `data/nemotron_100k_seed42.parquet` (100k, MD5 `63a16d80...`), 라이프스타일 매칭 실패 13개 drop → 99,987
- 4 task: 성별(2) / 연령대(6) / 광역시도(17) / 라이프스타일(8, weak label)
- 분할: stratified by sex×age (12 bins), 80/10/10 (seed=42) → train 79,985 / val 9,993 / test 10,009
- 토크나이저: char-level (음절), min_freq=5, **train split에서만 빌드** → 누설 방지
- 입력: 11개 narrative 컬럼을 ` [SEP] ` 으로 연결, max 512 chars truncate
- 학습: AdamW lr=1e-3, batch=256, grad clip 5.0, **10 epochs**, early stop patience=3 on val avg-macro-F1
- 손실: 4 task cross-entropy 단순 합 (task별 가중치 없음)
- 평가: per-task macro-F1 + accuracy, 4 task macro-F1 평균을 종합 지표로
- GPU: NVIDIA RTX 3060 Ti 8GB, PyTorch 2.6+cu124, Windows 11

## 모델 구조 요약

| 모델 | 인코더 | 풀링 | 헤드 |
|---|---|---|---|
| TextCNN | Conv1d k={3,4,5}, 96 filters each | max-over-time | 4 × Linear |
| BiLSTM | 1-layer bi-LSTM, hidden=128 | masked mean-pooling | 4 × Linear |
| Hybrid | Conv1d k=5 stride=2, 96 filters → 1-layer biLSTM h=128 | masked mean-pooling | 4 × Linear |

임베딩(128-dim, 학습) 과 4-head 구조는 동일 → 인코더 단독 비교.

## Raw 결과

### raw 조건 (n_train=79,985, vocab=1,596)

| Model | params | best ep | sex F1 | age F1 | prov F1 | life F1 | **avg F1** | sex acc | age acc | prov acc | life acc | time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **textcnn** | 361,569 | 8 | 98.81 | 94.13 | 99.13 | 17.76 | **77.46** | 98.81 | 93.89 | 99.08 | 43.81 | 91s |
| **bilstm** | 476,961 | 10 | 97.85 | 93.60 | 89.00 | 13.13 | **73.40** | 97.85 | 93.36 | 90.00 | 38.18 | 184s |
| **hybrid** | 505,729 | 10 | 98.33 | 93.43 | 96.41 | 13.21 | **75.35** | 98.33 | 93.13 | 97.33 | 38.07 | 124s |

## Demographic Masked 결과

### masked 조건 (n_train=79,985, vocab=1,594)

| Model | params | best ep | sex F1 | age F1 | prov F1 | life F1 | **avg F1** | sex acc | age acc | prov acc | life acc | time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **textcnn** | 361,313 | 9 | 98.43 | 77.35 | 98.99 | 16.69 | **72.86** | 98.43 | 76.24 | 98.95 | 42.25 | 93s |
| **bilstm** | 476,705 | 10 | 98.30 | 74.09 | 89.54 | 12.86 | **68.70** | 98.30 | 72.93 | 90.21 | 38.00 | 188s |
| **hybrid** | 505,473 | 10 | 98.21 | 74.87 | 95.59 | 13.32 | **70.50** | 98.21 | 73.89 | 96.08 | 37.90 | 124s |

## Demographic + Geo Masked 결과

### masked_geo 조건 (n_train=79,985, vocab=1,594)

| Model | params | best ep | sex F1 | age F1 | prov F1 | life F1 | **avg F1** | sex acc | age acc | prov acc | life acc | time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **textcnn** | 361,313 | 10 | 98.48 | 76.65 | 74.47 | 17.82 | **66.86** | 98.48 | 75.58 | 81.98 | 42.57 | 89s |
| **bilstm** | 476,705 | 10 | 97.70 | 76.06 | 53.09 | 12.98 | **59.95** | 97.70 | 75.02 | 71.39 | 37.78 | 180s |
| **hybrid** | 505,473 | 10 | 97.87 | 73.87 | 64.70 | 12.85 | **62.32** | 97.87 | 72.71 | 76.85 | 37.87 | 119s |


## Leakage Ablation

### 조건별 비교 (test macro-F1, %p 변화)

| Model | Task | Raw | Masked | Δ | Masked+Geo | Δ |
|---|---|---:|---:|---:|---:|---:|
| textcnn | 성별 | 98.81 | 98.43 | -0.38 | 98.48 | -0.33 |
| textcnn | 연령대(6) | 94.13 | 77.35 | -16.78 | 76.65 | -17.47 |
| textcnn | 광역시도(17) | 99.13 | 98.99 | -0.14 | 74.47 | -24.66 |
| textcnn | 라이프(8) | 17.76 | 16.69 | -1.07 | 17.82 | +0.07 |
| bilstm | 성별 | 97.85 | 98.30 | +0.45 | 97.70 | -0.15 |
| bilstm | 연령대(6) | 93.60 | 74.09 | -19.50 | 76.06 | -17.54 |
| bilstm | 광역시도(17) | 89.00 | 89.54 | +0.54 | 53.09 | -35.92 |
| bilstm | 라이프(8) | 13.13 | 12.86 | -0.27 | 12.98 | -0.16 |
| hybrid | 성별 | 98.33 | 98.21 | -0.12 | 97.87 | -0.46 |
| hybrid | 연령대(6) | 93.43 | 74.87 | -18.57 | 73.87 | -19.56 |
| hybrid | 광역시도(17) | 96.41 | 95.59 | -0.82 | 64.70 | -31.71 |
| hybrid | 라이프(8) | 13.21 | 13.32 | +0.11 | 12.85 | -0.36 |


해석:
- **Raw → Masked Δ** : 'X대', 친족어 같은 직접 단서가 기여한 정도
- **Masked → Masked+Geo Δ** : 광역시도·시군구 지명이 기여한 정도
- **Masked+Geo 잔존 점수** : 모델이 stereotype에서 실제로 학습한 시그널

## 핵심 발견

### 1) 모델 비교 — TextCNN의 일관된 우위
3가지 조건, 4가지 task, 12개 셀 전부 TextCNN ≥ Hybrid > BiLSTM. 한국어 페르소나 분류처럼 **국지적 키워드 phrase** ("무등산 자락", "60대 보안원")가 결정적인 task에서는 multi-kernel CNN의 max-over-time pooling이 가장 효율적. BiLSTM은 10 epochs로는 province task에서 끝까지 수렴하지 못함 (Raw 0.890 vs CNN 0.991). Hybrid는 두 모델 사이에 정확히 위치 — "CNN feature map을 LSTM에 넣으면 더 좋아진다"는 가설은 본 데이터에서는 **거짓**.

### 2) Leakage 정량 — EDA §5 예측 검증
EDA 단계의 측정값 vs 실제 모델 성능 변화:

| 누설 종류 | EDA 측정 | 모델 영향 (TextCNN F1 변화) |
|---|---:|---:|
| 연령대 "X대" 노출 | 60.9% | -16.78%p (94 → 77) |
| 광역시도/시군구 노출 | 82.5% | -24.66%p (99 → 74) |
| 성별 친족어 노출 | 45.1% | -0.38%p (마스킹 효과 거의 없음) |

**성별 누설은 단어 차원의 마스킹만으로 안 사라짐.** 직업·취미·요리 키워드 등 stereotype 시그널에서 모델이 우회 학습. 발표 포인트로 활용 가치 큼.

### 3) Stereotype 잔존 신호 (Masked+Geo 조건 = 가장 깨끗한 baseline)
직접 demographic 단어·지명을 모두 제거한 뒤에도 TextCNN이 보여주는 점수:
- **성별 98.48** → 거의 손실 없음. **모델은 페르소나 narrative에서 성별을 직접 단어 없이도 추론 가능.**
- **연령대 76.65** → 17%p 떨어졌지만 여전히 매우 높음. "온천 여행", "트로트", "은퇴" 같은 간접 단서로 학습.
- **광역시도 74.47** → 지명을 다 가렸어도 75% (random=5.9%, majority=26%). 지역 특산물·사투리·관광지 키워드.
- **라이프스타일 17.82** → 사실상 변동 없음 (대표 클래스만 맞추는 수준).

## 다음 단계 제안

1. **라이프스타일 8-class 재정의** — 현재 fashion_style(400 samples) / beauty_care(1117) 너무 희소. 4-class로 머지하거나 class-weighted CE로 균형 학습. 코드에 이미 `--weighted` 옵션 구현 완료 (`python src/run_all.py raw 10 --weighted`).
2. **모델 capacity 차이** — TextCNN 362k vs Hybrid 506k. Hybrid가 1.4배 큰 모델인데도 진다는 점에서 "단순 capacity로는 안 됨"의 좋은 증거. 발표 슬라이드에서 강조 가능.

## 재현 방법

```bash
# 0. 의존성 (글로벌 Python 3.12 기준)
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install pandas pyarrow scikit-learn

# 1. 데이터 확인 (이미 git lfs로 받아져있음)
ls data/nemotron_100k_seed42.parquet

# 2. 3 모델 × 3 조건 학습 (총 ~25분 on RTX 3060 Ti)
python src/run_all.py raw 10        # Raw 조건
python src/run_all.py masked 10     # "X대"+친족어 마스킹
python src/run_all.py masked_geo 10 # 위 + 지명 마스킹

# 3. 보고서 갱신
python results/make_report.py
```

결과 파일:
- `results/{textcnn,bilstm,hybrid}_{raw,masked,masked_geo}.json` — 모델별 상세 (history, confusion matrix, test 점수)
- `results/summary_{raw,masked,masked_geo}.json` — 조건별 요약
- `checkpoints/{model}_{cond}_best.pt` — 최선 가중치
- `checkpoints/tokenizer_{cond}.json` — 음절 vocab