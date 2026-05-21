"""
results/*.json + summary_*.json 을 모아 RESULTS.md 생성.
모드별 표 + 모델 비교 + (있다면) Raw vs Masked 차이 분석.
"""
from __future__ import annotations
import json
import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

MODELS = ["textcnn", "bilstm", "hybrid"]
TASKS = ["sex", "age", "province", "lifestyle"]
TASK_KO = {"sex": "성별", "age": "연령대(6)", "province": "광역시도(17)", "lifestyle": "라이프(8)"}


def load_summary(mode: str) -> dict | None:
    p = RESULTS_DIR / f"summary_{mode}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def fmt_pct(x: float) -> str:
    return f"{x*100:.2f}"


def render_table_for_mode(summary: dict) -> str:
    rows = []
    rows.append(f"### {summary['mode']} 조건 (n_train={summary['n_train']:,}, vocab={summary['vocab_size']:,})")
    rows.append("")
    rows.append("| Model | params | best ep | sex F1 | age F1 | prov F1 | life F1 | **avg F1** | sex acc | age acc | prov acc | life acc | time |")
    rows.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m in MODELS:
        if m not in summary["results"]:
            continue
        r = summary["results"][m]
        t = r["test_per_task"]
        row = (
            f"| **{m}** | {r['n_params']:,} | {r['best_epoch']} "
            f"| {fmt_pct(t['sex']['macro_f1'])} "
            f"| {fmt_pct(t['age']['macro_f1'])} "
            f"| {fmt_pct(t['province']['macro_f1'])} "
            f"| {fmt_pct(t['lifestyle']['macro_f1'])} "
            f"| **{fmt_pct(r['test_macro_f1_avg'])}** "
            f"| {fmt_pct(t['sex']['acc'])} "
            f"| {fmt_pct(t['age']['acc'])} "
            f"| {fmt_pct(t['province']['acc'])} "
            f"| {fmt_pct(t['lifestyle']['acc'])} "
            f"| {r['total_time_s']:.0f}s |"
        )
        rows.append(row)
    return "\n".join(rows)


def render_ablation(summaries: dict[str, dict]) -> str:
    if len(summaries) < 2:
        return ""
    rows = ["### 조건별 비교 (test macro-F1, %p 변화)\n"]
    rows.append("| Model | Task | Raw | Masked | Δ | Masked+Geo | Δ |")
    rows.append("|---|---|---:|---:|---:|---:|---:|")
    for m in MODELS:
        for task in TASKS:
            cells = []
            for mode in ["raw", "masked", "masked_geo"]:
                s = summaries.get(mode)
                if s and m in s["results"]:
                    v = s["results"][m]["test_per_task"][task]["macro_f1"]
                    cells.append(v)
                else:
                    cells.append(None)
            raw_v = cells[0]
            row = f"| {m} | {TASK_KO[task]} |"
            row += f" {fmt_pct(raw_v) if raw_v is not None else '—'} |"
            for v in cells[1:]:
                if v is None or raw_v is None:
                    row += " — | — |"
                else:
                    delta = (v - raw_v) * 100
                    row += f" {fmt_pct(v)} | {delta:+.2f} |"
            rows.append(row)
    return "\n".join(rows)


def main():
    summaries = {}
    for mode in ["raw", "masked", "masked_geo"]:
        s = load_summary(mode)
        if s:
            summaries[mode] = s

    if not summaries:
        print("no results yet")
        return

    out = []
    out.append("# DL-teamproject — 모델 비교 결과\n")
    out.append("- 데이터: `data/nemotron_100k_seed42.parquet` (100k, MD5 `63a16d80...`)")
    out.append("- 4 task: 성별(2) / 연령대(6) / 광역시도(17) / 라이프스타일(8, weak label)")
    out.append("- 분할: stratified by sex×age, 80/10/10 (seed=42)")
    out.append("- 토크나이저: char-level (음절), min_freq=5, train split에서만 빌드 → 누설 방지")
    out.append("- 입력: 11개 narrative 컬럼을 ` [SEP] ` 으로 연결, 길이 512 chars")
    out.append("- 학습: AdamW lr=1e-3, batch=256, grad clip 5.0, early stop patience=3 on val avg-macro-F1")
    out.append("- 손실: 4 task cross-entropy 단순 합")
    out.append("- GPU: NVIDIA RTX 3060 Ti, PyTorch 2.6+cu124\n")

    out.append("## 모델 구조 요약\n")
    out.append("| 모델 | 인코더 | 풀링 | 헤드 |")
    out.append("|---|---|---|---|")
    out.append("| TextCNN | Conv1d k={3,4,5}, 96 filters each | max-over-time | 4 × Linear |")
    out.append("| BiLSTM | 1-layer bi-LSTM, hidden=128 | masked mean-pooling | 4 × Linear |")
    out.append("| Hybrid | Conv1d k=5 stride=2, 96 filters → 1-layer biLSTM h=128 | masked mean-pooling | 4 × Linear |")
    out.append("\n임베딩(128-dim, 학습) 과 4-head 구조는 동일 → 인코더 단독 비교.\n")

    for mode in ["raw", "masked", "masked_geo"]:
        if mode in summaries:
            out.append("## " + {"raw": "Raw", "masked": "Demographic Masked",
                                "masked_geo": "Demographic + Geo Masked"}[mode] + " 결과\n")
            out.append(render_table_for_mode(summaries[mode]))
            out.append("")

    if len(summaries) >= 2:
        out.append("\n## Leakage Ablation\n")
        out.append(render_ablation(summaries))
        out.append("")
        out.append("\n해석:")
        out.append("- **Raw → Masked Δ** : 'X대', 친족어 같은 직접 단서가 기여한 정도")
        out.append("- **Masked → Masked+Geo Δ** : 광역시도·시군구 지명이 기여한 정도")
        out.append("- **Masked+Geo 잔존 점수** : 모델이 stereotype에서 실제로 학습한 시그널")

    md = "\n".join(out)
    (RESULTS_DIR / "RESULTS.md").write_text(md, encoding="utf-8")
    print("wrote: results/RESULTS.md")
    print("\n" + md)


if __name__ == "__main__":
    main()
