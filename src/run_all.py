"""
3개 모델(TextCNN / BiLSTM / Hybrid)을 동일 split·동일 하이퍼파라미터로 학습 후
results/ 에 JSON 저장. 모드(raw / masked / masked_geo)는 인자로 선택.

사용:
  python src/run_all.py                # raw 조건 3 모델
  python src/run_all.py masked         # demographic 단어 마스킹
  python src/run_all.py masked_geo     # 지명까지 마스킹
"""
from __future__ import annotations
import sys
import io
import json
from pathlib import Path

import torch

# UTF-8 stdout (Windows)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import prepare, make_loaders   # noqa: E402
from train import train_one_model, save_result  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
CKPT_DIR = ROOT / "checkpoints"


def _compute_class_weights(y, num_classes: int, mode: str = "inverse"):
    """클래스 빈도 역수 기반 가중치. mode='inverse' or 'sqrt'."""
    import numpy as np
    counts = np.bincount(y, minlength=num_classes).astype(float)
    counts[counts == 0] = 1.0
    if mode == "sqrt":
        w = 1.0 / np.sqrt(counts)
    else:
        w = 1.0 / counts
    # 평균 1로 normalize
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def main(mode: str = "raw", models=("textcnn", "bilstm", "hybrid"),
         epochs: int = 8, batch_size: int = 256, max_len: int = 512,
         weighted_lifestyle: bool = False):
    print(f"==== run_all  mode={mode}  epochs={epochs}  bs={batch_size}  max_len={max_len} ====")
    print("preparing data...")
    prep = prepare(mode=mode, max_len=max_len, seed=42)
    print(f"  n_total={prep['n_total']:,}  vocab={prep['tokenizer'].vocab_size:,}")
    for k, v in prep["splits"].items():
        print(f"  {k}: {v['n']:,}")
    loaders = make_loaders(prep, batch_size=batch_size)
    vocab_size = prep["tokenizer"].vocab_size

    # 토크나이저 저장 (각 모드별)
    tok_path = CKPT_DIR / f"tokenizer_{mode}.json"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    prep["tokenizer"].save(tok_path)

    # 라이프스타일 클래스 가중치 (옵션)
    class_weights = None
    label_suffix = mode
    if weighted_lifestyle:
        y_life_train = prep["splits"]["train"]["labels"]["lifestyle"]
        from data import NUM_CLASSES as NC
        w = _compute_class_weights(y_life_train, NC["lifestyle"], mode="sqrt")
        class_weights = {"lifestyle": w}
        label_suffix = mode + "_wl"
        print(f"  lifestyle class weights (sqrt-inverse): {w.tolist()}")

    all_results = {}
    for name in models:
        print(f"\n---- training {name} ({label_suffix}) ----")
        result = train_one_model(
            model_name=name,
            loaders=loaders,
            vocab_size=vocab_size,
            out_dir=CKPT_DIR,
            epochs=epochs,
            label=label_suffix,
            class_weights=class_weights,
        )
        save_result(result, RESULTS_DIR)
        all_results[name] = result

    # 요약 표
    print("\n==== summary ====")
    header = f"{'model':10s} {'sex':>10s} {'age':>10s} {'province':>10s} {'lifestyle':>10s} {'avgF1':>8s} {'time':>8s}"
    print(header)
    for name, r in all_results.items():
        t = r["test"]["per_task"]
        row = (
            f"{name:10s} "
            f"{t['sex']['macro_f1']:.4f}     "
            f"{t['age']['macro_f1']:.4f}     "
            f"{t['province']['macro_f1']:.4f}     "
            f"{t['lifestyle']['macro_f1']:.4f}    "
            f"{r['test']['macro_f1_avg']:.4f}   "
            f"{r['total_time_s']:.0f}s"
        )
        print(row)

    # 종합 요약 저장
    summary = {
        "mode": label_suffix,
        "epochs": epochs,
        "vocab_size": vocab_size,
        "weighted_lifestyle": weighted_lifestyle,
        "n_train": prep["splits"]["train"]["n"],
        "n_val": prep["splits"]["val"]["n"],
        "n_test": prep["splits"]["test"]["n"],
        "results": {
            name: {
                "n_params": r["n_params"],
                "best_epoch": r["best_epoch"],
                "test_per_task": r["test"]["per_task"],
                "test_macro_f1_avg": r["test"]["macro_f1_avg"],
                "total_time_s": r["total_time_s"],
            }
            for name, r in all_results.items()
        },
    }
    (RESULTS_DIR / f"summary_{label_suffix}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nsaved: results/summary_{label_suffix}.json")


if __name__ == "__main__":
    # 인자: mode [epochs] [--weighted]
    mode = "raw"
    epochs = 8
    weighted = False
    for a in sys.argv[1:]:
        if a == "--weighted":
            weighted = True
        elif a.isdigit():
            epochs = int(a)
        else:
            mode = a
    main(mode=mode, epochs=epochs, weighted_lifestyle=weighted)
