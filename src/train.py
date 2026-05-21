"""
Multi-task 학습 루프 + 4-task evaluation. 3개 모델 어느 것이든 공통으로 사용.
사용 예: from train import train_one_model
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from data import TASK_NAMES, NUM_CLASSES, AGE_CLASSES, PROVINCE_CLASSES, SEX_CLASSES
from models import build_model, count_params


LIFESTYLE_CLASSES = [
    "fitness_sports", "outdoor_diy", "fashion_style", "beauty_care",
    "food_gourmet", "learning_growth", "family_leisure", "digital_it",
]
CLASS_NAMES = {
    "sex": SEX_CLASSES,
    "age": AGE_CLASSES,
    "province": PROVINCE_CLASSES,
    "lifestyle": LIFESTYLE_CLASSES,
}


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict:
    model.eval()
    preds = {k: [] for k in TASK_NAMES}
    trues = {k: [] for k in TASK_NAMES}
    total_loss = 0.0
    n_batches = 0
    crit = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            out = model(x)
            loss = 0.0
            for k in TASK_NAMES:
                y = batch[f"y_{k}"].to(device, non_blocking=True)
                loss = loss + crit(out[k], y)
                preds[k].append(out[k].argmax(dim=1).cpu().numpy())
                trues[k].append(y.cpu().numpy())
            total_loss += float(loss.item())
            n_batches += 1

    metrics: dict = {"loss": total_loss / max(n_batches, 1), "per_task": {}}
    macro_f1s = []
    for k in TASK_NAMES:
        p = np.concatenate(preds[k])
        t = np.concatenate(trues[k])
        acc = float(accuracy_score(t, p))
        f1 = float(f1_score(t, p, average="macro", zero_division=0))
        metrics["per_task"][k] = {"acc": acc, "macro_f1": f1}
        macro_f1s.append(f1)
    metrics["macro_f1_avg"] = float(np.mean(macro_f1s))
    return metrics


def train_one_model(
    model_name: str,
    loaders: dict[str, DataLoader],
    vocab_size: int,
    out_dir: Path,
    *,
    epochs: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 3,
    device: str | None = None,
    seed: int = 42,
    log_every: int = 50,
    label: str = "",  # 결과 식별용 suffix (예: "raw", "masked")
    class_weights: dict[str, torch.Tensor] | None = None,
) -> dict:
    torch.manual_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(model_name, vocab_size, NUM_CLASSES).to(device)
    n_params = count_params(model)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # task별 손실 (필요시 class-weighted)
    crits = {}
    for k in TASK_NAMES:
        if class_weights and k in class_weights:
            w = class_weights[k].to(device)
            crits[k] = nn.CrossEntropyLoss(weight=w)
        else:
            crits[k] = nn.CrossEntropyLoss()
    crit = nn.CrossEntropyLoss()  # for evaluate (unweighted, fair comparison)

    history: list[dict] = []
    best_score = -1.0
    best_epoch = -1
    no_improve = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{label}" if label else ""
    ckpt_path = out_dir / f"{model_name}{suffix}_best.pt"

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        n_batches = 0
        t_ep = time.time()
        for step, batch in enumerate(loaders["train"], 1):
            x = batch["x"].to(device, non_blocking=True)
            out = model(x)
            loss = 0.0
            for k in TASK_NAMES:
                y = batch[f"y_{k}"].to(device, non_blocking=True)
                loss = loss + crits[k](out[k], y)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optim.step()
            ep_loss += float(loss.item())
            n_batches += 1

        train_loss = ep_loss / max(n_batches, 1)
        val_metrics = evaluate(model, loaders["val"], device)
        val_score = val_metrics["macro_f1_avg"]

        elapsed = time.time() - t_ep
        per_task_str = " ".join(
            f"{k[:3]}{val_metrics['per_task'][k]['macro_f1']:.3f}" for k in TASK_NAMES
        )
        print(
            f"[{model_name}{suffix}] ep{ep:02d}  train_loss={train_loss:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  val_F1avg={val_score:.4f}  "
            f"({per_task_str})  {elapsed:.1f}s",
            flush=True,
        )
        history.append({
            "epoch": ep,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_macro_f1_avg": val_score,
            "val_per_task": val_metrics["per_task"],
            "time_s": elapsed,
        })

        if val_score > best_score:
            best_score = val_score
            best_epoch = ep
            no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[{model_name}{suffix}] early stop at ep{ep} (best={best_epoch})", flush=True)
                break

    # 최선 가중치로 test 평가
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_metrics = evaluate(model, loaders["test"], device)
    total_time = time.time() - t0

    # confusion matrix는 small task만 (lifestyle, age, sex)
    model.eval()
    cms = {}
    with torch.no_grad():
        all_p = {k: [] for k in TASK_NAMES}
        all_t = {k: [] for k in TASK_NAMES}
        for batch in loaders["test"]:
            x = batch["x"].to(device, non_blocking=True)
            out = model(x)
            for k in TASK_NAMES:
                all_p[k].append(out[k].argmax(1).cpu().numpy())
                all_t[k].append(batch[f"y_{k}"].numpy())
        for k in TASK_NAMES:
            p = np.concatenate(all_p[k])
            t = np.concatenate(all_t[k])
            cm = confusion_matrix(t, p, labels=list(range(NUM_CLASSES[k])))
            cms[k] = cm.tolist()

    result = {
        "model": model_name,
        "label": label,
        "n_params": n_params,
        "best_epoch": best_epoch,
        "best_val_macro_f1_avg": best_score,
        "test": test_metrics,
        "history": history,
        "total_time_s": total_time,
        "device": device,
        "class_names": CLASS_NAMES,
        "confusion_matrices": cms,
    }
    return result


def save_result(result: dict, out_dir: Path, label_suffix: str = ""):
    out_dir.mkdir(parents=True, exist_ok=True)
    name = result["model"] + (f"_{result['label']}" if result["label"] else "") + label_suffix
    (out_dir / f"{name}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
