"""
3 모델 × 3 조건 체크포인트를 불러와 단일 텍스트 예측을 수행한다.
Streamlit app.py 에서 import 한다.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data import (  # noqa: E402
    CharTokenizer, NUM_CLASSES, TASK_NAMES, NARRATIVE_COLS,
    _mask_demographic, _build_geo_pattern,
    build_input_texts, build_labels, stratified_split_indices,
    DATA_PATH,
)
from models import build_model  # noqa: E402
from train import CLASS_NAMES  # noqa: E402

CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"

CONDITIONS = ["raw", "masked", "masked_geo"]
MODELS = ["textcnn", "bilstm", "hybrid"]

CONDITION_LABELS = {
    "raw": "Raw (원본)",
    "masked": "Masked (X대 + 친족어)",
    "masked_geo": "Masked + Geo (지명까지)",
}
MODEL_LABELS = {
    "textcnn": "TextCNN",
    "bilstm": "BiLSTM",
    "hybrid": "CNN-LSTM Hybrid",
}
TASK_LABELS_KO = {
    "sex": "성별",
    "age": "연령대",
    "province": "광역시도",
    "lifestyle": "라이프스타일",
}


def load_tokenizer(condition: str) -> CharTokenizer:
    return CharTokenizer.load(CKPT_DIR / f"tokenizer_{condition}.json")


def load_model(model_name: str, condition: str, vocab_size: int, device: str):
    model = build_model(model_name, vocab_size, NUM_CLASSES).to(device)
    state = torch.load(
        CKPT_DIR / f"{model_name}_{condition}_best.pt",
        map_location=device,
    )
    model.load_state_dict(state)
    model.eval()
    return model


def apply_mask(text: str, condition: str, geo_pattern=None) -> str:
    if condition == "raw":
        return text
    out = _mask_demographic(text)
    if condition == "masked_geo" and geo_pattern is not None:
        out = geo_pattern.sub("[GEO]", out)
    return out


def predict_one(model, tokenizer: CharTokenizer, text: str,
                device: str, max_len: int = 512) -> dict:
    """단일 텍스트 → 4-task 예측 결과 dict."""
    ids = torch.tensor(
        [tokenizer.encode(text, max_len)], dtype=torch.long, device=device
    )
    with torch.no_grad():
        out = model(ids)
    result = {}
    for task in TASK_NAMES:
        logits = out[task][0]
        probs = torch.softmax(logits, dim=0).cpu().numpy()
        pred_idx = int(probs.argmax())
        result[task] = {
            "pred_idx": pred_idx,
            "pred_label": CLASS_NAMES[task][pred_idx],
            "probs": probs.tolist(),
            "classes": CLASS_NAMES[task],
            "confidence": float(probs[pred_idx]),
        }
    return result


def compute_saliency(model, tokenizer: CharTokenizer, text: str, task: str,
                     device: str, max_len: int = 512):
    """
    문자 단위 입력 × gradient saliency. 예측된 class의 logit을 기준으로
    각 글자가 그 예측에 기여한 정도(절대값)를 돌려줌.
    Returns: (scores [L], pred_idx)
    """
    model.eval()
    ids = torch.tensor(
        [tokenizer.encode(text, max_len)], dtype=torch.long, device=device
    )

    # forward hook으로 embedding 출력 캡처
    captured = {}

    def hook(module, inp, out):
        out.retain_grad()
        captured["emb"] = out

    handle = model.embed.register_forward_hook(hook)
    try:
        out = model(ids)
    finally:
        handle.remove()

    logits = out[task][0]
    pred_idx = int(logits.argmax().item())
    target = logits[pred_idx]

    emb = captured["emb"]  # [1, L, D]
    grad = torch.autograd.grad(target, emb, retain_graph=False)[0]  # [1, L, D]
    saliency = (grad * emb).abs().sum(dim=-1).squeeze(0).detach().cpu().numpy()

    actual_len = int((ids[0] != tokenizer.pad_id).sum().item())
    return saliency[:actual_len], pred_idx


# ---------- Stereotype 깨기용 충돌 페르소나 프리셋 ----------
CONFLICTING_PERSONAS = {
    "🎮 60대 PC방 게이머": (
        "65세 김철수 어르신은 매일 PC방에서 6시간씩 e스포츠 게임을 하신다. "
        "디스코드로 길드원들과 활발히 교류하고, 주말마다 LCK 경기를 직관하러 다닌다. "
        "스트리머 방송을 보며 신상 게이밍 키보드를 매주 새로 구매한다."
    ),
    "⚽ 축구광 30대 여성": (
        "30세 박지영씨는 EPL 광팬으로 손흥민 경기는 절대 놓치지 않는다. "
        "회사 풋살 동호회의 주장이고, 주말마다 동네 축구장에서 직접 경기를 뛴다. "
        "FC온라인 매니저 모드를 매일 플레이한다."
    ),
    "🎤 20대 트로트 광팬": (
        "26세 김지수씨는 트로트 가수 임영웅의 열성팬이다. "
        "매주 트로트 콘서트를 다니며, 주말엔 부모님과 막걸리 한 잔에 트로트 메들리를 즐긴다. "
        "전국노래자랑 시청이 인생 최고의 낙이라고 한다."
    ),
    "🏙️ 서울 강남 어부": (
        "서울 강남구의 박여사는 매일 새벽 한강 갯벌에서 조개를 캐고, "
        "꼬막무침과 낙지볶음을 즐겨 만든다. 무등산 등반을 좋아하고, "
        "여수 향토 음식 레시피 책을 쓴다."
    ),
    "🧁 남성 가정주부": (
        "민호씨는 30대 가정주부로 매일 아이 셋을 등원시키고 베이킹과 살림에 매진한다. "
        "요리 블로그를 운영하며, 육아 카페 운영진이고, 뜨개질 동호회에서 활동한다."
    ),
    "🌾 70대 여성 게이머": (
        "75세 이여사는 매일 PC방에서 배틀그라운드를 플레이하고, "
        "트위치에서 게이밍 방송을 자주 본다. 게이밍 마우스 컬렉션이 100개가 넘는다. "
        "손자가 알려준 디스코드로 길드 활동에 열심이다."
    ),
}


def load_full_dataset_for_sampling():
    """test split 인덱스 + raw narrative 텍스트 + 정답 라벨."""
    import pandas as pd
    df = pd.read_parquet(DATA_PATH)
    labels = build_labels(df)
    valid_mask = labels["lifestyle"] >= 0
    df = df.loc[valid_mask].reset_index(drop=True)
    labels = {k: v[valid_mask] for k, v in labels.items()}
    raw_texts = build_input_texts(df, mode="raw")
    geo_pattern = _build_geo_pattern(df)
    y_strat = labels["sex"] * 6 + labels["age"]
    _, _, te_idx = stratified_split_indices(y_strat, seed=42)
    return {
        "df": df,
        "raw_texts": raw_texts,
        "labels": labels,
        "test_idx": te_idx,
        "geo_pattern": geo_pattern,
    }
