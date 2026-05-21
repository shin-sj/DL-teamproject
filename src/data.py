"""
데이터 로드, 4-task 라벨 생성, char-level 토크나이저, train/val/test 분할.
모든 결과는 seed=42 고정 → 팀원 간 재현 가능.
"""
from __future__ import annotations
import ast
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# ---------- 경로 / 상수 ----------
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "nemotron_100k_seed42.parquet"
LIFESTYLE_KW_PATH = ROOT / "mapping" / "lifestyle_keywords.json"

NARRATIVE_COLS = [
    "persona", "professional_persona", "family_persona", "culinary_persona",
    "arts_persona", "cultural_background", "travel_persona", "sports_persona",
    "hobbies_and_interests", "skills_and_expertise", "career_goals_and_ambitions",
]

# 라벨 매핑
SEX_MAP = {"여자": 0, "남자": 1}
SEX_CLASSES = ["여자", "남자"]

AGE_CLASSES = ["20s", "30s", "40s", "50s", "60s", "70+"]
def age_to_bucket(age: float) -> int:
    if age < 30: return 0   # ~29 (~19도 흡수)
    if age < 40: return 1
    if age < 50: return 2
    if age < 60: return 3
    if age < 70: return 4
    return 5

PROVINCE_CLASSES = [
    "강원", "경기", "경상남", "경상북", "광주", "대구", "대전", "부산",
    "서울", "세종", "울산", "인천", "전라남", "전북", "제주", "충청남", "충청북",
]
PROVINCE_MAP = {p: i for i, p in enumerate(PROVINCE_CLASSES)}


# ---------- 라이프스타일 weak labeling ----------
def load_lifestyle_keywords() -> tuple[list[str], dict[str, list[str]]]:
    obj = json.loads(LIFESTYLE_KW_PATH.read_text(encoding="utf-8"))
    return obj["categories"], obj["keywords"]


def _parse_list(cell: Any) -> list[str]:
    """hobbies_and_interests_list 등은 문자열로 저장됨 → 리스트로."""
    if isinstance(cell, list):
        return [str(x) for x in cell]
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    s = str(cell)
    if s.startswith("[") and s.endswith("]"):
        try:
            v = ast.literal_eval(s)
            if isinstance(v, list):
                return [str(x) for x in v]
        except Exception:
            pass
    return [s]


def label_lifestyle(df: pd.DataFrame) -> np.ndarray:
    """
    hobbies_and_interests_list + sports_persona 텍스트에서 카테고리별 키워드 빈도를 세고,
    가장 높은 카테고리 인덱스를 라벨로 부여. 0 매칭이면 -1.
    """
    categories, kw = load_lifestyle_keywords()
    cat_to_idx = {c: i for i, c in enumerate(categories)}

    # 키워드를 길이 내림차순 정렬해서 긴 키워드(예: "가족 여행")가 먼저 매칭되게
    sorted_kw = {c: sorted(set(words), key=lambda w: -len(w)) for c, words in kw.items()}

    labels = np.full(len(df), -1, dtype=np.int64)
    hobbies_lists = df["hobbies_and_interests_list"].tolist()
    sports = df["sports_persona"].fillna("").astype(str).tolist()
    hobbies_text = df["hobbies_and_interests"].fillna("").astype(str).tolist()

    for i in range(len(df)):
        text_blob = " ".join(_parse_list(hobbies_lists[i])) + " " + hobbies_text[i] + " " + sports[i]
        scores = [0] * len(categories)
        for cat, words in sorted_kw.items():
            ci = cat_to_idx[cat]
            for w in words:
                if w in text_blob:
                    scores[ci] += 1
        m = max(scores)
        if m == 0:
            continue
        # 동률시 categories 배열 순서가 빠른 쪽 우선 (index 함수가 첫 매치 반환)
        labels[i] = scores.index(m)
    return labels


# ---------- 라벨 빌더 ----------
def build_labels(df: pd.DataFrame) -> dict[str, np.ndarray]:
    y_sex = df["sex"].map(SEX_MAP).astype(np.int64).values
    y_age = df["age"].astype(float).map(age_to_bucket).astype(np.int64).values
    y_prov = df["province"].map(PROVINCE_MAP).astype(np.int64).values
    y_life = label_lifestyle(df)
    return {"sex": y_sex, "age": y_age, "province": y_prov, "lifestyle": y_life}


# ---------- 텍스트 빌더 (Raw / Masked 조건 지원) ----------
_AGE_PATTERN = re.compile(r"(\d{1,3})\s*대")
_FAMILY_TERMS = ["아내", "남편", "아들", "딸", "엄마", "아빠", "어머니", "아버지",
                 "며느리", "사위", "형", "누나", "오빠", "언니", "동생"]

def _mask_demographic(text: str) -> str:
    text = _AGE_PATTERN.sub("[AGE]", text)
    for w in _FAMILY_TERMS:
        text = text.replace(w, "[FAM]")
    return text

def _build_geo_pattern(df: pd.DataFrame) -> re.Pattern:
    """광역시도 + 시군구 이름을 한 번의 compiled regex로 만듦 (alternation, 긴 것 우선)."""
    words: set[str] = set()
    for p in df["province"].astype(str).unique():
        words.add(p)
        if p.endswith(("남", "북")):
            words.add(p + "도")
    for d in df["district"].astype(str).str.split("-").explode().str.strip().unique():
        if d and len(d) >= 2:
            words.add(d)
    # 길이 내림차순 (긴 매치가 먼저 시도되도록)
    pat = "|".join(re.escape(w) for w in sorted(words, key=lambda w: -len(w)))
    return re.compile(pat)


def build_input_texts(df: pd.DataFrame, mode: str = "raw") -> list[str]:
    """
    11 narrative 컬럼을 [SEP]으로 연결. mode:
      - "raw": 그대로
      - "masked": "X대" + 친족어 마스킹
      - "masked_geo": 위 + 광역시도/시군구 이름까지 마스킹 (regex 한 방)
    """
    sep = " [SEP] "
    geo_re = _build_geo_pattern(df) if mode == "masked_geo" else None

    parts = [df[c].fillna("").astype(str).tolist() for c in NARRATIVE_COLS]
    texts = []
    for i in range(len(df)):
        t = sep.join(parts[c][i] for c in range(len(NARRATIVE_COLS)))
        if mode in ("masked", "masked_geo"):
            t = _mask_demographic(t)
        if geo_re is not None:
            t = geo_re.sub("[GEO]", t)
        texts.append(t)
    return texts


# ---------- 토크나이저 (char/syllable level) ----------
class CharTokenizer:
    PAD = "<pad>"
    UNK = "<unk>"

    def __init__(self, vocab: dict[str, int] | None = None):
        if vocab is None:
            vocab = {self.PAD: 0, self.UNK: 1}
        self.vocab = vocab
        self.pad_id = self.vocab[self.PAD]
        self.unk_id = self.vocab[self.UNK]

    @classmethod
    def build(cls, texts: list[str], min_freq: int = 5) -> "CharTokenizer":
        cnt: Counter = Counter()
        for t in texts:
            cnt.update(t)
        vocab = {cls.PAD: 0, cls.UNK: 1}
        for ch, c in cnt.most_common():
            if c < min_freq:
                continue
            vocab[ch] = len(vocab)
        return cls(vocab)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str, max_len: int) -> list[int]:
        ids = [self.vocab.get(ch, self.unk_id) for ch in text[:max_len]]
        if len(ids) < max_len:
            ids += [self.pad_id] * (max_len - len(ids))
        return ids

    def save(self, path: Path):
        path.write_text(json.dumps(self.vocab, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CharTokenizer":
        return cls(json.loads(path.read_text(encoding="utf-8")))


# ---------- 분할 ----------
def stratified_split_indices(
    y_strat: np.ndarray, train: float = 0.8, val: float = 0.1, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    y_strat 기준 stratified 8/1/1 분할. test = 1 - train - val.
    """
    rng = np.random.default_rng(seed)
    n = len(y_strat)
    idx_train, idx_val, idx_test = [], [], []
    for cls in np.unique(y_strat):
        cls_idx = np.where(y_strat == cls)[0]
        rng.shuffle(cls_idx)
        n_train = int(len(cls_idx) * train)
        n_val = int(len(cls_idx) * val)
        idx_train.append(cls_idx[:n_train])
        idx_val.append(cls_idx[n_train:n_train + n_val])
        idx_test.append(cls_idx[n_train + n_val:])
    tr = np.concatenate(idx_train); rng.shuffle(tr)
    va = np.concatenate(idx_val); rng.shuffle(va)
    te = np.concatenate(idx_test); rng.shuffle(te)
    return tr, va, te


# ---------- Dataset ----------
class PersonaDataset(Dataset):
    def __init__(self, token_ids: np.ndarray, labels: dict[str, np.ndarray]):
        self.x = torch.from_numpy(token_ids).long()
        self.y = {k: torch.from_numpy(v).long() for k, v in labels.items()}

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, i):
        return {
            "x": self.x[i],
            **{f"y_{k}": v[i] for k, v in self.y.items()},
        }


# ---------- 일괄 빌더 ----------
def prepare(
    mode: str = "raw",
    max_len: int = 512,
    seed: int = 42,
    min_freq: int = 5,
) -> dict:
    """모든 준비를 한 번에. 토크나이저는 학습 split에서만 빌드 → 누설 방지."""
    df = pd.read_parquet(DATA_PATH)
    labels = build_labels(df)
    valid_mask = labels["lifestyle"] >= 0  # 라이프스타일 매칭 안 된 샘플 drop
    df = df.loc[valid_mask].reset_index(drop=True)
    labels = {k: v[valid_mask] for k, v in labels.items()}

    texts = build_input_texts(df, mode=mode)

    # stratify: 성별 × 연령대로 cross 라벨 만들어 split — 모든 task가 어느 정도 균형 잡히게
    y_strat = labels["sex"] * 6 + labels["age"]
    tr_idx, va_idx, te_idx = stratified_split_indices(y_strat, seed=seed)

    train_texts = [texts[i] for i in tr_idx]
    tok = CharTokenizer.build(train_texts, min_freq=min_freq)

    def encode_split(idx):
        arr = np.zeros((len(idx), max_len), dtype=np.int64)
        for j, i in enumerate(idx):
            arr[j] = tok.encode(texts[i], max_len)
        return arr

    splits = {}
    for name, idx in [("train", tr_idx), ("val", va_idx), ("test", te_idx)]:
        splits[name] = {
            "ids": encode_split(idx),
            "labels": {k: v[idx] for k, v in labels.items()},
            "n": len(idx),
        }

    return {
        "splits": splits,
        "tokenizer": tok,
        "n_total": len(df),
        "n_dropped": int((~valid_mask).sum()),
        "mode": mode,
        "max_len": max_len,
    }


# 각 task별 클래스 수 (모델에 전달)
NUM_CLASSES = {
    "sex": 2,
    "age": len(AGE_CLASSES),
    "province": len(PROVINCE_CLASSES),
    "lifestyle": 8,
}
TASK_NAMES = list(NUM_CLASSES.keys())


def make_loaders(prep: dict, batch_size: int = 256) -> dict[str, DataLoader]:
    loaders = {}
    for name, sp in prep["splits"].items():
        ds = PersonaDataset(sp["ids"], sp["labels"])
        loaders[name] = DataLoader(
            ds, batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=0, pin_memory=True,
        )
    return loaders


if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    p = prepare(mode="raw", max_len=512)
    print(f"after lifestyle filter: {p['n_total']:,} (dropped {p['n_dropped']:,})")
    print(f"vocab size: {p['tokenizer'].vocab_size:,}")
    for k, v in p["splits"].items():
        print(f"  {k}: n={v['n']:,}  ids shape={v['ids'].shape}")
    print("\nlabel distribution (train):")
    for task in TASK_NAMES:
        y = p["splits"]["train"]["labels"][task]
        vals, cnts = np.unique(y, return_counts=True)
        d = dict(zip(vals.tolist(), cnts.tolist()))
        print(f"  {task} ({NUM_CLASSES[task]}-class): {d}")
