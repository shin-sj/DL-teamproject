"""
Streamlit demo: 한국어 페르소나 → 4-task demographic 예측.
- 단일 예측 (모델/조건 선택 + 직접 입력 또는 test 샘플 랜덤 추출)
- 모델 × 조건 비교 (같은 입력 → 9 셀)
- 혼동행렬 (model × condition × task)
- 전체 성능 표 (RESULTS.md 의 표를 동적으로 재생성)

실행:  streamlit run app/app.py
"""
from __future__ import annotations
import sys
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(ROOT / "src"))

from inference import (  # noqa: E402
    CONDITIONS, MODELS, CONDITION_LABELS, MODEL_LABELS, TASK_LABELS_KO,
    RESULTS_DIR, load_tokenizer, load_model, predict_one, apply_mask,
    load_full_dataset_for_sampling, compute_saliency, CONFLICTING_PERSONAS,
)
from data import TASK_NAMES  # noqa: E402
from train import CLASS_NAMES  # noqa: E402


st.set_page_config(page_title="페르소나 분류 데모", layout="wide")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------- caches ----------
@st.cache_resource(show_spinner="토크나이저 로딩...")
def cached_tokenizer(condition: str):
    return load_tokenizer(condition)


@st.cache_resource(show_spinner="모델 로딩...")
def cached_model(model_name: str, condition: str):
    tok = cached_tokenizer(condition)
    return load_model(model_name, condition, tok.vocab_size, DEVICE)


@st.cache_resource(show_spinner="100k 데이터셋 로딩 (test split 샘플링용)...")
def cached_sampling():
    return load_full_dataset_for_sampling()


@st.cache_data
def load_summary(condition: str):
    return json.loads(
        (RESULTS_DIR / f"summary_{condition}.json").read_text(encoding="utf-8")
    )


@st.cache_data
def load_result(model_name: str, condition: str):
    return json.loads(
        (RESULTS_DIR / f"{model_name}_{condition}.json").read_text(encoding="utf-8")
    )


# ---------- helpers ----------
def task_card(task: str, task_result: dict, true_idx: int | None = None,
              show_chart: bool = True):
    pred_label = task_result["pred_label"]
    pred_idx = task_result["pred_idx"]
    conf = task_result["confidence"]
    classes = task_result["classes"]
    has_truth = true_idx is not None and true_idx >= 0
    true_label = classes[true_idx] if has_truth else None
    correct = (true_idx == pred_idx) if has_truth else None

    title = f"**{TASK_LABELS_KO[task]}**"
    if correct is True:
        title += " ✅"
    elif correct is False:
        title += " ❌"
    st.markdown(title)

    line = f"예측: `{pred_label}` ({conf:.1%})"
    if true_label is not None:
        line += f"  ·  실제: `{true_label}`"
    st.write(line)

    if show_chart:
        df = pd.DataFrame({
            "class": classes,
            "prob": task_result["probs"],
        }).sort_values("prob", ascending=False)
        st.bar_chart(df.set_index("class"), height=180)


def predict_with_mask(model_name: str, condition: str, raw_text: str,
                      geo_pattern) -> tuple[str, dict]:
    """raw_text 를 조건에 맞게 마스킹한 뒤 예측. (masked_text, result) 반환."""
    masked = apply_mask(raw_text, condition, geo_pattern=geo_pattern)
    tok = cached_tokenizer(condition)
    model = cached_model(model_name, condition)
    result = predict_one(model, tok, masked, DEVICE)
    return masked, result


def saliency_html(text: str, scores: np.ndarray, max_chars: int = 800) -> str:
    """글자별 saliency 점수를 HTML로 (빨간 배경 강도 ∝ 중요도)."""
    text = text[:max_chars]
    n = min(len(text), len(scores), max_chars)
    if n == 0:
        return ""
    smax = float(scores[:n].max()) if scores[:n].max() > 0 else 1.0
    parts = [
        '<div style="font-family:\'JetBrains Mono\',monospace; line-height:1.95; '
        'font-size:14px; padding:10px 12px; background:#F8FAFC; '
        'border:1px solid #E2E8F0; border-radius:8px; word-break:break-all;">'
    ]
    for i in range(n):
        ch = text[i]
        s = float(scores[i]) / smax
        if ch == "\n":
            parts.append("<br>")
            continue
        if ch == " ":
            parts.append("&nbsp;")
            continue
        esc = ch.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if s > 0.05:
            alpha = min(s, 1.0)
            parts.append(
                f'<span style="background:rgba(220,38,38,{alpha:.2f});'
                f' padding:1px 0; border-radius:2px;">{esc}</span>'
            )
        else:
            parts.append(esc)
    parts.append("</div>")
    return "".join(parts)


def render_saliency_for_task(model_name: str, condition: str, masked_text: str,
                              task: str):
    """탭에서 saliency를 한 task 단위로 그리는 헬퍼."""
    tok = cached_tokenizer(condition)
    model = cached_model(model_name, condition)
    scores, pred_idx = compute_saliency(model, tok, masked_text, task, DEVICE)
    pred_label = CLASS_NAMES[task][pred_idx]
    st.markdown(
        f"<small><b>{TASK_LABELS_KO[task]}</b> → <code>{pred_label}</code> "
        f"예측에 기여한 글자 (빨강 진할수록 중요)</small>",
        unsafe_allow_html=True,
    )
    st.markdown(saliency_html(masked_text, scores), unsafe_allow_html=True)


def render_confusion_matrix(cm: list[list[int]], classes: list[str], task: str):
    arr = np.array(cm, dtype=int)
    df = pd.DataFrame(arr, index=classes, columns=classes)
    df.index.name = "실제 (true)"
    df.columns.name = "예측 (pred)"

    row_sums = arr.sum(axis=1, keepdims=True).clip(min=1)
    norm = arr / row_sums

    styled = (
        df.style
        .background_gradient(cmap="Blues", axis=None,
                              gmap=pd.DataFrame(norm, index=classes, columns=classes))
        .format("{:d}")
    )
    st.dataframe(styled, use_container_width=True)
    st.caption(
        f"각 행의 합 = 해당 실제 클래스 샘플 수. 대각선 비중이 높을수록 정확. "
        f"(task = {TASK_LABELS_KO[task]}, {len(classes)}-class)"
    )


# ---------- header ----------
st.title("📚 페르소나 분류 데모")
st.caption(
    "Nemotron-Personas-Korea 100k → 4-task multi-task model "
    "(성별 / 연령대 / 광역시도 / 라이프스타일).  "
    f"Device = {DEVICE}"
)

tab1, tab_break, tab2, tab3, tab4 = st.tabs([
    "🎯 단일 예측",
    "🎭 Stereotype 깨기",
    "🔬 모델 × 조건 비교",
    "🧮 혼동행렬",
    "📊 전체 성능 표",
])


# ============================================================
# Tab 1 — 단일 예측
# ============================================================
with tab1:
    st.subheader("단일 예측")

    col_in, col_cfg = st.columns([3, 1])

    with col_cfg:
        st.markdown("**설정**")
        model_choice = st.selectbox(
            "모델", MODELS, format_func=lambda m: MODEL_LABELS[m], key="t1_model"
        )
        cond_choice = st.selectbox(
            "조건", CONDITIONS, format_func=lambda c: CONDITION_LABELS[c],
            key="t1_cond"
        )
        show_saliency = st.checkbox(
            "🔍 saliency 보기",
            help="예측에 어떤 글자가 기여했는지 빨간 강도로 표시",
            key="t1_saliency",
        )
        st.markdown("---")
        st.markdown("**테스트셋 샘플**")
        if st.button("🎲 랜덤 추출", use_container_width=True):
            ds = cached_sampling()
            idx = int(random.choice(ds["test_idx"]))
            st.session_state["t1_text"] = ds["raw_texts"][idx]
            st.session_state["t1_truth"] = {k: int(v[idx]) for k, v in ds["labels"].items()}
            st.session_state["t1_idx"] = idx

    with col_in:
        st.markdown("**페르소나 narrative** (11 컬럼을 ` [SEP] ` 으로 연결한 텍스트)")
        text = st.text_area(
            "input",
            value=st.session_state.get("t1_text", ""),
            height=240,
            label_visibility="collapsed",
            placeholder="예: 60대 후반의 김영자 여사는 경상남도 통영의 작은 어촌에서 ...",
            key="t1_text",
        )

    st.markdown("---")

    if not text.strip():
        st.info("페르소나 텍스트를 입력하거나 우측 '🎲 랜덤 추출' 버튼을 눌러주세요.")
    else:
        ds = cached_sampling() if "t1_truth" in st.session_state else None
        truth = st.session_state.get("t1_truth")

        ds_for_geo = cached_sampling()
        masked_text, result = predict_with_mask(
            model_choice, cond_choice, text, ds_for_geo["geo_pattern"]
        )

        if masked_text != text:
            with st.expander(
                f"🔍 모델이 실제로 본 입력 ({CONDITION_LABELS[cond_choice]})"
            ):
                st.text(masked_text[:2000] + ("..." if len(masked_text) > 2000 else ""))

        st.markdown(
            f"### 결과 — `{MODEL_LABELS[model_choice]}` × `{CONDITION_LABELS[cond_choice]}`"
        )

        rows = st.columns(2)
        for i, task in enumerate(TASK_NAMES):
            with rows[i % 2]:
                true_idx = truth.get(task) if truth else None
                task_card(task, result[task], true_idx=true_idx)

        if show_saliency:
            st.markdown("### 🔍 Saliency — task별 글자 기여도")
            st.caption(
                "예측된 클래스의 logit에 대해 각 글자가 얼마나 기여했는지 "
                "(input × gradient). 빨강 진할수록 그 글자가 모델 판단에 결정적."
            )
            for task in TASK_NAMES:
                with st.expander(
                    f"{TASK_LABELS_KO[task]} → {result[task]['pred_label']} "
                    f"({result[task]['confidence']:.0%})",
                    expanded=(task == "province"),
                ):
                    render_saliency_for_task(
                        model_choice, cond_choice, masked_text, task
                    )


# ============================================================
# Tab "Stereotype 깨기" (충돌 페르소나) — 정의 순서상 tab_break
# ============================================================
with tab_break:
    st.subheader("🎭 Stereotype 깨기 — 모델은 명시적 정보 vs stereotype 키워드 중 어느 쪽에 휘둘리나?")
    st.caption(
        "프리셋은 '60대 게이머', '축구 좋아하는 30대 여성'처럼 일부러 한국 사회의 "
        "고정관념과 어긋나는 페르소나. 모델이 명시적 단서(나이·성별)를 따라가는지, "
        "stereotype 키워드(게임·축구·트로트)에 끌려가는지 직접 확인."
    )

    col_l, col_r = st.columns([3, 1])

    with col_r:
        st.markdown("**프리셋**")
        preset_keys = list(CONFLICTING_PERSONAS.keys())
        for k in preset_keys:
            if st.button(k, use_container_width=True, key=f"tb_preset_{k}"):
                st.session_state["tb_text"] = CONFLICTING_PERSONAS[k]
                st.session_state["tb_preset_name"] = k

        st.markdown("---")
        st.markdown("**모델 / 조건**")
        tb_model = st.selectbox(
            "모델", MODELS, format_func=lambda m: MODEL_LABELS[m], key="tb_model"
        )
        tb_cond = st.selectbox(
            "조건", CONDITIONS, format_func=lambda c: CONDITION_LABELS[c],
            key="tb_cond", index=0,
        )
        tb_saliency = st.checkbox(
            "🔍 saliency 보기", value=True, key="tb_saliency",
            help="모델이 어떤 글자에 끌렸는지 표시",
        )

    with col_l:
        st.markdown("**페르소나 텍스트** (편집 가능)")
        tb_text = st.text_area(
            "tb_input",
            value=st.session_state.get("tb_text", ""),
            height=200,
            label_visibility="collapsed",
            placeholder="우측 프리셋 버튼을 누르거나 직접 입력...",
            key="tb_text",
        )

    if not tb_text.strip():
        st.info("우측 프리셋 중 하나를 골라보세요.")
    else:
        ds_for_geo = cached_sampling()
        masked_tb, result_tb = predict_with_mask(
            tb_model, tb_cond, tb_text, ds_for_geo["geo_pattern"]
        )

        if masked_tb != tb_text:
            with st.expander(
                f"🔍 모델이 실제로 본 입력 ({CONDITION_LABELS[tb_cond]})"
            ):
                st.text(masked_tb)

        st.markdown(
            f"#### 결과 — `{MODEL_LABELS[tb_model]}` × `{CONDITION_LABELS[tb_cond]}`"
        )

        # 4 task 결과 + 엔트로피(불확실도) 표시
        rows = st.columns(4)
        for i, task in enumerate(TASK_NAMES):
            with rows[i]:
                r = result_tb[task]
                probs = np.array(r["probs"])
                # 정규화된 엔트로피 = -Σ p log p / log(n_classes)
                eps = 1e-12
                ent = -(probs * np.log(probs + eps)).sum()
                ent_norm = ent / np.log(len(probs))
                st.metric(
                    label=TASK_LABELS_KO[task],
                    value=r["pred_label"],
                    delta=f"{r['confidence']:.0%} conf",
                )
                st.caption(f"불확실도 {ent_norm:.2f}")

        # 4 task 확률 막대를 한꺼번에
        st.markdown("##### 4 task 확률 분포")
        cols = st.columns(4)
        for i, task in enumerate(TASK_NAMES):
            with cols[i]:
                r = result_tb[task]
                df = pd.DataFrame({"class": r["classes"], "prob": r["probs"]})
                st.bar_chart(df.set_index("class"), height=160)

        if tb_saliency:
            st.markdown("---")
            st.markdown("##### 🔍 모델은 어떤 글자에 끌렸나")
            st.caption(
                "각 task의 예측된 클래스 logit에 대한 글자 기여도. "
                "stereotype 키워드(게임·트로트·갯벌 등)에 빨간색이 몰리면 "
                "모델이 키워드 기반 stereotype 추론을 하고 있다는 증거."
            )
            for task in TASK_NAMES:
                with st.expander(
                    f"{TASK_LABELS_KO[task]} → {result_tb[task]['pred_label']} "
                    f"({result_tb[task]['confidence']:.0%})",
                    expanded=False,
                ):
                    render_saliency_for_task(tb_model, tb_cond, masked_tb, task)


# ============================================================
# Tab 2 — 모델 × 조건 비교
# ============================================================
with tab2:
    st.subheader("모델 × 조건 비교 — 같은 입력에 대한 9 셀")

    col_in, col_cfg = st.columns([3, 1])

    with col_cfg:
        st.markdown("**관찰할 task**")
        focus_task = st.selectbox(
            "task", TASK_NAMES,
            format_func=lambda t: TASK_LABELS_KO[t],
            key="t2_task",
        )
        st.markdown("---")
        st.markdown("**테스트셋 샘플**")
        if st.button("🎲 랜덤 추출", use_container_width=True, key="t2_rand"):
            ds = cached_sampling()
            idx = int(random.choice(ds["test_idx"]))
            st.session_state["t2_text"] = ds["raw_texts"][idx]
            st.session_state["t2_truth"] = {k: int(v[idx]) for k, v in ds["labels"].items()}

    with col_in:
        st.markdown("**페르소나 narrative**")
        text2 = st.text_area(
            "input2",
            value=st.session_state.get("t2_text", ""),
            height=200,
            label_visibility="collapsed",
            key="t2_text",
        )

    st.markdown("---")

    if not text2.strip():
        st.info("입력이 비어 있습니다.")
    else:
        ds = cached_sampling()
        truth = st.session_state.get("t2_truth")
        true_idx = truth.get(focus_task) if truth else None
        true_label = (
            CLASS_NAMES[focus_task][true_idx]
            if true_idx is not None and true_idx >= 0 else None
        )

        if true_label is not None:
            st.markdown(f"**실제 정답** ({TASK_LABELS_KO[focus_task]}): `{true_label}`")

        # 9 셀 채우기
        with st.spinner("9 셀 추론 중..."):
            grid = {}
            for m in MODELS:
                for c in CONDITIONS:
                    _, r = predict_with_mask(m, c, text2, ds["geo_pattern"])
                    grid[(m, c)] = r[focus_task]

        # 표: 행=모델, 열=조건
        table_rows = []
        for m in MODELS:
            row = {"모델": MODEL_LABELS[m]}
            for c in CONDITIONS:
                cell = grid[(m, c)]
                label = cell["pred_label"]
                conf = cell["confidence"]
                mark = ""
                if true_idx is not None and true_idx >= 0:
                    mark = " ✅" if cell["pred_idx"] == true_idx else " ❌"
                row[CONDITION_LABELS[c]] = f"{label}{mark}  ({conf:.0%})"
            table_rows.append(row)

        st.dataframe(
            pd.DataFrame(table_rows).set_index("모델"),
            use_container_width=True,
        )

        st.markdown("#### 확률 막대 비교")
        chart_cols = st.columns(len(CONDITIONS))
        for j, c in enumerate(CONDITIONS):
            with chart_cols[j]:
                st.markdown(f"**{CONDITION_LABELS[c]}**")
                rows = []
                for m in MODELS:
                    cell = grid[(m, c)]
                    for cls, p in zip(cell["classes"], cell["probs"]):
                        rows.append({"model": MODEL_LABELS[m], "class": cls, "prob": p})
                df = pd.DataFrame(rows)
                pivot = df.pivot(index="class", columns="model", values="prob")
                st.bar_chart(pivot, height=260)


# ============================================================
# Tab 3 — 혼동행렬
# ============================================================
with tab3:
    st.subheader("혼동행렬 (test split)")

    c1, c2, c3 = st.columns(3)
    with c1:
        m_choice = st.selectbox(
            "모델", MODELS, format_func=lambda m: MODEL_LABELS[m], key="t3_model"
        )
    with c2:
        c_choice = st.selectbox(
            "조건", CONDITIONS, format_func=lambda c: CONDITION_LABELS[c], key="t3_cond"
        )
    with c3:
        t_choice = st.selectbox(
            "task", TASK_NAMES, format_func=lambda t: TASK_LABELS_KO[t], key="t3_task"
        )

    try:
        res = load_result(m_choice, c_choice)
    except FileNotFoundError:
        st.error(f"results/{m_choice}_{c_choice}.json 이 없습니다.")
    else:
        per = res["test"]["per_task"][t_choice]
        st.markdown(
            f"**macro-F1** = {per['macro_f1']*100:.2f}%   "
            f"·   **accuracy** = {per['acc']*100:.2f}%   "
            f"·   params = {res['n_params']:,}"
        )
        cm = res["confusion_matrices"][t_choice]
        classes = res["class_names"][t_choice]
        render_confusion_matrix(cm, classes, t_choice)


# ============================================================
# Tab 4 — 전체 성능 표
# ============================================================
with tab4:
    st.subheader("전체 성능 표")

    rows = []
    for cond in CONDITIONS:
        try:
            s = load_summary(cond)
        except FileNotFoundError:
            continue
        for m, r in s["results"].items():
            row = {
                "조건": CONDITION_LABELS[cond],
                "모델": MODEL_LABELS[m],
                "params": r["n_params"],
                "best ep": r["best_epoch"],
            }
            for t in TASK_NAMES:
                row[f"{TASK_LABELS_KO[t]} F1"] = r["test_per_task"][t]["macro_f1"] * 100
            row["avg F1"] = r["test_macro_f1_avg"] * 100
            row["time(s)"] = round(r["total_time_s"])
            rows.append(row)

    df = pd.DataFrame(rows)

    f1_cols = [c for c in df.columns if c.endswith("F1")]
    styled = (
        df.style
        .background_gradient(cmap="YlGn", subset=f1_cols, axis=None, vmin=0, vmax=100)
        .format({c: "{:.2f}" for c in f1_cols})
        .format({"params": "{:,}", "time(s)": "{:.0f}"})
    )
    st.dataframe(styled, use_container_width=True, height=420)

    st.markdown("---")
    st.markdown("#### Leakage Ablation — 조건 변화에 따른 Δ%p (TextCNN 기준 권장)")

    m_focus = st.selectbox(
        "기준 모델", MODELS, format_func=lambda m: MODEL_LABELS[m], key="t4_model"
    )
    abl = []
    base = {}
    for cond in CONDITIONS:
        try:
            s = load_summary(cond)
            base[cond] = s["results"][m_focus]["test_per_task"]
        except (FileNotFoundError, KeyError):
            pass

    if "raw" in base:
        for t in TASK_NAMES:
            raw = base["raw"][t]["macro_f1"] * 100
            masked = base.get("masked", {}).get(t, {}).get("macro_f1", float("nan")) * 100 \
                if "masked" in base else float("nan")
            mgeo = base.get("masked_geo", {}).get(t, {}).get("macro_f1", float("nan")) * 100 \
                if "masked_geo" in base else float("nan")
            abl.append({
                "task": TASK_LABELS_KO[t],
                "Raw": raw,
                "Masked": masked,
                "Δ(Raw→Masked)": masked - raw,
                "Masked+Geo": mgeo,
                "Δ(Masked→+Geo)": mgeo - masked,
            })

    abl_df = pd.DataFrame(abl)
    fmt = {c: "{:+.2f}" if c.startswith("Δ") else "{:.2f}" for c in abl_df.columns
           if c != "task"}
    delta_cols = [c for c in abl_df.columns if c.startswith("Δ")]
    abl_styled = (
        abl_df.style
        .background_gradient(cmap="RdBu", subset=delta_cols, axis=None, vmin=-40, vmax=40)
        .format(fmt)
    )
    st.dataframe(abl_styled, use_container_width=True)

    st.caption(
        "Δ(Raw→Masked) = 'X대' / 친족어 마스킹으로 잃은 점수 "
        "(연령대에서 크면 LLM이 X대를 직접 박아넣은 증거).  "
        "Δ(Masked→+Geo) = 광역시도/시군구 지명 마스킹으로 잃은 점수 "
        "(광역시도에서 크면 지명 복사에 의존하던 증거).  "
        "남은 점수 = stereotype 우회 학습으로 모델이 잡아낸 신호."
    )
