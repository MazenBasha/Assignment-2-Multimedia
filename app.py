"""Gradio web demo of the ColPali-RAG vs MedGemma pipeline.

Defaults to the laptop stack (CLIP + stub generator) so it runs fast
without GPU or model downloads. Flip CFG_PATH to a different config to
swap backends.

Launch:
    python app.py
Then open http://localhost:7860 in your browser.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Optional

# Cache redirection so HF model downloads don't blow up the C: drive.
REPO = Path(__file__).parent
os.environ.setdefault("HF_HOME", str(REPO / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(REPO / ".cache" / "huggingface" / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(REPO / ".cache" / "huggingface" / "transformers"))
os.environ.setdefault("TORCH_HOME", str(REPO / ".cache" / "torch"))

import gradio as gr
import pandas as pd
from PIL import Image

from data_prep.common import Paths, load_config
from data_prep.report_sections import reference_text_for_eval
from generation.medgemma_wrapper import GenConfig, split_prompt_template
from rag.factory import make_generator, make_searcher
from retrieval.colpali_search import format_retrieved_for_prompt


# -------------------------------------------------------------------
# Config + lazy singletons (loaded on first use).
# -------------------------------------------------------------------
CFG_PATH = os.environ.get("CFG", "configs/local_stub.yaml")
CFG = load_config(CFG_PATH)
PATHS = Paths.from_config(CFG)

PROMPTS_DIR = REPO / "prompts"
TPL_RAG_REPORT      = (PROMPTS_DIR / "rag_report_generation.txt").read_text(encoding="utf-8")
TPL_RAG_VQA         = (PROMPTS_DIR / "rag_vqa.txt").read_text(encoding="utf-8")
TPL_BASELINE_REPORT = (PROMPTS_DIR / "baseline_report_generation.txt").read_text(encoding="utf-8")
TPL_BASELINE_VQA    = (PROMPTS_DIR / "baseline_vqa.txt").read_text(encoding="utf-8")

# Build singletons once. Searcher loads CLIP + the index; generator is cheap (stub).
print(f"Loading pipeline from {CFG_PATH} ...", flush=True)
GENERATOR = make_generator(CFG)
SEARCHER  = make_searcher(CFG, PATHS.artifacts_dir)
print("Ready. Open http://localhost:7860", flush=True)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _test_set() -> pd.DataFrame:
    p = PATHS.splits_dir / "test.parquet"
    if not p.exists():
        return pd.DataFrame(columns=["study_id", "image_path", "text"])
    return pd.read_parquet(p)


def _vqa_test_set() -> pd.DataFrame:
    p = PATHS.splits_dir / "vqa_test.parquet"
    if not p.exists():
        return pd.DataFrame(columns=["question_id", "study_id", "image_path", "question", "answer", "question_type"])
    return pd.read_parquet(p)


TEST_DF     = _test_set()
TEST_LABELS = [f"{r['study_id']} - {(r['text'] or '')[:60].strip()}..."
               for _, r in TEST_DF.iterrows()]
LABEL_TO_ROW = {lab: r for lab, (_, r) in zip(TEST_LABELS, TEST_DF.iterrows())}

VQA_DF      = _vqa_test_set()
VQA_LABELS  = [f"{r['question_type']}: {r['question']}" for _, r in VQA_DF.iterrows()]
VQA_LABEL_TO_ROW = {lab: r for lab, (_, r) in zip(VQA_LABELS, VQA_DF.iterrows())}


def _format_retrieved_md(items) -> str:
    lines = ["### Retrieved similar X-rays\n"]
    for i, r in enumerate(items):
        lines.append(f"**#{i+1}**  study_id=`{r.study_id}`  similarity=`{r.score:.3f}`\n")
        txt = (r.text or "").strip().replace("\n", "  \n")
        lines.append(f"> {txt[:800]}\n")
    return "\n".join(lines)


def _gen_cfg_report() -> GenConfig:
    return GenConfig(
        max_new_tokens=CFG["generation"]["max_new_tokens"]["report"],
        temperature=CFG["generation"]["temperature"],
        do_sample=CFG["generation"]["do_sample"],
        num_beams=CFG["generation"]["num_beams"],
    )


def _gen_cfg_vqa() -> GenConfig:
    return GenConfig(
        max_new_tokens=CFG["generation"]["max_new_tokens"]["vqa"],
        temperature=CFG["generation"]["temperature"],
        do_sample=CFG["generation"]["do_sample"],
        num_beams=CFG["generation"]["num_beams"],
    )


# -------------------------------------------------------------------
# Inference handlers
# -------------------------------------------------------------------
def run_mode_a(uploaded_image, picked_label) -> tuple:
    """Returns (image, ground_truth, rag_prediction, baseline_prediction, retrieved_md)."""
    if uploaded_image is not None:
        image = Image.fromarray(uploaded_image).convert("RGB")
        gt = "(no ground truth — custom upload)"
    elif picked_label and picked_label in LABEL_TO_ROW:
        row = LABEL_TO_ROW[picked_label]
        image = Image.open(row["image_path"]).convert("RGB")
        gt = reference_text_for_eval(str(row["text"]))
    else:
        return None, "Pick an example or upload an image.", "", "", ""

    # RAG
    retrieved = SEARCHER.search(image, top_k=CFG["retrieval"]["top_k"])
    rag_sys, rag_user_tpl = split_prompt_template(TPL_RAG_REPORT)
    rag_user = rag_user_tpl.format(k=len(retrieved),
                                   retrieved_reports=format_retrieved_for_prompt(retrieved))
    rag_pred = GENERATOR.generate(image, system=rag_sys, user=rag_user, cfg=_gen_cfg_report())

    # Baseline
    base_sys, base_user = split_prompt_template(TPL_BASELINE_REPORT)
    base_pred = GENERATOR.generate(image, system=base_sys, user=base_user, cfg=_gen_cfg_report())

    return image, gt, rag_pred, base_pred, _format_retrieved_md(retrieved)


def run_mode_b(uploaded_image, picked_label, question) -> tuple:
    """Returns (image, ground_truth_answer, rag_answer, baseline_answer, retrieved_md)."""
    if uploaded_image is not None:
        image = Image.fromarray(uploaded_image).convert("RGB")
        gt = "(no ground truth — custom upload)"
    elif picked_label and picked_label in VQA_LABEL_TO_ROW:
        row = VQA_LABEL_TO_ROW[picked_label]
        image = Image.open(row["image_path"]).convert("RGB")
        question = row["question"]
        gt = row["answer"]
    else:
        return None, "Pick a question or upload an image + type a question.", "", "", ""

    if not question or not question.strip():
        return image, gt, "Please enter a question.", "Please enter a question.", ""

    retrieved = SEARCHER.search(image, top_k=CFG["retrieval"]["top_k"])
    rag_sys, rag_user_tpl = split_prompt_template(TPL_RAG_VQA)
    rag_user = rag_user_tpl.format(
        retrieved_reports=format_retrieved_for_prompt(retrieved),
        question=question.strip(),
    )
    rag_pred = GENERATOR.generate(image, system=rag_sys, user=rag_user, cfg=_gen_cfg_vqa())

    base_sys, base_user_tpl = split_prompt_template(TPL_BASELINE_VQA)
    base_user = base_user_tpl.format(question=question.strip())
    base_pred = GENERATOR.generate(image, system=base_sys, user=base_user, cfg=_gen_cfg_vqa())

    return image, gt, rag_pred, base_pred, _format_retrieved_md(retrieved)


def populate_question(picked_label: str) -> str:
    if picked_label and picked_label in VQA_LABEL_TO_ROW:
        return str(VQA_LABEL_TO_ROW[picked_label]["question"])
    return ""


# -------------------------------------------------------------------
# UI
# -------------------------------------------------------------------
with gr.Blocks(title="Chest X-ray RAG demo") as demo:
    gr.Markdown(
        "# Chest X-ray RAG demo  \n"
        f"Backend: **{CFG['retrieval'].get('backend','?')}** retrieval + "
        f"**{CFG['generation'].get('backend','?')}** generation  "
        f"(config: `{CFG_PATH}`)"
    )

    with gr.Tab("Mode A - Report generation"):
        with gr.Row():
            with gr.Column():
                a_pick = gr.Dropdown(choices=TEST_LABELS, label="Pick a test X-ray", value=None)
                a_upload = gr.Image(label="...or upload your own (PNG/JPG)", type="numpy")
                a_btn = gr.Button("Generate report", variant="primary")
                a_image_out = gr.Image(label="Input image", interactive=False)
            with gr.Column():
                a_gt = gr.Textbox(label="Ground truth (reference)", lines=8)
                a_rag = gr.Textbox(label="ColPali/CLIP-RAG prediction", lines=8)
                a_base = gr.Textbox(label="Baseline (no retrieval) prediction", lines=6)
                a_retr = gr.Markdown(label="Retrieved reports")
        a_btn.click(run_mode_a,
                    inputs=[a_upload, a_pick],
                    outputs=[a_image_out, a_gt, a_rag, a_base, a_retr])

    with gr.Tab("Mode B - VQA"):
        with gr.Row():
            with gr.Column():
                b_pick = gr.Dropdown(choices=VQA_LABELS, label="Pick a test question", value=None)
                b_question = gr.Textbox(label="Question", placeholder="e.g. Is there pleural effusion?")
                b_upload = gr.Image(label="...or upload your own + type a question", type="numpy")
                b_btn = gr.Button("Answer", variant="primary")
                b_image_out = gr.Image(label="Input image", interactive=False)
            with gr.Column():
                b_gt = gr.Textbox(label="Ground truth answer", lines=2)
                b_rag = gr.Textbox(label="RAG answer", lines=2)
                b_base = gr.Textbox(label="Baseline answer", lines=2)
                b_retr = gr.Markdown(label="Retrieved reports")
        b_pick.change(populate_question, inputs=b_pick, outputs=b_question)
        b_btn.click(run_mode_b,
                    inputs=[b_upload, b_pick, b_question],
                    outputs=[b_image_out, b_gt, b_rag, b_base, b_retr])

    gr.Markdown(
        f"---\nTest set: {len(TEST_DF)} X-rays, {len(VQA_DF)} VQA pairs. "
        f"Corpus: {CFG['dataset']['max_train_corpus']} images. "
        "Edit `CFG_PATH` env var or change `configs/local_stub.yaml` to switch backends."
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True)
