#!/usr/bin/env bash
# Run the full pipeline LOCALLY with the laptop stack (CLIP + Moondream2 + Open-i).
#
# Redirects HF / torch caches onto E: because C: is space-constrained.

set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
export HF_HOME="$REPO/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME="$REPO/.cache/torch"
mkdir -p "$HF_HOME" "$TORCH_HOME"

CFG="${CFG:-configs/local.yaml}"
echo "=== HF cache: $HF_HOME"
echo "=== Torch cache: $TORCH_HOME"
echo "=== Config: $CFG"

step() { echo; echo ">>> [$1/$2] $3"; }

step 1 8 "Download Open-i (Indiana University Chest X-rays)"
python -m data_prep.download_openi --config "$CFG"

step 2 8 "Patient-level train/test split"
python -m data_prep.split --config "$CFG"

step 3 8 "Build rule-based VQA dataset (no LLM judge)"
python -m data_prep.build_vqa_dataset_local --config "$CFG"

step 4 8 "Split VQA by patient"
python -m data_prep.split_vqa --config "$CFG"

step 5 8 "CLIP image index over training corpus"
python -m retrieval.clip_index --config "$CFG"

step 6 8 "Run Mode A (report generation) - RAG + baseline"
python -m rag.run_mode_a --config "$CFG" --system both

step 7 8 "Run Mode B (VQA) - RAG + baseline"
python -m rag.run_mode_b --config "$CFG" --system both

step 8 8 "Evaluate (skip CheXbert + BERTScore: too heavy for CPU)"
python -m eval.metrics_report  --config "$CFG" --skip-chexbert --skip-bertscore
python -m eval.metrics_vqa     --config "$CFG"
python -m eval.qualitative_dump --config "$CFG"

echo
echo "DONE. See outputs/ for metrics_mode_a.json, metrics_mode_b.json, and the qualitative dumps."
