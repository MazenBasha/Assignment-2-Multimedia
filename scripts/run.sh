#!/usr/bin/env bash
set -euo pipefail
CFG="${CFG:-configs/config.yaml}"

echo ">>> [1/8] Download MIMIC-CXR (Kaggle)"
python -m data_prep.download_kaggle --config "$CFG"

echo ">>> [2/8] Preprocess images + reports"
python -m data_prep.preprocess --config "$CFG"

echo ">>> [3/8] Patient-level train/test split"
python -m data_prep.split --config "$CFG"

echo ">>> [4/8] Build VQA dataset (LLM judge)"
python -m data_prep.build_vqa_dataset --config "$CFG"

echo ">>> [5/8] Split VQA by patient"
python -m data_prep.split_vqa --config "$CFG"

echo ">>> [6/8] Build ColPali index over training corpus"
python -m retrieval.colpali_index --config "$CFG"

echo ">>> [7/8] Run Mode A (report generation) — RAG + baseline"
python -m rag.run_mode_a --config "$CFG" --system both

echo ">>> [7/8] Run Mode B (VQA) — RAG + baseline"
python -m rag.run_mode_b --config "$CFG" --system both

echo ">>> [8/8] Evaluate"
python -m eval.metrics_report  --config "$CFG"
python -m eval.metrics_vqa     --config "$CFG"
python -m eval.qualitative_dump --config "$CFG"

echo "Done. See outputs/metrics_mode_a.json, outputs/metrics_mode_b.json."
