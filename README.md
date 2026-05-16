# DSAI 413 — Assignment 2

**Multimodal Medical AI: ColPali-RAG vs. MedGemma on MIMIC-CXR**

Two systems, two modes (report generation + VQA), benchmarked on chest
X-rays from the Kaggle MIMIC-CXR mirror.

| System             | Retrieval                 | Generator                        |
|--------------------|---------------------------|----------------------------------|
| **ColPali-RAG**    | ColPali v1.3 (MaxSim, k=5)| `google/medgemma-1.5-4b-it`      |
| **MedGemma-only**  | none (zero-shot)          | `google/medgemma-1.5-4b-it`      |
| *(optional)* Fused | ColPali → MedGemma + text | `google/medgemma-1.5-4b-it`      |

Both modes:
* **Mode A — report generation** : image → full FINDINGS+IMPRESSION report.
* **Mode B — VQA**                : image + question → concise answer.

## Repo layout

```
data_prep/    # Kaggle download, image preprocess, patient-level split, VQA generation
retrieval/    # ColPali indexing + MaxSim search
generation/   # MedGemma multimodal wrapper
rag/          # End-to-end pipeline (Mode A + Mode B, with/without RAG)
eval/         # BLEU/ROUGE/METEOR/BERTScore, CheXbert F1, VQA F1
notebooks/    # Exploration + final results notebook
prompts/      # Diff-friendly prompt templates (vqa-gen, rag, baseline)
configs/      # config.yaml
scripts/      # run.sh / run.ps1 / Makefile entrypoints
report/       # final PDF + sources
demo/         # demo video + screenshots
```

## Quick start

### 1. Environment
```bash
python -m venv .venv
source .venv/bin/activate                      # or .venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
```

### 2. Credentials
```bash
# Kaggle (data download)
mkdir -p ~/.kaggle && mv path/to/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

# Hugging Face (MedGemma is gated — accept the license once on the model page)
huggingface-cli login

# LLM-as-a-judge for VQA dataset construction (pick one)
export ANTHROPIC_API_KEY=...
# or:  export OPENAI_API_KEY=...
```

### 3. Reproduce everything

**Recommended for laptops without a 16 GB+ GPU: headless Kaggle runs from VSCode.**
Set up Kaggle credentials once, then everything runs from your terminal:

```bash
pip install kaggle
# put kaggle.json in ~/.kaggle/  (Windows: %USERPROFILE%\.kaggle\kaggle.json)
python scripts/run_on_kaggle.py        # pushes notebook, watches it, downloads outputs
```

You edit code in VSCode, `git push` to GitHub, then run the script. Kaggle
clones from GitHub, runs on a free T4, and writes everything back to
`outputs/`. The Kaggle web UI is only touched once at setup (to attach
the `HF_TOKEN` and `ANTHROPIC_API_KEY` secrets — see `kaggle/README.md`).

**Local with a GPU (24 GB+):**


```bash
bash scripts/run.sh                    # Linux/macOS
# or:
powershell -File scripts/run.ps1       # Windows
# or:
make all                               # via Makefile
```

The driver runs, in order:

```
data_prep.download_kaggle   →  data/mimic-cxr/
data_prep.preprocess        →  resized JPGs + reports.csv
data_prep.split             →  patient-level train / test parquet
data_prep.build_vqa_dataset →  data/vqa/mimic_vqa.parquet
retrieval.colpali_index     →  artifacts/colpali_index.pt
rag.run_mode_a              →  outputs/mode_a_{rag,baseline}.jsonl
rag.run_mode_b              →  outputs/mode_b_{rag,baseline}.jsonl
eval.metrics_report         →  outputs/metrics_mode_a.json
eval.metrics_vqa            →  outputs/metrics_mode_b.json
```

Single steps:
```bash
python -m data_prep.split            --config configs/config.yaml
python -m retrieval.colpali_index    --config configs/config.yaml
python -m rag.run_mode_a system=rag  --config configs/config.yaml
python -m eval.metrics_report        --config configs/config.yaml
```

## Hardware

Designed for a **single 24 GB GPU** (e.g., RTX 4090 / L4 / A10G):

* ColPali indexing: ~4 ms / image, ~6 GB.
* MedGemma-4B in 4-bit (`bitsandbytes`): ~5 GB VRAM at inference.
* Corpus capped at 20 000 images by default
  (`dataset.max_train_corpus`); raise if you have more VRAM/time.

## Reproducibility

* `dataset.seed = 42` everywhere.
* Patient-level split — no `subject_id` appears in both train and test.
* All prompts are versioned files under `prompts/`.
* Every run writes its full resolved config + git SHA to its output dir.

## Ethics / PHI

* MIMIC-CXR is de-identified credentialed data. We use the Kaggle mirror
  for academic coursework only and never commit images or reports.
  `.gitignore` excludes `data/` and `artifacts/`.
* This system is **not** a clinical tool. Any output requires
  radiologist review.

## Citations

See `report/references.bib`. Key works:

* Faysse et al. 2024 — *ColPali: Efficient Document Retrieval with Vision
  Language Models*. arXiv:2407.01449.
* Google Health 2025 — *MedGemma Technical Report*.
* Ranjit et al. 2023 — *Retrieval Augmented Chest X-ray Report Generation
  using OpenAI GPT models*. arXiv:2305.03660.
* Bae et al. 2023 — *EHRXQA / MIMIC-CXR-VQA*.
* Johnson et al. 2019 — *MIMIC-CXR-JPG*.
* Irvin et al. 2019 — *CheXpert / CheXbert*.
