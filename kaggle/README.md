# Running this project on Kaggle Notebooks (free T4 16 GB)

Why Kaggle: free GPU, MIMIC-CXR mounts in one click, 9-hour session, no
local install pain.

## One-time prep

1. **Push this repo to GitHub** (private is fine — Kaggle just needs a
   git URL). Let's call it `https://github.com/<you>/cxr-rag`.
2. **Accept the MedGemma license** at
   https://huggingface.co/google/medgemma-1.5-4b-it (must be signed in).
3. **Get an HF access token** at https://huggingface.co/settings/tokens
   (Read scope is enough since MedGemma is public-after-gate).
4. **Get an Anthropic API key** (or OpenAI) for the VQA dataset
   constructor. Cheaper to use `claude-haiku-4-5-20251001` for the
   dataset build — `configs/kaggle.yaml` defaults to it.

## In the Kaggle Notebook UI

1. **New Notebook → Add-ons → Internet** : ON.
2. **Accelerator** : `GPU T4 x1` (or P100). Persistence : "Files only".
3. **Add data**: search "mimic-cxr-dataset" → add
   `simhadrisadaram/mimic-cxr-dataset`.
4. **Add secrets** (Add-ons → Secrets):
   - `HF_TOKEN`            : your Hugging Face token
   - `ANTHROPIC_API_KEY`   : your Anthropic key
5. **Paste-import** `kaggle/notebook.ipynb` (or just upload it).
6. **Edit the first code cell** to point at your GitHub URL.
7. **Run All**.

Expected wall-clock on T4 with the defaults in `configs/kaggle.yaml`:

| Step                                       | Time     |
|--------------------------------------------|----------|
| Install requirements                       | 3–5 min  |
| Preprocess + split (2 000 images)          | 2–4 min  |
| Build VQA dataset (500 reports, Haiku 4.5) | 4–8 min  |
| ColPali index (2 000 images)               | 5–10 min |
| Mode A inference (50 × {RAG, baseline})    | 15–25 min|
| Mode B inference (200 × {RAG, baseline})   | 15–25 min|
| Eval (BLEU/ROUGE/METEOR/BERTScore/CheXbert)| 5–10 min |
| **Total**                                  | ~60–90 min |

After Run All, all results land in `/kaggle/working/outputs/`:
- `metrics_mode_a.json`, `metrics_mode_b.json`
- `mode_{a,b}_{rag,baseline}.jsonl`
- `qualitative_mode_a.md`, `qualitative_mode_b.md`

Download the notebook output to ship `outputs/` back into the repo and
use those numbers in `report/`.
