# Status — while you were away

What I did, what I couldn't do, and exactly what to click when you're back.

## What's done

| Area | Status |
|---|---|
| Repo skeleton (`data_prep/`, `retrieval/`, `generation/`, `rag/`, `eval/`, `prompts/`, `tests/`) | Complete |
| Prompt templates (VQA gen, RAG report, RAG VQA, baseline report, baseline VQA) | Complete |
| Patient-level train/test split | Complete |
| Kaggle-mirror preprocessor for `simhadrisadaram/mimic-cxr-dataset` | Complete |
| ColPali indexing + MaxSim search | Complete |
| MedGemma multimodal wrapper (4-bit option) | Complete |
| End-to-end RAG pipeline (Mode A + Mode B, RAG + baseline) | Complete |
| Metrics: BLEU-1..4, ROUGE-L, METEOR, BERTScore, CheXbert, EM, token-F1, per-type accuracy | Complete |
| `scripts/run_on_kaggle.py` for VSCode → headless Kaggle run | Complete |
| Tests (17 passing) | Complete |
| Notebook rewrite + hardening (idempotent, soft-fails VQA gen, stages outputs at kernel root) | Complete |
| GitHub repo (`MazenBasha/Assignment-2-Multimedia`, branch `main`) | Pushed: `bae382b` + this commit |

## What I couldn't do

Two hard blockers that only **you** can resolve. Both take ~2 minutes once you're at the laptop.

### Blocker 1 — No `kaggle.json` on this machine

I can't authenticate to your Kaggle account, so I can't run `scripts/run_on_kaggle.py`. Fix:

1. https://www.kaggle.com → click your profile photo → **Account** → **Create New Token**. A `kaggle.json` downloads.
2. Move it to:
   ```
   %USERPROFILE%\.kaggle\kaggle.json
   ```
   (create the `.kaggle` folder if missing)

### Blocker 2 — `HF_TOKEN` Kaggle secret + MedGemma license

The MedGemma model is gated on Hugging Face. Without these the inference cells will 403.

1. https://huggingface.co/google/medgemma-1.5-4b-it → click **"Agree and access repository"**.
2. https://huggingface.co/settings/tokens → **New token** → type = **Read** → copy the `hf_...`.
3. On any Kaggle notebook page → **Add-ons → Secrets → Add new secret**:
   - Label: `HF_TOKEN`
   - Value: paste the token
   - Tick the **Attached** checkbox.

(You already set up `ANTHROPIC_API_KEY` earlier — good.)

## How to actually run it when you're back

```powershell
cd "E:\Assignment 2 Multimedia"
python scripts/run_on_kaggle.py
```

The script will:
1. Push `kaggle/notebook.ipynb` as a private kernel with GPU + Internet + MIMIC-CXR attached.
2. Poll Kaggle every 30 s until the run finishes (~60–90 min).
3. Download `metrics_mode_*.json`, `mode_*.jsonl`, `qualitative_*.md` into your local `outputs/`.

If something fails mid-way, the notebook soft-fails individual stages (e.g. if Claude API breaks, Mode B is skipped but Mode A still completes). Re-running is idempotent — it `git pull`s instead of re-cloning.

## What to check after the run

Open these files in VSCode:
- `outputs/metrics_mode_a.json` — should have `rag`, `baseline`, `delta_rag_minus_baseline` blocks.
- `outputs/metrics_mode_b.json` — same structure plus per-question-type breakdowns.
- `outputs/qualitative_mode_a.md` — handful of side-by-side examples for the report.
- `outputs/qualitative_mode_b.md` — same for VQA.

Copy the numbers into `report/outline.md` → render the PDF.

## Known small gotchas

1. **VSCode line-ending warnings on every git command** — harmless, Windows CRLF vs Unix LF. Ignore.
2. **Hugging Face cache will be re-downloaded each Kaggle run** (~9 GB: 6 GB ColPali + 3 GB MedGemma). Kaggle session disk is ephemeral; nothing to do.
3. **Anthropic API cost**: VQA generation calls Claude Haiku ~500 × 1 = ~500 requests. Should be well under $0.50.

## Commits in this session

- `e5fdb96` — Initial commit (full codebase)
- `9957fed` — Add `preprocess_kaggle.py` for the Kaggle MIMIC mirror schema + config fix
- `bae382b` — Add `scripts/run_on_kaggle.py` for headless VSCode workflow
- `<this commit>` — Notebook rewrite (clean cells, idempotent, soft-fail VQA, output staging) + STATUS.md

Repo is at `https://github.com/MazenBasha/Assignment-2-Multimedia`.

## What I'd suggest as the very first thing you do

1. Drop `kaggle.json` into `%USERPROFILE%\.kaggle\`
2. Accept the MedGemma license + add `HF_TOKEN` Kaggle secret
3. `python scripts/run_on_kaggle.py`
4. Walk away again for ~75 minutes; check `outputs/` when you're back.
