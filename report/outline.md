# Report outline — DSAI 413 Assignment 2

Target length: ~10–15 pages PDF. The numbered sections map 1-to-1 to the
deliverables described in the brief.

## 1. Abstract (½ page)
Two systems, two modes, key metric deltas in one paragraph.

## 2. Medical & radiology background (1–1.5 pages)
* Anatomy review of a PA/AP chest X-ray. Sources: *Radiology Assistant —
  Chest X-ray: Basic Interpretation*; the Chest X-ray Intro PDF supplied
  with the assignment.
* The 14 CheXpert findings and why they matter clinically.
* Structured reporting: FINDINGS vs. IMPRESSION.

## 3. Multimodal learning + RAG background (1–1.5 pages)
* Vision-language models: SigLIP → PaliGemma → MedGemma.
* Retrieval-augmented generation: dense bi-encoders vs. late-interaction
  (ColBERT family).
* ColPali specifically: PaliGemma-3B backbone, multi-vector page
  embeddings, MaxSim scoring.

## 4. Methods (2–3 pages)
### 4.1 Dataset
* Kaggle `simhadrisadaram/mimic-cxr-dataset`. Schema. Patient-level split.
### 4.2 ColPali-RAG pipeline (diagram)
  query image → ColPali image encoder → MaxSim(top-k) → retrieved reports
  → MedGemma(image, retrieved_reports) → answer / report
### 4.3 MedGemma-only baseline
* Zero-shot, same generation hyperparameters.
### 4.4 (Optional) Fused variant: RAG + n-shot text.

## 5. QA dataset construction (1–1.5 pages)
* Source: MIMIC-CXR `text` column with a section parser (FINDINGS,
  IMPRESSION, INDICATION).
* Generation prompt (`prompts/vqa_generation.txt`) reproduced verbatim.
* Generator: Claude Opus 4.7 used as the LLM judge (cite, with rate-limit
  / cost). Alternatives tested: MedGemma-27B-text, Llama-3-Instruct.
* Question-type taxonomy from MIMIC-CXR-VQA + ReXVQA.
* Filtering: dedup, length filter, JSON-parse filter, rationale must be
  a substring of the source report.
* **Human review**: spot-check {n_human_review} pairs by sampling
  ~2 % across types; report agreement.

## 6. Results
### 6.1 Mode A — report generation
| Method      | BLEU-1 | BLEU-4 | ROUGE-L | METEOR | BERTScore-F1 | CheXbert macro-F1 |
|-------------|--------|--------|---------|--------|--------------|-------------------|
| ColPali-RAG |        |        |         |        |              |                   |
| MedGemma    |        |        |         |        |              |                   |
| Δ (RAG−base)|        |        |         |        |              |                   |

5+ qualitative side-by-side examples (ground truth | RAG | baseline).

### 6.2 Mode B — VQA
* Exact-match, token-F1, accuracy per question type (presence /
  location / severity / differential / description).
* Confusion analysis on yes/no questions.

## 7. Failure-mode analysis (1–2 pages)
* Hallucinated devices / catheters (MedGemma-only specific?).
* Retrieval misses: when top-k neighbours describe a different
  pathology and the RAG model copies it.
* Verbosity / impression bloat.

## 8. Limitations and ethics (½ page)
* MIMIC-CXR PHI risks; no clinical deployment.
* Distribution shift: indoor portables vs. PA standing.
* Caveats of automatic metrics for medical reports
  (BLEU ≠ clinical correctness — hence CheXbert).

## 9. Conclusion + future work (½ page)
* When retrieval helps (rare findings, vocab consistency) and when it
  hurts (common cases, copy-paste).
* Next steps: fine-tuning MedGemma on the constructed VQA set; switching
  the retriever to ColQwen2.5.

## 10. References
Auto-generated from `references.bib`.

---

### Required figures
1. System diagram (ColPali → MedGemma)
2. Question-type histogram of the constructed VQA set
3. Per-finding CheXbert F1 bar chart, RAG vs. baseline
4. 1–2 qualitative comparisons (image + GT report + RAG + baseline)
