# Prompt templates

All prompts live in this directory as plain `.txt` so they are diff-friendly
and easy to iterate without touching code. `{placeholder}` fields are
filled in by Python `str.format(...)`.

| File                              | Used by                                   | Placeholders                                |
|-----------------------------------|-------------------------------------------|---------------------------------------------|
| `vqa_generation.txt`              | `data_prep/build_vqa_dataset.py`          | `n_pairs`, `report_id`, `report_text`        |
| `rag_report_generation.txt`       | `rag/rag_pipeline.py` (Mode A, RAG)       | `k`, `retrieved_reports`                     |
| `rag_vqa.txt`                     | `rag/rag_pipeline.py` (Mode B, RAG)       | `retrieved_reports`, `question`              |
| `baseline_report_generation.txt`  | `rag/rag_pipeline.py` (Mode A, no-RAG)    | *(none — image only)*                        |
| `baseline_vqa.txt`                | `rag/rag_pipeline.py` (Mode B, no-RAG)    | `question`                                   |

## Design choices

* **Grounding**: every prompt explicitly tells the model that retrieved
  reports are reference context, not ground truth — they may describe a
  different patient. This mitigates the "copy the nearest neighbour"
  failure mode reported by Ranjit et al., arXiv:2305.03660.
* **Structure**: report-generation prompts force a `FINDINGS:` /
  `IMPRESSION:` structure to match MIMIC-CXR style and make CheXbert
  label extraction reliable.
* **Refusal escape hatch**: VQA prompts include an exact-string refusal
  (`"Cannot be determined from the image."`) so token-F1 scoring is not
  punished for sensible abstention.
* **Question taxonomy**: VQA generation uses the five categories from the
  MIMIC-CXR-VQA paper (Bae et al., 2023) and the ReXVQA benchmark.
