# Windows equivalent of scripts/run.sh — runs the full pipeline.
$ErrorActionPreference = 'Stop'
$CFG = if ($env:CFG) { $env:CFG } else { 'configs/config.yaml' }

function Step($n, $msg) { Write-Host ">>> [$n/8] $msg" -ForegroundColor Cyan }

Step 1 'Download MIMIC-CXR (Kaggle)';                        python -m data_prep.download_kaggle    --config $CFG
Step 2 'Preprocess images + reports';                        python -m data_prep.preprocess         --config $CFG
Step 3 'Patient-level train/test split';                     python -m data_prep.split              --config $CFG
Step 4 'Build VQA dataset (LLM judge)';                      python -m data_prep.build_vqa_dataset  --config $CFG
Step 5 'Split VQA by patient';                               python -m data_prep.split_vqa          --config $CFG
Step 6 'Build ColPali index over training corpus';           python -m retrieval.colpali_index      --config $CFG
Step 7 'Run Mode A (report) — RAG + baseline';               python -m rag.run_mode_a               --config $CFG --system both
Step 7 'Run Mode B (VQA)    — RAG + baseline';               python -m rag.run_mode_b               --config $CFG --system both
Step 8 'Evaluate';                                           python -m eval.metrics_report          --config $CFG
                                                             python -m eval.metrics_vqa             --config $CFG
                                                             python -m eval.qualitative_dump        --config $CFG

Write-Host 'Done. See outputs/metrics_mode_a.json, outputs/metrics_mode_b.json.' -ForegroundColor Green
