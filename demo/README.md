# Demo

Required by the assignment: a 3–7 minute screen recording showing:
1. The repo running end-to-end.
2. **Mode A** — feed an unseen X-ray, show top-k retrieved reports, show
   generated report vs. ground truth.
3. **Mode B** — feed an X-ray + a question, show the answer.
4. One ColPali-RAG vs. MedGemma-only side-by-side comparison.
5. Brief recap of the key metric deltas from `outputs/metrics_*.json`.

Suggested capture:
* OBS Studio, 1080p30, system audio off, mic on for narration.
* Show the terminal command + `outputs/` JSONL files being written.
* Use the cherry-picked qualitative examples saved by
  `eval/qualitative_dump.py`.

Place the final file as `demo/demo.mp4` (or `demo/link.txt` with a
private YouTube unlisted link if it exceeds GitHub's 100 MB limit).
