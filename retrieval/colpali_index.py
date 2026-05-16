"""Build a ColPali multi-vector index over the training X-ray corpus.

Output (single file):
    artifacts/colpali_index.pt
      {
        "embeddings": list[Tensor[Q_i, D]]   # per-image multi-vector
        "metadata":   list[dict] (study_id, subject_id, image_path, text)
        "model_name": str
      }

We keep everything in CPU memory; for the 20 k cap in `config.yaml`
this is ~3 GB at fp16 (≈800 patch tokens × 128-dim × 2 bytes × 20 k).
That fits easily on a 32 GB host. Swap to Qdrant if you cross 100 k.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from data_prep.common import Paths, ensure_dir, get_logger, load_config

log = get_logger("colpali_index")


def _load_model(model_name: str, device: str, use_4bit: bool):
    # colpali-engine ≥ 0.3 exposes ColPali + processor as standard HF classes.
    from colpali_engine.models import ColPali, ColPaliProcessor

    log.info("Loading ColPali model: %s", model_name)
    quant_args = {}
    if use_4bit:
        from transformers import BitsAndBytesConfig
        quant_args = dict(
            quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                                                   bnb_4bit_compute_dtype=torch.float16),
            device_map="auto",
        )
    else:
        quant_args = dict(torch_dtype=torch.float16, device_map=device)

    model = ColPali.from_pretrained(model_name, **quant_args).eval()
    processor = ColPaliProcessor.from_pretrained(model_name)
    return model, processor


@torch.inference_mode()
def _embed_images(model, processor, image_paths: list[Path], batch_size: int) -> list[torch.Tensor]:
    out: list[torch.Tensor] = []
    n_batches = math.ceil(len(image_paths) / batch_size)
    for b in tqdm(range(n_batches), desc="ColPali embed"):
        batch_paths = image_paths[b * batch_size:(b + 1) * batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor.process_images(images).to(model.device)
        emb = model(**inputs)  # (B, Q, D)
        # Keep one tensor per image so we can drop padding rows later.
        # colpali-engine handles variable-length queries; for images each
        # has the same Q so we can split along batch dim directly.
        for i in range(emb.shape[0]):
            out.append(emb[i].detach().to("cpu", dtype=torch.float16).contiguous())
    return out


def build_index(cfg: dict, paths: Paths) -> Path:
    train_parquet = paths.splits_dir / "train.parquet"
    if not train_parquet.exists():
        raise FileNotFoundError(f"{train_parquet} not found. Run data_prep.split first.")
    df = pd.read_parquet(train_parquet)
    image_paths = [Path(p) for p in df["image_path"].tolist()]

    model, processor = _load_model(
        cfg["models"]["colpali"],
        device=cfg["generation"]["device"],
        use_4bit=cfg["retrieval"]["use_4bit"],
    )

    embeddings = _embed_images(model, processor, image_paths,
                               batch_size=cfg["retrieval"]["batch_size"])

    metadata = df[["study_id", "subject_id", "image_path", "text"]].to_dict("records")

    out_path = paths.artifacts_dir / "colpali_index.pt"
    ensure_dir(out_path.parent)
    torch.save({
        "embeddings": embeddings,
        "metadata":   metadata,
        "model_name": cfg["models"]["colpali"],
    }, out_path)
    log.info("Saved index with %d items -> %s", len(embeddings), out_path)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = Paths.from_config(cfg)
    build_index(cfg, paths)


if __name__ == "__main__":
    main()
