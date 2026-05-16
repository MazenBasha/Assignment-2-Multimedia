"""CLIP-based image retrieval index — local substitute for ColPali.

Single-vector cosine similarity instead of MaxSim late interaction. Loses
some retrieval precision but runs comfortably on CPU and needs no HF auth.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from data_prep.common import Paths, ensure_dir, get_logger, load_config

log = get_logger("clip_index")


def _load_model(model_name: str, device: str):
    from transformers import CLIPModel, CLIPProcessor
    log.info("Loading CLIP: %s", model_name)
    model = CLIPModel.from_pretrained(model_name).eval().to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor


@torch.inference_mode()
def _embed_images(model, processor, image_paths, batch_size: int, device: str) -> torch.Tensor:
    embs = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="CLIP embed"):
        batch_paths = image_paths[i:i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt").to(device)
        feats = model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        embs.append(feats.cpu())
    return torch.cat(embs, dim=0)


def build_index(cfg: dict, paths: Paths) -> Path:
    train_parquet = paths.splits_dir / "train.parquet"
    if not train_parquet.exists():
        raise FileNotFoundError(f"{train_parquet} not found. Run data_prep.split first.")
    df = pd.read_parquet(train_parquet)
    image_paths = [Path(p) for p in df["image_path"].tolist()]
    log.info("Indexing %d images with CLIP", len(image_paths))

    device = cfg["generation"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model, processor = _load_model(cfg["retrieval"]["clip_model"], device=device)
    embs = _embed_images(model, processor, image_paths,
                         batch_size=cfg["retrieval"]["batch_size"], device=device)

    metadata = df[["study_id", "subject_id", "image_path", "text"]].to_dict("records")
    out_path = paths.artifacts_dir / "clip_index.pt"
    ensure_dir(out_path.parent)
    torch.save({
        "embeddings": embs,
        "metadata":   metadata,
        "model_name": cfg["retrieval"]["clip_model"],
    }, out_path)
    log.info("Saved CLIP index with %d items -> %s", len(embs), out_path)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/local.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = Paths.from_config(cfg)
    build_index(cfg, paths)


if __name__ == "__main__":
    main()
