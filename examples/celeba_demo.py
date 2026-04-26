"""CelebA-HQ demo: IdentityFilter with real face images.

This script is NOT self-contained — you must download CelebA-HQ separately
(the dataset is not included in this repository).

Download instructions
---------------------
CelebA-HQ is hosted on Google Drive.  One common method:

    # Install gdown
    pip install gdown

    # Download and unzip (approx 3 GB for 30 000-image 256px version)
    gdown --fuzzy "https://drive.google.com/file/d/1badu11NqxGf6qM3PTTooQDJvQbejgbTv/view"
    unzip CelebA-HQ-img.zip -d celeba_hq

Note: CelebA-HQ is for **non-commercial research only**.  See the CelebA
license: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

Requirements
------------
    pip install supercut-judge-cascade[arcface]
    insightface-cli model.download buffalo_l

Usage::

    python examples/celeba_demo.py --dataset-dir /path/to/celeba_hq --ref-ids 1 2 3

Arguments
---------
--dataset-dir   Path to the CelebA-HQ image directory (contains *.jpg)
--ref-ids       Space-separated integer image IDs to use as reference (1-indexed)
--query-ids     Space-separated integer image IDs to query (default: 4 5 6 7 8)
--threshold     Cosine similarity threshold (default 0.40)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("celeba_demo")


def load_image(path: Path) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("opencv-python required: pip install opencv-python") from exc
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def main() -> int:
    parser = argparse.ArgumentParser(description="CelebA-HQ IdentityFilter demo")
    parser.add_argument("--dataset-dir", required=True, type=Path,
                        help="Path to CelebA-HQ image directory")
    parser.add_argument("--ref-ids", nargs="+", type=int, default=[1, 2, 3],
                        help="Image IDs to use as reference (1-indexed)")
    parser.add_argument("--query-ids", nargs="+", type=int, default=[4, 5, 6, 7, 8],
                        help="Image IDs to query against reference")
    parser.add_argument("--threshold", type=float, default=0.40,
                        help="Cosine similarity threshold (default 0.40)")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    if not dataset_dir.is_dir():
        log.error("Dataset directory not found: %s", dataset_dir)
        log.error("Download CelebA-HQ first (see module docstring for instructions).")
        return 1

    # Import pipeline components
    try:
        from supercut_cascade.arcface import ArcFaceEmbedder
        from supercut_cascade.identity_filter import (
            IdentityFilter,
            build_reference_embeddings,
        )
    except ImportError as exc:
        log.error("arcface extra not installed: %s", exc)
        log.error("Install with: pip install 'supercut-judge-cascade[arcface]'")
        log.error("Then download weights: insightface-cli model.download buffalo_l")
        return 1

    # Collect reference image paths
    ref_paths: list[Path] = []
    for rid in args.ref_ids:
        # CelebA-HQ naming: 00001.jpg, 00002.jpg, ...
        p = dataset_dir / f"{rid:05d}.jpg"
        if not p.exists():
            log.warning("Reference image not found (skipping): %s", p)
        else:
            ref_paths.append(p)

    if not ref_paths:
        log.error("No reference images found in %s", dataset_dir)
        return 1

    log.info("Loading ArcFaceEmbedder (buffalo_l) ...")
    log.warning("buffalo_l weights are for NON-COMMERCIAL use only.")
    embedder = ArcFaceEmbedder()

    log.info("Building reference embeddings from %d images ...", len(ref_paths))
    ref_embs = build_reference_embeddings(
        image_paths=ref_paths,
        embedder=embedder,
        target_label="REFERENCE_SUBJECT",
    )
    if ref_embs is None or len(ref_embs) == 0:
        log.error("No face detected in any reference image.")
        return 1

    log.info("Reference embeddings: shape=%s", ref_embs.shape)
    filt = IdentityFilter(reference_embeddings=ref_embs, threshold=args.threshold)

    # Query images
    log.info("Querying %d images ...", len(args.query_ids))
    for qid in args.query_ids:
        p = dataset_dir / f"{qid:05d}.jpg"
        if not p.exists():
            log.warning("Query image not found (skipping): %s", p)
            continue
        img = load_image(p)
        results = embedder.get_embeddings(img)
        if not results:
            log.info("  %05d.jpg: no face detected", qid)
            continue
        best_emb = results[0]  # primary face
        is_match, sim = filt.is_target(best_emb)
        log.info("  %05d.jpg: match=%-5s  cosine_sim=%.4f", qid, is_match, sim)

    return 0


if __name__ == "__main__":
    sys.exit(main())
