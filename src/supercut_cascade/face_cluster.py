# Copyright 2024 supercut-judge-cascade contributors
# SPDX-License-Identifier: Apache-2.0
"""ArcFace embedding extraction and k-means clustering.

Requires the ``cluster`` extra::

    pip install "supercut-judge-cascade[cluster]"

which installs ``insightface`` and ``scikit-learn``.

Public API
----------
:func:`load_recognizer`   — initialise InsightFace ``buffalo_l`` with recognition.
:func:`embed_image`       — extract ArcFace 512-D L2-normalised embedding.
:func:`face_cluster`      — k-means cluster an embedding matrix.
:func:`make_cluster_labels` — build a cluster-id → label mapping.
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from .exceptions import BackendUnavailableError

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model initialisation
# ---------------------------------------------------------------------------


def load_recognizer(
    providers: Sequence[str] | None = None,
    det_size: tuple[int, int] = (640, 640),
    det_thresh: float = 0.35,
) -> "insightface.app.FaceAnalysis":  # type: ignore[name-defined]
    """Initialise InsightFace ``buffalo_l`` with the recognition module.

    Parameters
    ----------
    providers:
        ONNX Runtime execution providers in priority order.  When ``None``,
        tries ``CUDAExecutionProvider`` first, then ``CPUExecutionProvider``.
    det_size:
        Detector input size ``(width, height)``.
    det_thresh:
        Face-detection confidence threshold (0–1).

    Returns
    -------
    A prepared ``FaceAnalysis`` instance.

    Raises
    ------
    BackendUnavailableError
        If ``insightface`` is not installed.
    """
    try:
        from insightface.app import FaceAnalysis  # type: ignore[import]
    except ImportError as exc:
        raise BackendUnavailableError(
            "insightface is required for face_cluster. "
            "Install it with: pip install 'supercut-judge-cascade[cluster]'"
        ) from exc

    if providers is None:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    app = FaceAnalysis(
        name="buffalo_l",
        providers=list(providers),
        allowed_modules=["detection", "recognition"],
    )
    ctx_id = 0 if "CUDAExecutionProvider" in providers else -1
    app.prepare(ctx_id=ctx_id, det_size=det_size, det_thresh=det_thresh)
    log.info("InsightFace ready  ctx_id=%d  providers=%s", ctx_id, list(providers))
    return app


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def embed_image(
    app: "insightface.app.FaceAnalysis",  # type: ignore[name-defined]
    img_bgr: np.ndarray,
) -> np.ndarray | None:
    """Extract an ArcFace 512-D embedding from the largest detected face.

    Parameters
    ----------
    app:
        Prepared ``FaceAnalysis`` instance from :func:`load_recognizer`.
    img_bgr:
        BGR uint8 image.

    Returns
    -------
    L2-normalised float32 array of shape ``(512,)``, or ``None`` if no face
    is detected or embedding extraction fails.
    """
    try:
        faces = app.get(img_bgr)
    except Exception as exc:
        log.debug("face detection failed: %s", exc)
        return None

    if not faces:
        return None

    biggest = max(
        faces,
        key=lambda f: (f.bbox[3] - f.bbox[1]) * (f.bbox[2] - f.bbox[0]),
    )
    emb = getattr(biggest, "normed_embedding", None)
    if emb is None:
        emb = getattr(biggest, "embedding", None)
    if emb is None:
        log.debug("face has no embedding attribute")
        return None

    emb = np.asarray(emb, dtype=np.float32)
    norm = float(np.linalg.norm(emb))
    if norm < 1e-8:
        log.debug("zero-norm embedding, skipping")
        return None
    return emb / norm


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def face_cluster(
    embeddings: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
) -> np.ndarray:
    """K-means cluster L2-normalised ArcFace embeddings.

    Parameters
    ----------
    embeddings:
        Float32 array of shape ``(N, 512)``.  Rows must be L2-normalised
        (guaranteed by :func:`embed_image`).
    n_clusters:
        Number of clusters ``k``.  Automatically clamped to
        ``min(n_clusters, len(embeddings))`` to avoid empty-cluster errors.
    random_state:
        Random seed for reproducible results.

    Returns
    -------
    Integer label array of shape ``(N,)`` in ``[0, k)``.

    Raises
    ------
    BackendUnavailableError
        If ``scikit-learn`` is not installed.
    ValueError
        If ``embeddings`` is empty.
    """
    if len(embeddings) == 0:
        raise ValueError("embeddings array must not be empty")

    try:
        from sklearn.cluster import KMeans  # type: ignore[import]
    except ImportError as exc:
        raise BackendUnavailableError(
            "scikit-learn is required for face_cluster. "
            "Install it with: pip install 'supercut-judge-cascade[cluster]'"
        ) from exc

    n_actual = len(embeddings)
    k = min(n_clusters, n_actual)
    if k < n_clusters:
        log.warning(
            "only %d embeddings available; reducing n_clusters from %d to %d",
            n_actual, n_clusters, k,
        )

    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels: np.ndarray = km.fit_predict(embeddings)
    log.info("KMeans fit  n=%d  k=%d  inertia=%.2f", n_actual, k, float(km.inertia_))
    return labels.astype(np.int32)


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------


def make_cluster_labels(
    n_clusters: int,
    seed_map: dict[int, str] | None = None,
) -> dict[int, str]:
    """Build a ``{cluster_id: label}`` mapping.

    Parameters
    ----------
    n_clusters:
        Total number of clusters.
    seed_map:
        Optional overrides mapping ``cluster_id`` to a human-readable name.
        Unknown cluster IDs receive ``"unknown_{id}"``.

    Returns
    -------
    Dict mapping every cluster id in ``[0, n_clusters)`` to a string label.
    """
    labels: dict[int, str] = {i: f"unknown_{i}" for i in range(n_clusters)}
    if seed_map:
        for cid, name in seed_map.items():
            if 0 <= cid < n_clusters:
                labels[cid] = name
    return labels


__all__ = [
    "load_recognizer",
    "embed_image",
    "face_cluster",
    "make_cluster_labels",
]
