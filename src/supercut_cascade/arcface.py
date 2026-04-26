# Copyright 2024 supercut-judge-cascade contributors
# SPDX-License-Identifier: Apache-2.0
"""ArcFace 512-D face embedding wrapper backed by InsightFace.

Legal notices
-------------
* The ``buffalo_l`` weights shipped by InsightFace are trained on MS1MV2 and
  are licensed for **NON-COMMERCIAL use only**.  Users must download the weights
  separately::

      insightface-cli model.download buffalo_l

  The weights MUST NOT be bundled with this package or committed to any
  repository.

* **GDPR Art. 9 / BIPA compliance**: Biometric embeddings derived from face
  images constitute sensitive personal data in many jurisdictions.  It is the
  **user's sole responsibility** to obtain necessary consents, honour data-
  subject rights, and comply with all applicable privacy laws (including the EU
  General Data Protection Regulation Article 9 and the Illinois Biometric
  Information Privacy Act) before processing face images with this module.

Install
-------
To enable this module install the optional extra::

    pip install supercut-judge-cascade[arcface]
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from supercut_cascade.exceptions import BackendUnavailableError

if TYPE_CHECKING:
    pass  # insightface imported lazily to keep the base install light

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "buffalo_l"
_DEFAULT_DET_SIZE = (640, 640)


class ArcFaceEmbedder:
    """ArcFace 512-D embedding wrapper backed by InsightFace.

    Parameters
    ----------
    model_name : str
        InsightFace model pack name.  Defaults to ``"buffalo_l"``.
        **Non-commercial only** — see module docstring.
    det_size : tuple[int, int]
        Detection input resolution passed to ``FaceAnalysis.prepare``.
        Larger values improve recall at the cost of speed.
    det_thresh : float
        Face-detection confidence threshold (0–1).
    providers : list[str] | None
        ONNX Runtime execution providers, e.g.
        ``["CUDAExecutionProvider", "CPUExecutionProvider"]``.
        Defaults to CPU-only.

    Raises
    ------
    BackendUnavailableError
        If ``insightface`` is not installed.

    Notes
    -----
    * Weights (``buffalo_l``) are MS1MV2-licensed, **NON-COMMERCIAL only**.
    * Users must run ``insightface-cli model.download buffalo_l`` separately.
    * **GDPR Art. 9 / BIPA**: Biometric data compliance is the user's
      responsibility.  Do not process face images without proper consent.

    Examples
    --------
    >>> embedder = ArcFaceEmbedder()
    >>> emb = embedder.extract(frame_bgr)  # np.ndarray shape (512,) or None
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        det_size: tuple[int, int] = _DEFAULT_DET_SIZE,
        det_thresh: float = 0.3,
        providers: list[str] | None = None,
    ) -> None:
        try:
            from insightface.app import FaceAnalysis  # noqa: PLC0415
        except ImportError as exc:
            raise BackendUnavailableError(
                "insightface is required for ArcFaceEmbedder.\n"
                "Install it with:  pip install supercut-judge-cascade[arcface]\n"
                "Then download the weights:  insightface-cli model.download buffalo_l\n"
                "\nIMPORTANT — buffalo_l weights are NON-COMMERCIAL only (MS1MV2).\n"
                "GDPR Art.9 / BIPA: Ensure biometric data processing is lawful."
            ) from exc

        if providers is None:
            providers = ["CPUExecutionProvider"]

        logger.debug(
            "Initialising InsightFace FaceAnalysis(name=%r, providers=%r)",
            model_name,
            providers,
        )
        self._app = FaceAnalysis(name=model_name, providers=providers)
        self._app.prepare(ctx_id=0, det_thresh=det_thresh, det_size=det_size)
        logger.info("ArcFaceEmbedder ready (model=%s, det_size=%s)", model_name, det_size)

    def extract(self, image: np.ndarray) -> np.ndarray | None:
        """Extract a 512-D L2-normalised ArcFace embedding from *image*.

        Parameters
        ----------
        image : np.ndarray
            BGR image (H × W × 3, uint8), as returned by ``cv2.imread`` or
            ``cv2.VideoCapture.read``.

        Returns
        -------
        np.ndarray or None
            Float32 array of shape ``(512,)`` with unit L2 norm, or ``None``
            if no face was detected.

        Notes
        -----
        When multiple faces are detected the one with the highest detection
        score is used.  Embeddings are L2-normalised so cosine similarity
        equals the dot product.

        .. warning::
           Processing face images may constitute biometric data processing
           under GDPR Art. 9 / BIPA.  Ensure you have obtained the necessary
           legal basis before calling this method.
        """
        faces = self._app.get(image)
        if not faces:
            return None

        # Pick highest-confidence detection
        best = max(faces, key=lambda f: float(f.det_score))
        emb = best.normed_embedding.astype(np.float32)

        # Ensure unit norm (InsightFace should already normalise, but be safe)
        norm = float(np.linalg.norm(emb))
        if norm > 1e-6:
            emb /= norm

        return emb
