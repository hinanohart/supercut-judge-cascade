# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - 2024-01-01

### Added

- 3-stage Vision LLM judge cascade (Stage C defect gate, Stage A viewer score,
  Stage B editor score) backed by `litellm` for provider-agnostic model access.
- `VisionLLMJudge`: encode frames as base-64 JPEG, send to any litellm-
  compatible model, parse structured JSON response.
- `JudgeResult`, `checkpoint_path`, `write_checkpoint`, `read_checkpoint`,
  `is_done`, `is_c_accepted`: atomic checkpoint I/O for resumable batch runs.
- `plan_batch`, `ingest_result`, `ingest_file`, `ingest_batch_dir`,
  `aggregate_summary`, `progress_status`: batch planning and result ingestion.
- `ArcFaceEmbedder`: InsightFace `buffalo_l` wrapper for 512-D face embeddings.
- `IdentityFilter` and `build_reference_embeddings`: cosine-similarity 1-vs-all
  identity gate with per-image and mean embedding support.
- `WindowExtractor`: sliding-window candidate extraction from video timestamps.
- `CVPrefilter`: OpenCV-based sharpness, face-ratio, and motion-energy heuristic
  scoring (Pearson r ≈ 0.029 vs Vision LLM; used as a complementary signal).
- `BucketNMS`: temporal non-maximum suppression to prevent duplicate frames in
  the final supercut.
- `PhashDedup`: perceptual hash deduplication for near-duplicate frame removal.
- `FaceCluster`: DBSCAN-based face clustering for unsupervised group detection.
- Judge prompt templates: `prompts/judge_a.md`, `prompts/judge_b.md`,
  `prompts/judge_c.md`.
- `supercut-cascade` CLI entry point (stub; full implementation in a future
  release).
- Apache 2.0 license, NOTICE, CREDITS, KNOWN_LIMITATIONS, BENCHMARKS, DESIGN,
  SECURITY, ETHICS, CITATION.cff.
- CI: GitHub Actions matrix (ubuntu 3.10/3.11/3.12 full, mac/win 3.12 build-
  only), CodeQL, release workflow with PyPI Trusted Publisher and identity-leak
  guard.
