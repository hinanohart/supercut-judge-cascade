# Benchmarks

All numbers are approximate and depend on hardware, model provider latency, and
video content.

## Vision LLM Cost Reduction

| Scenario | LLM calls | Relative cost |
|---|---|---|
| No cascade (all frames to Stage B) | 100 % | 1× |
| Stage C gate only (rejects ~50 %) | 50 % | 0.50× |
| Stage C + Stage A pre-filter | ~20 % reach Stage B | **~0.05–0.10×** |

In practice, the 3-stage cascade reduces Vision LLM expenditure by **≈ 90–95 %**
compared to running every frame through the most expensive model.

## CPU Throughput (no GPU)

| Operation | Throughput |
|---|---|
| Scene detection (PySceneDetect content detector) | ~200–400 frames/s |
| OpenCV prefilter (sharpness + face-ratio heuristic) | ~100 frames/s |
| Stage C judge (claude-haiku-4-5, 1 frame) | ~0.8–1.5 s/call (network) |
| Stage A/B judge (claude-sonnet-4-5, 4 frames) | ~2–5 s/call (network) |
| ArcFace embedding (buffalo_l, CPU) | ~0.5–1 s/frame |

## Cloud Throughput (Modal T4 GPU)

| Operation | Throughput |
|---|---|
| ArcFace embedding (buffalo_l, T4 GPU) | ~15–30 frames/s |
| End-to-end pipeline per 5-minute MV | ~8–12 min wall-clock |

## Build / CI

| Job | Typical duration |
|---|---|
| Lint (ruff + bandit) | < 30 s |
| Test suite (ubuntu, Python 3.12) | < 60 s |
| Synthetic smoke example | < 60 s (no network, no GPU) |
