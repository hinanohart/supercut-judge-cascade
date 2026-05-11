# supercut-judge-cascade

Vision-LLM 3-stage judge cascade for video supercuts — 95% prompt-cost reduction vs single-shot judging.

## Install

```bash
pip install supercut-judge-cascade
```

Install optional extras as needed:

```bash
# ArcFace identity gating (requires separate weight download — see Legal Notice)
pip install "supercut-judge-cascade[arcface]"

# PySceneDetect scene boundary detection
pip install "supercut-judge-cascade[detect]"

# Vision LLM judge (Claude / OpenAI / Gemini via litellm)
pip install "supercut-judge-cascade[vision]"

# All extras
pip install "supercut-judge-cascade[all]"
```

## Quickstart

```bash
# 1. Detect scene boundaries
supercut-cascade detect \
  --input big_buck_bunny.mp4 \
  --output scenes.json

# 2. Build a supercut (requires arcface + detect + vision extras)
supercut-cascade build \
  --refs reference_face_1.jpg reference_face_2.jpg \
  --pool big_buck_bunny.mp4 \
  --output supercut.mp4 \
  --target-label "Bunny" \
  --judge-model claude-haiku-4-5 \
  --threshold-cosine 0.40

# 3. QA check the output
supercut-cascade qa --input supercut.mp4
```

Python API:

```python
import sys
sys.path.insert(0, "src")  # if running from source

from supercut_cascade import (
    VisionLLMJudge, ArcFaceEmbedder, IdentityFilter,
    detect_scenes, build_windows, select_and_order, final_qa,
    extract_frame, cut_clip, concat_clips, probe_duration,
)

# Probe video duration
dur = probe_duration("video.mp4")

# Extract a single frame at 12.5 s
frame = extract_frame("video.mp4", 12.5)  # numpy BGR uint8

# Detect scenes
scenes = detect_scenes("video.mp4")

# Score a frame with the judge cascade
judge = VisionLLMJudge(model="claude-haiku-4-5", target_label="Alice")
result = judge.judge(frame, prompt_template, stable_id="sid_001", phase="C")
```

## Architecture

### 3-Stage Judge Cascade

```
Pool video
    |
    +-- Scene detect (PySceneDetect ContentDetector)
    |
    +-- Sliding windows (1 s, stride 0.5 s)
    |
    +-- [Optional] ArcFace identity gate (cosine similarity >= threshold)
    |
    +-- Stage C (Cheap / Gate)
    |       Fast, cheap Vision LLM.  Binary accept/reject on technical quality.
    |       ~80% of windows rejected here at minimal cost.
    |
    +-- Stage A (Mid / Viewer)
    |       Scene quality score 1-10 from a viewer perspective.
    |       Only windows that passed Stage C are scored.
    |
    +-- Stage B (Expensive / Editor)
    |       Scene quality score 1-10 from an editor perspective.
    |       Only windows that passed Stage C are scored.
    |
    +-- select_and_order()
    |       Temporal NMS, per-video cap, 3-act reorder, no-consecutive-video.
    |
    +-- concat_clips()  →  supercut.mp4
```

**Cost reduction:** A three-stage cascade that gates ~80% of windows at Stage C
(cheapest model) means Stages A and B are called for only the top ~20% of
candidates, reducing total token cost by approximately 95% vs scoring all
windows with a single expensive model.

**ArcFace identity gate:** Before judge scoring, an optional cosine-similarity
gate compares each candidate frame's 512-D ArcFace embedding against a
pre-built mean reference embedding.  Windows with similarity below
`--threshold-cosine` are dropped without any LLM call.

**Bucket NMS:** Within each source video, temporal IoU NMS and a minimum
center-to-center gap prevent near-duplicate clips from appearing in the output.

### Key modules

| Module | Role |
|---|---|
| `supercut_cascade.detect` | PySceneDetect wrapper |
| `supercut_cascade.windows` | Sliding-window planner |
| `supercut_cascade.arcface` | ArcFace 512-D embedding (InsightFace) |
| `supercut_cascade.identity_filter` | Cosine-similarity 1-vs-all filter |
| `supercut_cascade.judge` | 3-stage Vision LLM judge (litellm) |
| `supercut_cascade.select` | NMS + ordering pipeline |
| `supercut_cascade.qa` | Decode / blackframe / PTS QA |
| `supercut_cascade.io` | FFmpeg subprocess wrapper |
| `supercut_cascade.cli` | CLI entry point |

## Legal Notice

### ArcFace / buffalo_l model weights

The `buffalo_l` model weights distributed by InsightFace are trained on the
MS1MV2 dataset and are licensed for **NON-COMMERCIAL use only**.

Users must download the weights separately before use:

```bash
insightface-cli model.download buffalo_l
```

The weights must not be bundled with this package or committed to any
repository.  Commercial users must obtain a separately licensed model.

### Biometric data — GDPR Article 9 and US Illinois BIPA

Face embeddings derived from images of natural persons constitute **biometric
data** and may be classified as **special category personal data** under EU
General Data Protection Regulation Article 9, and as **biometric identifiers**
under the Illinois Biometric Information Privacy Act (BIPA) and equivalent
laws in other jurisdictions.

**It is the user's sole responsibility to:**

- Obtain informed, freely given, specific, and unambiguous consent from all
  data subjects before processing their face images or derived embeddings.
- Honour all data-subject rights (access, rectification, erasure, portability)
  under applicable privacy law.
- Implement appropriate technical and organisational safeguards (access
  controls, encryption at rest and in transit, retention limits).
- Never store reference embeddings in shared or publicly accessible locations.
- Comply with all local, national, and supranational privacy regulations
  applicable to your use case.

The authors of this software make no representation that any particular use of
this library complies with applicable law.

### Scope of use

This software is provided for educational and research purposes.  The authors
and contributors are not liable for any direct, indirect, incidental, special,
or consequential damages arising from misuse of this software, including but
not limited to privacy violations, biometric data breaches, or non-compliance
with applicable law.

## License

MIT License — see [LICENSE](LICENSE).
