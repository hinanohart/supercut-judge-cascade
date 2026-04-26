# Design: supercut-judge-cascade

## Overview

`supercut-judge-cascade` is a video supercut generation pipeline that uses a
3-stage Vision LLM judge cascade to select the best candidate shots from raw
footage.  The design is driven by two empirical observations:

1. **Cost reduction is essential.** Calling an expensive Vision LLM on every
   candidate frame is prohibitively slow and costly.  A cheap gate running
   first eliminates technically defective frames before they reach costly
   models.
2. **Orthogonality matters.** Internal measurements on real footage showed a
   Pearson correlation of r ≈ 0.029 between a heuristic scoring function
   (sharpness, face-ratio, motion energy) and the Vision LLM Judge A scores.
   This near-zero correlation means the two signals are nearly independent and
   must both be collected — neither can substitute for the other.

---

## 3-Stage Cascade: Cheap → Mid → Expensive

### Stage C — Defect Gate (Cheap)

- Model: any fast, low-cost Vision LLM (e.g. `claude-haiku-4-5`, `gpt-4o-mini`)
- Task: binary accept/reject on **technical defects only** (closed eyes, heavy
  blur, overexposure, text overlay, split-screen, wrong subject)
- Errs on the side of accept to avoid discarding borderline frames
- Only frames that pass Stage C are forwarded to Stages A and B
- Typical cost gate: rejects ~40-60 % of candidates before expensive calls

### Stage A — Viewer Score (Mid)

- Model: mid-tier Vision LLM
- Task: scene quality from a viewer's perspective, 1-10 integer
- Axes: gaze/eyes, emotional intensity, composition, lighting, motion/atmosphere
- Prompt: `prompts/judge_a.md`

### Stage B — Editor Score (Expensive)

- Model: highest-quality Vision LLM available
- Task: editorial value for final cut, 1-10 integer with sub-axis breakdown
  (Composition, Motion naturalness, Authenticity/Emotional resonance)
- Prompt: `prompts/judge_b.md`
- Final score: `ab_avg = (score_A + score_B) / 2`

---

## Vision LLM Provider Abstraction

All model calls route through [litellm](https://github.com/BerriAI/litellm),
which provides a unified interface across Anthropic, OpenAI, Google Gemini,
Azure, and any other provider supported by litellm.  The model string is passed
directly to `litellm.completion(model=...)`.

Switching providers requires only changing the model string; no code changes are
needed:

```python
judge = VisionLLMJudge(model="claude-haiku-4-5")          # Anthropic
judge = VisionLLMJudge(model="gpt-4o-mini")               # OpenAI
judge = VisionLLMJudge(model="gemini/gemini-1.5-flash")   # Google
```

---

## ArcFace Identity Gate

For single-subject supercuts (selecting only frames showing a specific person),
an optional 4th gate runs **before** Stage C:

1. `ArcFaceEmbedder` (backed by InsightFace `buffalo_l`) embeds each face
   detected in the candidate frame into a 512-D L2-normalised vector.
2. `IdentityFilter` computes cosine similarity against the mean reference
   embedding built from 1-3 reference images of the target person.
3. Frames where the primary face similarity falls below `threshold` (default
   0.40) are discarded without any Vision LLM call.

**Legal note**: `buffalo_l` weights are non-commercial only.  See
`KNOWN_LIMITATIONS.md` and `CREDITS.md`.

---

## Bucket NMS (Non-Maximum Suppression)

To avoid temporal redundancy in the output supercut, candidate windows are
grouped into fixed-duration "buckets" (default: one bucket per source video
minute).  Within each bucket, only the highest-scoring window is retained.
This is implemented in `supercut_cascade.select` and prevents clusters of
visually identical frames from dominating the final cut.

---

## Checkpoint I/O

Results are written atomically to `<judgements_dir>/<stable_id>_<phase>.json`
after each judge call.  A file containing `"error": "parse"` is retry-eligible.
This design allows long batch runs to be interrupted and resumed without
re-processing completed windows.

---

## Key Empirical Finding

Heuristic pre-filter score vs. Vision LLM Judge A score: **Pearson r ≈ 0.029**.

This near-zero correlation (measured on a real 67-video dataset with 18 000+
candidate windows) is the primary motivation for using the full Vision LLM
cascade rather than relying on CV heuristics alone.  Score `v4` combining
0.55 × Judge A + 0.30 × heuristic + 0.15 × sharpness yielded the best
subjective quality in user evaluation.
