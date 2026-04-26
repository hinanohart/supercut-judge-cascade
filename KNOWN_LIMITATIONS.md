# Known Limitations

## 1. Non-Commercial ArcFace Weights

The default InsightFace model (`buffalo_l`) is trained on the MS1MV2 dataset
and distributed under a **non-commercial license**.  You must download the
weights separately:

```bash
insightface-cli model.download buffalo_l
```

The weights **must not** be bundled with this package, committed to any
repository, or used in any commercial application without a separate license
from the InsightFace authors.

If you require commercial use, replace `ArcFaceEmbedder` with a commercially
licensed embedding model and implement the `IdentityFilter` interface.

---

## 2. GDPR / BIPA Compliance

Face embeddings are **biometric data** and constitute sensitive personal data
under the EU General Data Protection Regulation (Article 9) and the Illinois
Biometric Information Privacy Act (BIPA), among other laws worldwide.

**You are solely responsible for:**

- Obtaining explicit, informed consent from data subjects before processing
  their face images.
- Complying with data minimisation, purpose limitation, and retention limits.
- Honouring data-subject rights (erasure, access, portability).
- Not exporting embeddings to jurisdictions that prohibit biometric data
  transfer.
- Ensuring adequate technical and organisational security measures for stored
  embeddings.

Reference embeddings (`.npy` files) **must not** be committed to any public
repository.

---

## 3. InsightFace Gender/Age Accuracy on Non-Western Subjects

InsightFace's gender and age estimation models show a known ~15–20 % error rate
on East-Asian female subjects in their 20s.  If you use these attributes as
filters, use them as **soft penalties** (score reduction) rather than hard
rejects to avoid systematic exclusion of valid frames.

---

## 4. ffmpeg Dependency

Scene detection (`scenedetect`) and video assembly require `ffmpeg` to be
available on `PATH`.  ffmpeg is distributed separately under the LGPL / GPL;
see https://ffmpeg.org/legal.html.

---

## 5. No Streaming / Real-Time Support

The pipeline is designed for **offline batch processing**.  It does not support
real-time video streams or latency-sensitive applications.

---

## 6. Vision LLM Prompt Injection

Image text (lyrics, subtitles, on-screen captions) is treated as data, not
instructions, by the judge prompts.  However, adversarial image content could
potentially manipulate model outputs.  Do not use this pipeline in adversarial
or untrusted-input contexts without additional safeguards.

---

## 7. Python 3.10+ Only

The codebase uses `match` statements, `X | Y` union types, and other features
that require Python 3.10 or later.
