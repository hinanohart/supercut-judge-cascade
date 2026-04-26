# Credits and Third-Party Licenses

`supercut-judge-cascade` itself is distributed under the **Apache License 2.0**
(see `LICENSE`).  It depends on the following third-party components:

---

## RAFT (Optical Flow)

- Authors: Zachary Teed and Jia Deng
- Paper: "RAFT: Recurrent All-Pairs Field Transforms for Optical Flow" (ECCV 2020)
- License: BSD 3-Clause
- Source: https://github.com/princeton-vl/RAFT

---

## scipy

- Authors: scipy contributors
- License: BSD 3-Clause
- Source: https://github.com/scipy/scipy

---

## PySceneDetect (scenedetect)

- Author: Brandon Castellano
- License: BSD 3-Clause
- Source: https://github.com/Breakthrough/PySceneDetect

---

## litellm

- Authors: BerriAI
- License: MIT
- Source: https://github.com/BerriAI/litellm

---

## InsightFace / buffalo_l weights

- Authors: InsightFace contributors
- License: MIT (library); **NON-COMMERCIAL ONLY** (pre-trained `buffalo_l` weights)
- Source: https://github.com/deepinsight/insightface
- Weight download: `insightface-cli model.download buffalo_l`

**IMPORTANT**: The `buffalo_l` pre-trained weights are trained on the MS1MV2
dataset and are distributed for **non-commercial research and education use
only**.  Commercial use of the weights requires a separate license agreement
with the InsightFace authors.  The weights are **not bundled** with this
package.

---

## OpenCV (opencv-python)

- Authors: OpenCV contributors
- License: Apache 2.0
- Source: https://github.com/opencv/opencv

---

## Other Dependencies

All other dependencies (`numpy`, `Pillow`, `click`, `tqdm`, `scikit-learn`,
`imagehash`, `hypothesis`, `pytest`, `ruff`) are distributed under permissive
open-source licenses (MIT, BSD, Apache 2.0).  See each package's own
`LICENSE` file for details.
