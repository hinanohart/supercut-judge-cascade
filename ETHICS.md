# Ethics Guidelines

`supercut-judge-cascade` uses face recognition technology (ArcFace / InsightFace)
and Vision LLM analysis to identify and score frames containing specific individuals.
This raises serious ethical obligations that all users and contributors must observe.

---

## 1. Consent is Mandatory

Processing face images of real people requires their **informed, explicit, and
freely given consent** in most legal jurisdictions:

- **GDPR Article 9** (EU/EEA): biometric data is a special category requiring
  explicit consent or another Article 9(2) lawful basis.
- **BIPA** (Illinois, USA): written consent required before collecting biometric
  identifiers.
- **PIPL** (China), **APPI** (Japan), **PDPA** (Thailand), and many other
  national laws contain equivalent requirements.

**Do not use this tool to process images of any person without first obtaining
the required legal consent.**

---

## 2. Prohibited Uses

The following uses are explicitly prohibited:

- **Surveillance or tracking**: building databases to monitor individuals'
  movements, activities, or associations without their knowledge and consent.
- **Harassment or stalking**: identifying or locating individuals for the
  purpose of harassment, intimidation, or unwanted contact.
- **Deepfake / synthetic media generation**: using extracted face embeddings or
  selected frames as inputs to generative models to create deceptive content
  depicting real people.
- **Non-consensual intimate imagery**: selecting or processing frames of a
  sexual or intimate nature without the depicted person's explicit consent.
- **Law enforcement / immigration**: identifying suspects, tracking protesters,
  or assisting border control without appropriate legal authority and oversight.
- **Profiling for employment, credit, or insurance**: using face analysis
  outputs as a basis for consequential decisions about individuals.
- **Processing children's faces**: minors cannot meaningfully consent to
  biometric data collection; do not process images of persons under 18.

---

## 3. Intended Use Cases

This tool is designed for:

- **Educational and research use**: studying Vision LLM cascade design,
  face embedding quality, and video analysis pipelines.
- **Personal media organisation**: creating supercuts from footage where the
  depicted people have consented to the processing (e.g. family videos, fan-
  made compilations of public performers who have consented to fan use).
- **Content creator workflows**: assisting creators who have obtained the
  necessary rights and consents for the footage they process.

---

## 4. Copyright and DMCA

Users are responsible for ensuring they have the legal right to process the
video content they provide to this pipeline:

- Processing commercially distributed video (films, music videos, TV) without
  a licence may infringe copyright.
- In Japan, Article 30-4 of the Copyright Act permits use for information
  analysis (情報解析) without rights-holder consent, subject to conditions;
  consult a qualified lawyer before relying on this exception.
- The US DMCA (17 U.S.C. § 512) safe harbour does not apply to private
  pipeline operators; ensure your use complies with applicable fair-use
  or equivalent doctrines.

---

## 5. Data Minimisation

- Collect only the embeddings and metadata you need for your specific use case.
- Delete intermediate files (frame thumbnails, contact sheets, embeddings) when
  they are no longer required.
- Do not share reference embeddings or checkpoint files without the data
  subject's consent.

---

## 6. Transparency

If you publish research or applications built on this tool, disclose:

- That face recognition technology was used.
- The consent basis on which face data was processed.
- Any limitations of the system that may affect the reliability of results.

---

## 7. Reporting Misuse

If you become aware of this tool being used in a manner that violates these
guidelines, please report it via the repository's issue tracker or the
security contact described in `SECURITY.md`.
