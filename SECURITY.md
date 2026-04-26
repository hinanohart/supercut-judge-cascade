# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| 0.1.x | Yes |

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Please report security vulnerabilities by emailing the maintainers directly.
Use the subject line `[SECURITY] supercut-judge-cascade <brief description>`.

You can also use GitHub's private vulnerability reporting feature:
**Security → Report a vulnerability** on the repository page.

We aim to acknowledge reports within **72 hours** and provide a remediation
timeline within **7 days**.

## Scope

Reports are in scope for:

- Remote code execution via malicious video files or model responses
- Credential / API key leakage in logs or checkpoint files
- Prompt injection attacks that cause the pipeline to process or exfiltrate
  data outside its intended scope
- Dependencies with known CVEs that affect this package's threat model

Out of scope:

- Denial-of-service via extremely large video files (resource exhaustion is
  the caller's responsibility)
- Theoretical vulnerabilities with no practical exploit path
- Issues in third-party services (litellm, OpenAI, Anthropic, etc.) themselves

## Security Considerations for Operators

1. **API keys**: Store Vision LLM API keys in environment variables or a
   secrets manager.  Never hardcode them or commit them to version control.
2. **Biometric data**: Face embeddings are sensitive personal data.  Store them
   in access-restricted directories with appropriate encryption at rest.
   See `KNOWN_LIMITATIONS.md` for GDPR/BIPA notes.
3. **Checkpoint files**: Checkpoint JSON files contain frame paths and model
   responses.  Restrict access to the `judgements_dir` directory.
4. **Prompt injection**: Image text (subtitles, captions) could contain
   adversarial content.  The judge prompts include a prompt-injection defence
   instruction, but no software defence is infallible.
5. **Model responses**: Parse model responses defensively; malformed JSON is
   handled gracefully but unexpected response content should be logged and
   audited.
