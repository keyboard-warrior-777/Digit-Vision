# Security Policy

## Supported Versions

DigitVision v1.0.x is the only currently supported version.

| Version | Supported |
|---------|-----------|
| 1.0.x   | ✅ Yes    |

---

## Scope

DigitVision is a portfolio/educational application that runs locally or in a
Docker container on a private machine. It does not:

- Accept arbitrary code execution from external users
- Store user credentials or personal data
- Connect to external APIs or databases
- Expose a public-facing network endpoint by default

The primary security concerns for this project are:

1. **Dependency vulnerabilities** in `tensorflow`, `streamlit`, `pillow`, or
   `opencv-python-headless`
2. **Container hardening** (the Dockerfile runs as a non-root user)
3. **Model file integrity** (trained `.keras` files loaded from disk)

---

## Reporting a Vulnerability

If you discover a security vulnerability, please do **not** open a public
GitHub issue. Instead, report it privately:

**Email:** [your-email@example.com]

Please include:
- A description of the vulnerability
- Steps to reproduce
- The potential impact
- Any suggested mitigations

You will receive a response within **72 hours**. If the vulnerability is
confirmed, a patch will be prepared and a security advisory published before
public disclosure (coordinated disclosure).

---

## Dependency Auditing

You can audit the current dependency tree for known vulnerabilities using:

```bash
pip install pip-audit
pip-audit -r requirements.txt
```

Or with the Safety CLI:

```bash
pip install safety
safety check -r requirements.txt
```

---

## Docker Security Notes

The production Dockerfile:
- Uses `python:3.12-slim` (minimal attack surface)
- Runs as a non-root user (`digitvision`)
- Does not expose any port other than 8501
- Uses `--no-install-recommends` for all apt packages

When deploying publicly, place the app behind a reverse proxy (nginx, Caddy)
with TLS and appropriate firewall rules.
