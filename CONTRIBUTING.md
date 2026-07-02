# Contributing to DigitVision

Thank you for your interest in contributing! DigitVision is a portfolio project
demonstrating production-quality ML engineering. Contributions that improve
correctness, documentation, test coverage, or deployment reliability are welcome.

---

## Before You Start

- **The ML architecture is stable.** The three model architectures (Dense NN,
  LeNet-5, Custom CNN), the training pipeline, and the Grad-CAM implementation
  are intentionally kept as-is. Do not open pull requests that redesign these
  components.

- **Open an issue first** for anything beyond a typo fix or documentation
  improvement. This avoids wasted effort on changes that won't be merged.

---

## Development Setup

```bash
git clone https://github.com/keyboard-warrior-777/digitvision.git
cd digitvision

# Windows
py -3.12 -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3.12 -m venv .venv
source .venv/bin/activate

# Install with dev extras
pip install -e ".[dev]"
```

---

## Making Changes

### 1. Create a branch

```bash
git checkout -b fix/short-description
```

Branch names follow `fix/`, `docs/`, or `test/` prefixes. Feature branches
(`feat/`) should only be opened after issue discussion.

### 2. Make your changes

Keep changes focused. One logical change per pull request.

### 3. Run the full test suite

```bash
make test
# Or:
pytest tests/ -v --cov=src --cov-report=term-missing
```

All 145 tests must pass before submitting a PR.

### 4. Run linting and formatting

```bash
make lint     # ruff check
make format   # black + ruff --fix
```

CI enforces both. PRs that fail lint checks will not be reviewed.

### 5. Update CHANGELOG.md

Add a brief entry under an `Unreleased` section describing what you changed
and why.

---

## Pull Request Guidelines

- **Title**: Use a clear, concise description: `Fix: canvas inversion fails for thick strokes`
- **Description**: Explain what was wrong, what you changed, and how to verify it
- **Tests**: Add or update tests for any code change. Do not reduce test coverage.
- **Documentation**: Update the relevant `docs/` file if the change affects user-facing behaviour

---

## What We Welcome

| Type | Examples |
|------|----------|
| Bug fixes | Incorrect preprocessing, broken page navigation |
| Documentation | Typos, outdated commands, missing steps |
| Test coverage | New edge cases, missing assertions |
| Deployment | Docker improvements, CI enhancements |

## What We Will Decline

| Type | Reason |
|------|--------|
| Architecture redesigns | The ML pipeline is frozen for v1.0 |
| New model architectures | Out of scope for this release |
| Dependency upgrades | Requires careful compatibility testing |
| Speculative improvements | Changes must address verified, measurable issues |

---

## Code Style

- **Formatter**: Black (`line-length = 88`)
- **Linter**: Ruff with `E, F, W, I, N, UP, B, SIM` rules
- **Docstrings**: Google-style, mandatory for all public functions
- **Type hints**: Required for all public function parameters and return types
- **Log calls**: Use `%s` formatting, not f-strings: `logger.info("Value: %s", val)`

---

## Reporting Issues

When reporting a bug, please include:

1. Exact error message (copy from terminal or logs)
2. Python version: `python --version`
3. TensorFlow version: `python -c "import tensorflow; print(tensorflow.__version__)"`
4. Operating system
5. Steps to reproduce

---

## Code of Conduct

All contributors are expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

---

*Thank you for helping make DigitVision better.*
