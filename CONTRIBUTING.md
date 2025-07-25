# Contributing to SAHI

Thank you for your interest in contributing to SAHI! This guide will help you get started.

## Setting Up Development Environment

### 1. Fork and Clone

```bash
git clone https://github.com/YOUR_USERNAME/sahi.git
cd sahi
```

### 2. Create Environment

We recommend Python 3.10 for development:

```bash
pip install uv
uv venv --python 3.10
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install core + dev dependencies
uv sync --extra dev

# For testing specific models, install their dependencies.
```

## Code Formatting

We use `ruff` for code formatting and linting. To format your code:

```bash
# Check formatting
uv run ruff check .
uv run ruff format --check .

# Fix formatting
uv run ruff check --fix .
uv run ruff format .
```

Or use the convenience script:

```bash
# Check formatting
python scripts/format_code.py check

# Fix formatting
python scripts/format_code.py fix
```

## Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_predict.py

# Run with coverage
uv run pytest --cov=sahi
```

## Submitting Pull Requests

1. Create a new branch: `git checkout -b feature-name`
2. Make your changes
3. Format your code: `python scripts/format_code.py fix`
4. Run tests: `uv run pytest`
5. Commit with clear message: `git commit -m "Add feature X"`
6. Push and create PR: `git push origin feature-name`

## CI Build Failures

If the CI build fails due to formatting:

1. Check the CI output for the specific Python version that failed
2. Create environment with that Python version:

   ```bash
   uv venv --python 3.X  # Replace X with the version from CI
   source .venv/bin/activate
   ```

3. Install dev dependencies:

   ```bash
   uv sync --extra dev
   ```

4. Fix formatting:

   ```bash
   python scripts/format_code.py fix
   ```

5. Commit and push the changes

## Adding New Model Support

To add support for a new detection framework:

1. Create a new file under `sahi/models/your_framework.py`
2. Implement a class that inherits from `DetectionModel`
3. Add your framework to `MODEL_TYPE_TO_MODEL_CLASS_NAME` in `sahi/auto_model.py`
4. Add tests under `tests/test_yourframework.py`
5. Add a demo notebook under `docs/notebooks/inference_for_your_framework.ipynb`
6. Update [`README.md`](README.md) and related docs under `docs/` to include your new model

See existing implementations like `sahi/models/ultralytics.py` for reference.

## Questions?

Feel free to [start a discussion](https://github.com/obss/sahi/discussions) if you have questions!
