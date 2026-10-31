# SAHI Test Suite

Tests use pytest. Most files cover one model backend (for example
`test_ultralytics_model.py`, `test_huggingface_model.py`,
`test_mmdetection_model.py`) or one module (for example `test_slicing.py`,
`test_coco_utils.py`).

## Running tests

```bash
# install sahi with the dev tools and every test backend
uv sync --group dev --extra ci

# run the whole suite in parallel
uv run pytest -n auto

# run a single file
uv run pytest tests/test_slicing.py
```

Model weights are downloaded into `tests/data/models` on first use.

## Optional backends

Tests for a backend that is not installed skip themselves, so a partial install
still runs. Some backends are also limited by Python version:

- MMDetection tests only run on Python 3.11 and are not installed by the `ci`
  extra, so CI skips them.
- Roboflow tests run on Python 3.12 and 3.13 on Linux.

## CI

`.github/workflows/ci.yml` runs the suite on Python 3.10 to 3.14 on Ubuntu, and
on Python 3.12 on macOS and Windows, followed by a few `sahi` CLI commands.

## Writing new tests

- Use plain pytest assertions.
- Skip with `pytest.importorskip("package")` when a test needs an optional
  dependency.
- Keep related tests in the same file.
