# SAHI Test Suite

This directory contains the test suite for SAHI. The tests are organized by model type and functionality.

## Test Structure

- **test_mmdetectionmodel.py** - MMDetection model tests (requires Python 3.11)
- **test_huggingfacemodel.py** - HuggingFace model tests (requires Python 3.9+)
- **test_ultralyticsmodel.py** - Ultralytics YOLO model tests
- **test_predict.py** - General prediction functionality tests
- Other test files for specific functionality

## Running Tests

### Running all tests (excluding MMDet)
```bash
# Install dependencies
uv sync --extra dev --extra ci

# Run all tests except MMDet
pytest -k "not mmdet"
```

### Running MMDet tests separately (Python 3.11 only)
```bash
# Ensure you're using Python 3.11
python --version  # Should show 3.11.x

# Install MMDet dependencies
uv sync --extra dev --extra mmdet

# Run only MMDet tests
pytest tests/test_mmdetectionmodel.py
# Or run specific MMDet tests from test_predict.py
pytest -k "mmdet"
```

### Running specific test categories
```bash
# Run only Ultralytics tests
pytest tests/test_ultralyticsmodel.py

# Run only HuggingFace tests (Python 3.9+)
pytest tests/test_huggingfacemodel.py

# Run tests in parallel
pytest -n auto
```

## CI/CD Setup

The test suite is split into two GitHub Actions workflows:

1. **Main CI** (`.github/workflows/ci.yml`):
   - Runs on Python 3.8, 3.9, 3.10, 3.11, 3.12
   - Excludes MMDet tests
   - Tests core functionality and all other model integrations

2. **MMDet CI** (`.github/workflows/mmdet.yml`):
   - Runs only on Python 3.11
   - Tests MMDetection integration separately
   - Can fail without affecting the main CI pipeline

This separation ensures that:
- MMDet dependency conflicts don't affect other tests
- MMDet failures don't block PRs
- Simplified dependency management
- Faster CI runs for the main test suite

## Test Dependencies

The test dependencies are organized in `pyproject.toml`:

- `[project.optional-dependencies.dev]` - Basic development dependencies
- `[project.optional-dependencies.ci]` - Main CI dependencies (no MMDet)
- `[project.optional-dependencies.mmdet]` - MMDetection dependencies (Python 3.11 only)

## Writing New Tests

All tests use pytest format. When adding new tests:

1. Use pytest assertions instead of unittest assertions
2. Use pytest decorators for skipping tests conditionally
3. Group related tests in the same file
4. Add appropriate skip conditions for Python version or missing dependencies

Example:
```python
import pytest
import sys

# Skip if Python version is too old
pytestmark = pytest.mark.skipif(
    sys.version_info[:2] < (3, 9), 
    reason="Feature requires Python 3.9+"
)

def test_my_feature():
    # Skip if optional dependency is missing
    pytest.importorskip("optional_package")
    
    # Your test code here
    assert result == expected
```