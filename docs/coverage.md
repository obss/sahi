# Code Coverage Guide

This document explains how code coverage is configured and used in the SAHI project.

## Overview

SAHI uses [pytest-cov](https://pytest-cov.readthedocs.io/) for measuring code coverage and [Codecov](https://codecov.io/) for coverage reporting and tracking.

## Configuration

### pytest-cov Configuration

Coverage settings are defined in `pyproject.toml`:

```toml
[tool.coverage.run]
source = ["sahi"]
omit = [
    "*/tests/*",
    "*/test_*.py",
    "*/__pycache__/*",
    "*/site-packages/*",
    "sahi/scripts/*",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
    "if typing.TYPE_CHECKING:",
    "@abstractmethod",
    "@abc.abstractmethod",
]
precision = 2
show_missing = true
```

### Codecov Configuration

Coverage reporting is configured in `.codecov.yml`:

- **Target Coverage**: Auto-adjusts based on previous runs
- **Threshold**: 1% change tolerance
- **Ignored Paths**: Tests, scripts, demos, and docs

## Running Coverage Locally

### Generate Coverage Report

```bash
# Run tests with coverage
uv run pytest --cov=sahi --cov-report=html --cov-report=term-missing

# View HTML report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### Coverage Report Options

```bash
# Terminal output only
uv run pytest --cov=sahi --cov-report=term-missing

# XML output (for Codecov)
uv run pytest --cov=sahi --cov-report=xml

# HTML report
uv run pytest --cov=sahi --cov-report=html

# Multiple report formats
uv run pytest --cov=sahi --cov-report=xml --cov-report=html --cov-report=term-missing
```

### Run Coverage for Specific Modules

```bash
# Coverage for specific module
uv run pytest tests/test_predict.py --cov=sahi.predict --cov-report=term-missing

# Coverage for multiple modules
uv run pytest --cov=sahi.predict --cov=sahi.slicing --cov-report=term-missing
```

## CI/CD Integration

### GitHub Actions Workflow

Coverage is automatically collected during CI runs:

```yaml
- name: Test with python ${{ matrix.python-version }}
  run: |
    uv run pytest --cov=sahi --cov-report=xml --cov-report=term-missing

- name: Upload coverage to Codecov
  if: matrix.os == 'ubuntu-latest' && matrix.python-version == '3.11'
  uses: codecov/codecov-action@v4
  with:
    file: ./coverage.xml
    flags: unittests
```

**Note**: Coverage is only uploaded from Ubuntu with Python 3.11 to avoid duplicate reports.

### Codecov Setup

1. **Enable Codecov**: The repository is connected to [Codecov](https://codecov.io/gh/obss/sahi)
2. **Token**: `CODECOV_TOKEN` secret is configured in GitHub repository settings
3. **Badge**: Coverage badge is displayed in README.md

## Coverage Badge

The coverage badge in README.md shows the current coverage percentage:

```markdown
[![codecov](https://codecov.io/gh/obss/sahi/branch/main/graph/badge.svg)](https://codecov.io/gh/obss/sahi)
```

## Excluding Code from Coverage

### Using Pragma Comments

```python
# Exclude single line
if debug:  # pragma: no cover
    print("Debug mode")

# Exclude block
def development_only_function():  # pragma: no cover
    """This function is not covered by tests."""
    pass
```

### Already Excluded Patterns

The following patterns are automatically excluded:

- `def __repr__`: String representation methods
- `raise AssertionError`: Assertion errors
- `raise NotImplementedError`: Abstract method stubs
- `if __name__ == .__main__.:`: Main blocks
- `if TYPE_CHECKING:`: Type checking blocks
- `@abstractmethod`: Abstract methods

## Coverage Goals

- **Minimum Target**: 70% overall coverage
- **Ideal Target**: 80%+ overall coverage
- **New Code**: Should maintain or improve coverage
- **Critical Modules**: Aim for 90%+ coverage on core modules

## Best Practices

1. **Write Tests First**: Use TDD when possible
2. **Cover Edge Cases**: Test boundary conditions and error paths
3. **Test Public APIs**: Ensure all public functions/classes are tested
4. **Avoid Testing Implementation Details**: Focus on behavior
5. **Review Coverage Reports**: Regularly check which code needs tests

## Interpreting Coverage Reports

### Terminal Output

```
Name                           Stmts   Miss  Cover   Missing
------------------------------------------------------------
sahi/__init__.py                  10      0   100%
sahi/predict.py                  250     25    90%   45-48, 123-125
sahi/slicing.py                  180     10    94%   234-238
------------------------------------------------------------
TOTAL                           2450    150    94%
```

- **Stmts**: Total statements
- **Miss**: Statements not covered
- **Cover**: Coverage percentage
- **Missing**: Line numbers not covered

### HTML Report

The HTML report provides:
- File-by-file coverage breakdown
- Line-by-line coverage visualization
- Branch coverage information
- Sortable columns

## Troubleshooting

### Coverage Not Being Collected

```bash
# Ensure pytest-cov is installed
uv pip list | grep pytest-cov

# Check if tests are being discovered
uv run pytest --collect-only

# Run with verbose coverage
uv run pytest --cov=sahi --cov-report=term-missing -v
```

### Low Coverage Numbers

1. Check if test files are being excluded properly
2. Verify `source` and `omit` settings in `pyproject.toml`
3. Ensure tests are actually running (check test count)
4. Review coverage report for specific missing lines

### Codecov Upload Failures

1. Verify `CODECOV_TOKEN` is set in repository secrets
2. Check Codecov action version is up to date
3. Ensure `coverage.xml` file exists after tests
4. Review GitHub Actions logs for errors

## Additional Resources

- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [Codecov Documentation](https://docs.codecov.com/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [SAHI Testing Guide](./CONTRIBUTING.md#running-tests)

## Questions?

If you have questions about coverage or need help improving test coverage, please:

1. Check this guide first
2. Review existing tests in the `tests/` directory
3. Ask in [GitHub Discussions](https://github.com/obss/sahi/discussions)
