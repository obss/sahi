import sys
import warnings

from scripts.utils import shell, validate_and_exit

if __name__ == "__main__":
    arg = sys.argv[1]
    warnings.warn(
        "Please use 'ruff check' and 'ruff format' instead. Precede with 'uv run' to run in virtual environment. Remember to activate the pre-commit hook to do that automatically on every commit.",
        DeprecationWarning,
    )

    if arg == "check":
        sts_flake = shell("flake8 . --config setup.cfg --select=E9,F63,F7,F82")
        sts_isort = shell("isort . --check --settings pyproject.toml")
        sts_black = shell("black . --check --config pyproject.toml")
        validate_and_exit(flake8=sts_flake, isort=sts_isort, black=sts_black)
    elif arg == "format":
        sts_isort = shell("isort . --settings pyproject.toml")
        sts_black = shell("black . --config pyproject.toml")
        validate_and_exit(isort=sts_isort, black=sts_black)
