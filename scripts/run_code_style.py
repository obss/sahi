import sys

from scripts.utils import shell, validate_and_exit

if __name__ == "__main__":
    arg = sys.argv[1]

    if arg == "check":
        sts_flake = shell("flake8 . --config setup.cfg --select=E9,F63,F7,F82")
        sts_isort = shell("isort . --check --settings pyproject.toml")
        sts_black = shell("black . --check --config pyproject.toml")
        validate_and_exit(flake8=sts_flake, isort=sts_isort, black=sts_black)
    elif arg == "format":
        sts_isort = shell("isort . --settings pyproject.toml")
        sts_black = shell("black . --config pyproject.toml")
        validate_and_exit(isort=sts_isort, black=sts_black)
