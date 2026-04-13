"""Command-line interface for SAHI object detection predictions."""

import fire

from sahi.predict import predict


def main() -> None:
    """Run SAHI prediction via command-line interface."""
    fire.Fire(predict)


if __name__ == "__main__":
    main()
