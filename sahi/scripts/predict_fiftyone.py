"""Command-line interface for SAHI predictions with FiftyOne integration."""

import fire

from sahi.predict import predict_fiftyone


def main() -> None:
    """Run SAHI prediction with FiftyOne visualization via command-line interface."""
    fire.Fire(predict_fiftyone)


if __name__ == "__main__":
    main()
