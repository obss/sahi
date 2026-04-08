import fire

from sahi.predict import predict_fiftyone


def main() -> None:
    fire.Fire(predict_fiftyone)


if __name__ == "__main__":
    main()
