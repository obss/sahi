import fire

from sahi.predict import predict_fiftyone


def main():
    fire.Fire(predict_fiftyone)


if __name__ == "__main__":
    main()
