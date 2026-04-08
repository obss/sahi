import fire

from sahi.predict import predict


def main() -> None:
    fire.Fire(predict)


if __name__ == "__main__":
    main()
