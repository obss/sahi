import fire

from sahi.predict_batch import predict


def main():
    fire.Fire(predict)


if __name__ == "__main__":
    main()
