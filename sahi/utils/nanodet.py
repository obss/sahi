import logging
from pathlib import Path

import requests


class NanodetConstants:
    NANODET_PLUS_CONFIG = Path("tests/data/models/nanodet/nanodet-plus-m_416.yml").resolve().as_posix()

    NANODET_PLUS_MODEL = Path("tests/data/models/nanodet/model.ckpt").resolve().as_posix()

    NANODET_PLUS_URL = (
        "https://github.com/RangiLyu/nanodet/releases/download/v1.0.0-alpha-1/nanodet-plus-m_416_checkpoint.ckpt"
    )

    def __init__(self) -> None:
        if not Path(self.NANODET_PLUS_MODEL).exists():
            logging.info("Downloading Nanodet model.")
            response = requests.get(self.NANODET_PLUS_URL, allow_redirects=True, timeout=10)
            logging.info("Downloaded Nanodet model.")
            with open(self.NANODET_PLUS_MODEL, "wb") as model_file:
                model_file.write(response.content)
