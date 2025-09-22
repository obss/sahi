import logging
import os
import re
from abc import ABC, abstractmethod
from typing import ClassVar, Protocol, cast

PKG_INFO_LEVEL = logging.INFO + 5
logging.addLevelName(PKG_INFO_LEVEL, "PKG_INFO")


class SupportsPkgInfo(Protocol):
    def pkg_info(self, message: str, *args, **kws) -> None: ...  # pragma: no cover


class BaseSahiLogger(logging.Logger, ABC):
    @abstractmethod
    def pkg_info(self, message: str, *args, **kws) -> None:
        """Log a package info message at PKG_INFO level."""
        raise NotImplementedError


class SahiLogger(BaseSahiLogger):
    def pkg_info(self, message: str, *args, **kws) -> None:
        if self.isEnabledFor(PKG_INFO_LEVEL):
            self._log(PKG_INFO_LEVEL, message, args, **kws)


# ensure subsequent getLogger returns SahiLogger instances
logging.setLoggerClass(SahiLogger)


class SahiLoggerFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    cyan = "\x1b[36;20m"  # package name color
    green = "\x1b[32;20m"  # version color
    reset = "\x1b[0m"
    base_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS: ClassVar[dict[int, str]] = {
        logging.DEBUG: grey + base_format + reset,
        logging.INFO: grey + base_format + reset,
        logging.WARNING: yellow + base_format + reset,
        logging.ERROR: red + base_format + reset,
        logging.CRITICAL: bold_red + base_format + reset,
        PKG_INFO_LEVEL: base_format,
    }

    pkg_info_pattern = re.compile(r"^(?P<name>\S+)\s+version\s+(?P<version>\S+)\s+(?P<rest>.*)$")

    def format(self, record):  # type: ignore[override]
        if record.levelno == PKG_INFO_LEVEL:
            # Custom minimal line without timestamp/file info
            msg = record.getMessage()
            m = self.pkg_info_pattern.match(msg)
            if m:
                name = f"{self.cyan}{m.group('name')}{self.reset}"
                version = f"{self.green}{m.group('version')}{self.reset}"
                rest = m.group("rest")
                return f"{name} version {version} {rest}".rstrip()
            else:
                return msg  # fallback
        # default handling
        log_fmt = self.FORMATS.get(record.levelno, self.base_format)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


logger = cast(SahiLogger, logging.getLogger("sahi"))
if os.environ.get("SAHI_DEBUG"):
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
console_handler.setFormatter(SahiLoggerFormatter())
logger.addHandler(console_handler)

__all__ = ["PKG_INFO_LEVEL", "BaseSahiLogger", "SahiLogger", "SupportsPkgInfo", "logger"]
