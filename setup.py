import io
import os
import re

import setuptools


def get_long_description():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    with io.open(os.path.join(base_dir, "README.md"), encoding="utf-8") as f:
        return f.read()


def get_requirements():
    with open("requirements.txt") as f:
        return f.read().splitlines()


def get_version():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    version_file = os.path.join(current_dir, "sahi", "__init__.py")
    with io.open(version_file, encoding="utf-8") as f:
        return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)


setuptools.setup(
    name="sahi",
    version=get_version(),
    author="OBSS",
    license="MIT",
    description="A vision library for performing sliced inference on large images/small objects",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/obss/sahi",
    packages=setuptools.find_packages(exclude=["conda", "demo", "tests"]),
    python_requires=">=3.7",
    install_requires=get_requirements(),
    extras_require={
        "tests": ["pytest", "mmdet", "norfair"],
        "dev": ["black==21.7b0", "flake8==3.9.2", "isort==5.9.2", "jupyterlab==3.0.14"],
    },
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
