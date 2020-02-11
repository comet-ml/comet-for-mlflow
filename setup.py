#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("HISTORY.md") as history_file:
    history = history_file.read()

requirements = ["mlflow", "comet_ml>=3.0.3", "tabulate", "tqdm", "typing"]


setup(
    author="Boris Feld",
    author_email="boris@comet.ml",
    python_requires=">=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    description="Extend MLFlow with Comet.ml",
    entry_points={"console_scripts": ["comet_for_mlflow=comet_for_mlflow.cli:main"]},
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + "\n\n" + history,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="comet_for_mlflow",
    name="comet_for_mlflow",
    packages=find_packages(include=["comet_for_mlflow", "comet_for_mlflow.*"]),
    test_suite="tests",
    url="https://github.com/comet-ml/comet_for_mlflow",
    version="0.1.0",
    zip_safe=False,
)
