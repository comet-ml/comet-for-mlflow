# -*- coding: utf-8 -*-

"""Tests for `comet_for_mlflow` package."""

import os
import os.path
from random import randint, random

import mlflow
import responses
from comet_ml.utils import url_join
from mlflow import log_artifacts, log_metric, log_param

from comet_for_mlflow import comet_for_mlflow

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

SERVER_ADDRESS = "https://www.comet.com/clientlib/"


def mlflow_example():
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    with open("outputs/test.txt", "w") as f:
        f.write("hello world!")

    with mlflow.start_run():
        log_param("param1", randint(0, 100))
        log_metric("foo", random())
        log_metric("foo", random() + 1)
        log_metric("foo", random() + 2)
        log_artifacts("outputs")


@responses.activate
def test_conversion(tmp_path, monkeypatch):
    path = tmp_path.resolve().as_posix()
    os.chdir(path)

    # Run the MLFlow example
    mlflow_example()

    # Check that MLFlow have created its on-disk content
    assert os.path.isdir(os.path.join(path, "mlruns"))

    # Monkey-patch HTTP interactions
    backend_version_body = {
        "msg": "1.2.131",
        "name": "Python-Backend",
        "ip": "",
        "hostname": "",
        "version": "1.2.131",
    }

    url = url_join(SERVER_ADDRESS, "isAlive/ver")

    responses.add(
        responses.GET,
        url,
        json=backend_version_body,
        status=200,
    )

    # Check that comet_for_mlflow have created an offline experiment
    api_key = "XXX"
    monkeypatch.setenv("COMET_WORKSPACE", "WORKSPACE")
    conv = comet_for_mlflow.Translator(False, api_key, path, None, None, "no")
    conv.prepare()

    assert len(list(tmp_path.glob("*.zip"))) == 1
