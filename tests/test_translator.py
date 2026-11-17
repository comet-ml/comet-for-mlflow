# -*- coding: utf-8 -*-

"""Tests for `comet_for_mlflow.comet_for_mlflow.Translator`."""

import json
import re
from unittest import mock
from zipfile import ZipFile

import mlflow
import pytest
import responses
from comet_ml.config import get_config

from comet_for_mlflow import comet_for_mlflow
from comet_for_mlflow.comet_for_mlflow import Translator

API_KEY = "XXX"
WORKSPACE = "WORKSPACE"


@pytest.fixture
def no_network():
    """Fail any HTTP request, none of these tests should reach a server."""
    with responses.RequestsMock(assert_all_requests_are_fired=False) as requests_mock:
        yield requests_mock


@pytest.fixture
def reporting(monkeypatch):
    reporting_mock = mock.Mock()
    monkeypatch.setattr(comet_for_mlflow, "Reporting", reporting_mock)
    return reporting_mock


@pytest.fixture
def mlflow_store_uri(tmp_path, monkeypatch):
    """An empty MLflow store, also used as the working directory."""
    store_uri = "sqlite:///{}".format((tmp_path / "mlflow.db").as_posix())

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", store_uri)

    return store_uri


def log_mlflow_run(tmp_path):
    """Log a single MLflow run with known content and return its run ID."""
    artifacts_dir = tmp_path / "outputs"
    artifacts_dir.mkdir()
    (artifacts_dir / "test.txt").write_text("hello world!")

    with mlflow.start_run() as run:
        mlflow.log_param("param1", 42)
        mlflow.log_metric("foo", 0.5, step=0)
        mlflow.log_metric("foo", 1.5, step=1)
        mlflow.log_metric("foo", 2.5, step=2)
        mlflow.set_tag("team", "opik")
        mlflow.log_artifacts(artifacts_dir.as_posix())

    return run.info.run_id


def convert(mlflow_store_uri, output_dir, monkeypatch):
    """Convert the MLflow store into offline archives without uploading."""
    monkeypatch.setenv("COMET_WORKSPACE", WORKSPACE)
    monkeypatch.setattr(comet_for_mlflow, "API", mock.Mock())

    output_dir.mkdir()

    translator = Translator(
        False, API_KEY, output_dir.as_posix(), False, mlflow_store_uri, False
    )
    translator.prepare()

    return translator


def read_archive(archive_path):
    """Return the messages, the experiment metadata and the files of an archive."""
    with ZipFile(archive_path) as archive:
        messages = [
            json.loads(line)
            for line in archive.read("messages.json").decode("utf-8").splitlines()
        ]
        metadata = json.loads(archive.read("experiment.json"))
        files = {name: archive.read(name) for name in archive.namelist()}

    return messages, metadata, files


def payloads_with(messages, key):
    return [
        message["payload"][key] for message in messages if key in message["payload"]
    ]


class TestGetApiKeyOrLogin(object):
    @pytest.fixture
    def translator(self):
        # Skip __init__, it needs a MLflow store and a Comet API client
        translator = Translator.__new__(Translator)
        translator.config = get_config()
        return translator

    def test_prompts_for_api_key_when_none_is_configured(
        self, translator, monkeypatch, reporting, no_network
    ):
        monkeypatch.setattr(comet_for_mlflow, "get_api_key", lambda *args: None)
        input_mock = mock.Mock(return_value="TYPED-KEY")
        monkeypatch.setattr("builtins.input", input_mock)

        assert translator.get_api_key_or_login(None) == "TYPED-KEY"

        # A single prompt, for the API key only: no email, no username
        input_mock.assert_called_once_with("API Key: ")
        reporting.report.assert_called_once_with(
            translator.config, event_name="mlflow_existing_user", api_key="TYPED-KEY"
        )
        # No account lookup or creation
        assert len(no_network.calls) == 0

    def test_does_not_prompt_when_api_key_is_configured(
        self, translator, monkeypatch, reporting, no_network
    ):
        input_mock = mock.Mock()
        monkeypatch.setattr("builtins.input", input_mock)

        assert translator.get_api_key_or_login(API_KEY) == API_KEY

        input_mock.assert_not_called()
        reporting.report.assert_called_once_with(
            translator.config, event_name="mlflow_existing_user", api_key=API_KEY
        )
        assert len(no_network.calls) == 0

    def test_account_creation_is_gone(self):
        assert not hasattr(Translator, "create_or_login")


class TestArchiveContent(object):
    @pytest.fixture
    def archive(self, tmp_path, mlflow_store_uri, monkeypatch, reporting, no_network):
        run_id = log_mlflow_run(tmp_path)
        output_dir = tmp_path / "archives"

        convert(mlflow_store_uri, output_dir, monkeypatch)

        archive_paths = list(output_dir.glob("*.zip"))
        assert [path.name for path in archive_paths] == ["{}.zip".format(run_id)]

        messages, metadata, files = read_archive(archive_paths[0])
        return {
            "run_id": run_id,
            "messages": messages,
            "metadata": metadata,
            "files": files,
        }

    def test_params(self, archive):
        params = payloads_with(archive["messages"], "param")

        assert params == [{"paramName": "param1", "paramValue": "42", "step": None}]

    def test_metrics(self, archive):
        metrics = payloads_with(archive["messages"], "metric")

        assert [
            (metric["metricName"], metric["step"], metric["metricValue"])
            for metric in metrics
        ] == [("foo", 0, 0.5), ("foo", 1, 1.5), ("foo", 2, 2.5)]

    def test_tags(self, archive):
        others = {
            other["key"]: other["val"]
            for other in payloads_with(archive["messages"], "log_other")
        }

        assert others["team"] == "opik"
        assert others["mlflow.runId"] == archive["run_id"]
        assert others["mlflow.experimentName"] == "Default"
        assert others["Uploaded from"] == "MLFlow"

    def test_artifacts(self, archive):
        uploads = [
            message["payload"]
            for message in archive["messages"]
            if message["type"] == "file_upload"
        ]

        assert len(uploads) == 1
        assert uploads[0]["upload_type"] == "asset"
        assert uploads[0]["additional_params"]["fileName"] == "test.txt"
        assert uploads[0]["additional_params"]["extension"] == "txt"
        # The artifact content is stored in the archive
        assert archive["files"][uploads[0]["file_path"]] == b"hello world!"

    def test_experiment_metadata(self, archive):
        metadata = archive["metadata"]

        assert re.match(r"^mlflow-default-[0-9a-f]{6}$", metadata["project_name"])
        assert metadata["workspace"] == WORKSPACE
        assert metadata["offline_id"] == archive["run_id"]
        assert metadata["start_time"] is not None
        assert metadata["stop_time"] >= metadata["start_time"]


def test_run_without_source_and_user_tags_is_converted(
    tmp_path, mlflow_store_uri, monkeypatch, reporting, no_network
):
    run_id = log_mlflow_run(tmp_path)

    # MLflow sets these tags itself, but runs logged through the REST API or
    # other clients are not guaranteed to have them
    client = mlflow.tracking.MlflowClient()
    client.delete_tag(run_id, "mlflow.source.name")
    client.delete_tag(run_id, "mlflow.user")

    output_dir = tmp_path / "archives"
    translator = convert(mlflow_store_uri, output_dir, monkeypatch)

    assert translator.summary["runs"] == 1
    reporting.report.assert_called_once_with(
        translator.config, event_name="mlflow_existing_user", api_key=API_KEY
    )

    messages, _, _ = read_archive(output_dir / "{}.zip".format(run_id))

    assert payloads_with(messages, "fileName") == [None]
    assert [details["user"] for details in payloads_with(messages, "env_details")] == [
        None
    ]
    # The rest of the run is still imported
    assert len(payloads_with(messages, "metric")) == 3
