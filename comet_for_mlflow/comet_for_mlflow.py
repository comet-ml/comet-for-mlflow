#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Comet.ml Team.
#
# This file is part of Comet-For-MLFlow
# (see https://github.com/comet-ml/comet-for-mlflow).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

"""Main module."""

from __future__ import print_function

import logging
import os.path
import shutil
import sys
import tempfile
import traceback
from os.path import abspath
from typing import Optional
from zipfile import ZipFile

from comet_ml import API
from comet_ml.comet import format_url
from comet_ml.config import get_api_key, get_config
from comet_ml.connection import Reporting, get_comet_api_client, url_join
from comet_ml.exceptions import CometRestApiException, NotFound
from comet_ml.offline import upload_single_offline_experiment
from mlflow.entities.run_tag import RunTag
from mlflow.entities.view_type import ViewType
from mlflow.tracking import _get_store
from mlflow.tracking._model_registry.utils import _get_store as get_model_registry_store
from mlflow.tracking.registry import UnsupportedModelRegistryStoreURIException
from tabulate import tabulate
from tqdm import tqdm

from .file_writer import JsonLinesFile
from .utils import (
    get_comet_project_name,
    get_store_id,
    save_api_key,
    walk_run_artifacts,
    write_comet_experiment_metadata_file,
)

try:
    # Python 2
    input = raw_input
except NameError:
    # Python 3
    pass


try:
    # MLFLOW version 1.4.0
    from mlflow.store.artifact.artifact_repository_registry import (
        get_artifact_repository,
    )
except ImportError:
    # MLFLOW version < 1.4.0
    from mlflow.store.artifact_repository_registry import get_artifact_repository

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger()

# Install a global exception hook
def except_hook(exc_type, exc_value, exc_traceback):
    Reporting.report(
        "mlflow_error",
        err_msg="".join(traceback.format_exception(exc_type, exc_value, exc_traceback)),
    )
    sys.__excepthook__(exc_type, exc_value, exc_traceback)


sys.excepthook = except_hook

BANNER = r""" __   __         ___ ___     ___  __   __                 ___       __
/  ` /  \  |\/| |__   |  __ |__  /  \ |__) __  |\/| |    |__  |    /  \ |  |
\__, \__/  |  | |___  |     |    \__/ |  \     |  | |___ |    |___ \__/ |/\|

"""


class Translator(object):
    def __init__(
        self,
        upload_experiment,
        api_key,
        output_dir,
        force_reupload,
        mlflow_store_uri,
        answer,
        email,
    ):
        # type: (bool, str, str, bool, str, Optional[bool], str) -> None
        self.answer = answer
        self.email = email
        self.config = get_config()

        # Display the start banner
        LOGGER.info(BANNER)

        # get_api_key
        self.api_key, self.token = self.get_api_key_or_login(api_key)
        # May not need this always?
        self.api_client = API(self.api_key, cache=False)

        self.workspace = self.config["comet.workspace"]

        # Get a workspace
        if not self.workspace:
            details = self.api_client.get_account_details()

            self.workspace = details["defaultWorkspaceName"]

        if output_dir is None:
            output_dir = tempfile.mkdtemp()

        # MLFlow conversion
        self.store = _get_store(mlflow_store_uri)
        try:
            self.model_registry_store = get_model_registry_store(mlflow_store_uri)
        except UnsupportedModelRegistryStoreURIException:
            self.model_registry_store = None

        # Most of list_experiments returns a list anyway
        self.mlflow_experiments = list(self.store.list_experiments())
        self.len_experiments = len(self.mlflow_experiments)  # We start counting at 0

        self.summary = {
            "experiments": 0,
            "runs": 0,
            "tags": 0,
            "params": 0,
            "metrics": 0,
            "artifacts": 0,
        }

        self.upload_experiment = upload_experiment
        self.output_dir = output_dir
        self.force_reupload = force_reupload
        self.mlflow_store_uri = mlflow_store_uri

    def prepare(self):
        LOGGER.info("Starting Comet Extension for MLFlow")

        LOGGER.info("")
        LOGGER.info("Preparing data locally from: %r", get_store_id(self.store))
        LOGGER.info("You will have an opportunity to review.")
        LOGGER.info("")

        prepared_data = []

        # First prepare all the data except the metadata as we need a project name
        for experiment_number, experiment in enumerate(self.mlflow_experiments):

            experiment_name = experiment.experiment_id
            if experiment.name:
                experiment_name = experiment.name

            LOGGER.info(
                "# Preparing experiment %d/%d: %s",
                experiment_number + 1,
                self.len_experiments,
                experiment_name,
            )
            LOGGER.debug(
                "# Preparing experiment %d/%d: %r",
                experiment_number + 1,
                self.len_experiments,
                experiment,
            )
            self.summary["experiments"] += 1
            try:
                prepared_runs = list(self.prepare_mlflow_exp(experiment))

                prepared_data.append({"experiment": experiment, "runs": prepared_runs})
                LOGGER.info("")
            except Exception:
                LOGGER.exception(
                    "# Error preparing experiment %d/%d: %r",
                    experiment_number + 1,
                    self.len_experiments,
                    experiment,
                )
                LOGGER.error("")
                Reporting.report(
                    "mlflow_error", api_key=self.api_key, err_msg=traceback.format_exc()
                )

        table = [
            ("Experiments", "Projects", self.summary["experiments"]),
            ("Runs", "Experiments", self.summary["runs"]),
            ("Tags", "Others", self.summary["tags"]),
            ("Parameters", "Parameters", self.summary["params"]),
            ("Metrics", "Metrics", self.summary["metrics"]),
            ("Artifacts", "Assets", self.summary["artifacts"]),
        ]
        LOGGER.info(
            tabulate(
                table,
                headers=["MLFlow name:", "Comet.ml name:", "Prepared count:"],
                tablefmt="presto",
            )
        )

        LOGGER.info("")
        LOGGER.info("All prepared data has been saved to: %s", abspath(self.output_dir))

        # Upload or not?
        print("")
        if self.answer is None:
            upload = input("Upload prepared data to Comet.ml? [y/N] ") in ("Y", "y")
        else:
            upload = self.answer
        print("")

        should_upload = self.upload_experiment
        should_upload = should_upload and upload

        if should_upload:
            self.upload(prepared_data)
        else:
            self.save_locally(prepared_data)

        LOGGER.info("")
        LOGGER.info(
            "If you need support, you can contact us at http://chat.comet.ml/ or https://comet.ml/docs/quick-start/#getting-support"
        )
        LOGGER.info("")

    def prepare_mlflow_exp(
        self, exp,
    ):
        runs_info = self.store.list_run_infos(exp.experiment_id, ViewType.ALL)
        len_runs = len(runs_info)

        for run_number, run_info in enumerate(runs_info):
            try:
                run_id = run_info.run_id
                run = self.store.get_run(run_id)
                LOGGER.info(
                    "## Preparing run %d/%d [%s]", run_number + 1, len_runs, run_id,
                )
                LOGGER.debug(
                    "## Preparing run %d/%d: %r", run_number + 1, len_runs, run
                )

                offline_archive = self.prepare_single_mlflow_run(run, exp.name)

                if offline_archive:
                    self.summary["runs"] += 1
                    yield (run, offline_archive)
            except Exception:
                LOGGER.exception(
                    "## Error preparing run %d/%d [%s]",
                    run_number + 1,
                    len_runs,
                    run_id,
                )
                LOGGER.error("")
                Reporting.report(
                    "mlflow_error", api_key=self.api_key, err_msg=traceback.format_exc()
                )

    def prepare_single_mlflow_run(self, run, original_experiment_name):
        self.tmpdir = tempfile.mkdtemp()

        if not run.info.end_time:
            # Seems to be the case when using the optimizer, some runs doesn't have an end_time
            LOGGER.warning("### Skipping run, no end time")
            return False

        run_start_time = run.info.start_time

        messages_file_path = os.path.join(self.tmpdir, "messages.json")

        with JsonLinesFile(messages_file_path, self.tmpdir) as json_writer:
            # Get mlflow tags
            tags = run.data.tags

            if not tags:
                tags = {}

            LOGGER.debug("### Preparing env details")
            json_writer.write_filename_msg(tags["mlflow.source.name"], run_start_time)

            json_writer.write_user_msg(tags["mlflow.user"], run_start_time)

            LOGGER.debug("### Preparing git details")
            json_writer.write_git_meta_msg(
                tags.get("mlflow.source.git.commit"),
                tags.get("mlflow.source.git.repoURL"),
                run_start_time,
            )

            # Import any custom name
            if tags.get("mlflow.runName"):
                tags["Name"] = tags["mlflow.runName"]

            # Save the run id as tag too as Experiment id can be different in case
            # of multiple uploads
            tags["mlflow.runId"] = run.info.run_id

            if tags.get("mlflow.parentRunId"):
                base_url = url_join(
                    self.api_client.server_url, "/api/experiment/redirect"
                )
                tags["mlflow.parentRunUrl"] = format_url(
                    base_url, experimentKey=tags["mlflow.parentRunId"]
                )

            # Save the original MLFlow experiment name too as Comet.ml project might
            # get renamed
            tags["mlflow.experimentName"] = original_experiment_name

            LOGGER.debug("### Importing tags")
            for tag_name, tag_value in tags.items():
                LOGGER.debug("#### Tag %r: %r", tag_name, tag_value)
                json_writer.write_log_other_msg(tag_name, tag_value, run_start_time)

                self.summary["tags"] += 1

            # Mark the experiments has being uploaded from MLFlow
            json_writer.write_log_other_msg("Uploaded from", "MLFlow", run_start_time)

            LOGGER.debug("### Importing params")
            for param_key, param_value in run.data.params.items():
                LOGGER.debug("#### Param %r: %r", param_key, param_value)

                json_writer.write_param_msg(param_key, param_value, run_start_time)

                self.summary["params"] += 1

            LOGGER.debug("### Importing metrics")
            for metric in run.data._metric_objs:
                metric_history = self.store.get_metric_history(
                    run.info.run_id, metric.key
                )
                # Check if all steps are uniques, if not we don't pass any so the backend
                # fallback to the unique timestamp
                steps = [mh.step for mh in metric_history]

                use_steps = True

                if len(set(steps)) != len(metric_history):
                    LOGGER.warning(
                        "Non-unique steps detected, importing metrics with wall time instead"
                    )
                    use_steps = False

                for mh in metric_history:
                    if use_steps:
                        step = mh.step
                    else:
                        step = None

                    json_writer.write_metric_msg(mh.key, step, mh.timestamp, mh.value)

                    self.summary["metrics"] += 1

                LOGGER.debug("#### Metric %r: %r", metric.key, metric_history)

            LOGGER.debug("### Importing artifacts")
            artifact_store = get_artifact_repository(run.info.artifact_uri)

            # List all the registered models if possible
            models_prefixes = {}
            if self.model_registry_store:
                query = "run_id='%s'" % run.info.run_id
                registered_models = self.model_registry_store.search_model_versions(
                    query
                )

                for model in registered_models:
                    model_relpath = os.path.relpath(model.source, run.info.artifact_uri)
                    models_prefixes[model_relpath] = model

            for artifact in walk_run_artifacts(artifact_store):
                artifact_path = artifact.path

                LOGGER.debug("### Artifact %r: %r", artifact, artifact_path)
                # Check if the file is an visualization or not
                _, extension = os.path.splitext(artifact_path)

                local_artifact_path = artifact_store.download_artifacts(artifact_path)

                self.summary["artifacts"] += 1

                # Check if it's belonging to one of the registered model
                matching_model = None
                for model_prefix, model in models_prefixes.items():
                    if artifact_path.startswith(model_prefix):
                        matching_model = model
                        # We should match at most one model
                        break

                if matching_model:
                    json_writer.log_artifact_as_model(
                        local_artifact_path,
                        artifact_path,
                        run_start_time,
                        matching_model.registered_model.name,
                    )
                else:
                    json_writer.log_artifact_as_asset(
                        local_artifact_path, artifact_path, run_start_time,
                    )

        return self.compress_archive(run.info.run_id)

    def upload(self, prepared_data):
        LOGGER.info("# Start uploading data to Comet.ml")

        all_project_names = []

        with tqdm(total=self.summary["runs"]) as pbar:
            for experiment_data in prepared_data:
                experiment = experiment_data["experiment"]

                project_name = self.get_or_create_comet_project(experiment)

                # Sync the experiment note
                project_note = experiment.tags.get("mlflow.note.content", None)
                if project_note:
                    note_template = (
                        u"/!\\ This project notes has been copied from MLFlow. It might be overwritten if you run comet_for_mlflow again/!\\ \n%s"
                        % project_note
                    )
                    # We don't support Unicode project notes yet
                    self.api_client.set_project_notes(
                        self.workspace,
                        project_name,
                        note_template.encode("ascii", "backslashreplace"),
                    )

                all_project_names.append(project_name)

                runs = experiment_data["runs"]

                for mlflow_run, archive_path in runs:
                    write_comet_experiment_metadata_file(
                        mlflow_run, project_name, archive_path, self.workspace
                    )

                    upload_single_offline_experiment(
                        archive_path,
                        self.api_key,
                        force_reupload=self.force_reupload,
                        display_level="debug",
                    )

                    pbar.update(1)

        LOGGER.info("")
        LOGGER.info(
            "Explore your experiment data on Comet.ml with the following links:",
        )
        if len(all_project_names) < 6:
            for project_name in all_project_names:
                project_url = url_join(
                    self.api_client._get_url_server(),
                    self.workspace + "/",
                    project_name,
                    loginToken=self.token,
                )
                LOGGER.info("\t- %s", project_url)
        else:
            url = url_join(
                self.api_client._get_url_server(),
                self.workspace,
                query="mlflow",
                loginToken=self.token,
            )
            LOGGER.info("\t- %s", url)

        LOGGER.info(
            "Get deeper instrumentation by adding Comet SDK to your project: https://comet.ml/docs/python-sdk/mlflow/"
        )
        LOGGER.info("")

    def save_locally(self, prepared_data):
        for experiment_data in prepared_data:
            experiment = experiment_data["experiment"]

            project_name = get_comet_project_name(self.store, experiment.name)

            runs = experiment_data["runs"]

            for mlflow_run, archive_path in runs:
                write_comet_experiment_metadata_file(
                    mlflow_run, project_name, archive_path, self.workspace
                )

        LOGGER.info("Data not uploaded. To upload later run:")
        LOGGER.info("   comet upload %s/*.zip", abspath(self.output_dir))
        LOGGER.info("")
        LOGGER.info("To get a preview of what was prepared, run:")
        LOGGER.info("   comet offline %s/*.zip", abspath(self.output_dir))

    def compress_archive(self, run_id):
        filepath = os.path.join(self.output_dir, "%s.zip" % run_id)
        zipfile = ZipFile(filepath, "w")

        for file in os.listdir(self.tmpdir):
            zipfile.write(os.path.join(self.tmpdir, file), file)

        zipfile.close()

        shutil.rmtree(self.tmpdir)

        return filepath

    def create_and_save_comet_project(self, exp, tag_name):
        # Create a Comet project with the name and description
        project_name = get_comet_project_name(self.store, exp.name)

        # Check if the project exists already
        try:
            project = self.api_client.get_project(self.workspace, project_name)
            if not project:
                raise NotFound("POST", {})
            project_id = project["projectId"]
        except NotFound:
            project = self.api_client.create_project(
                self.workspace, project_name, public=False
            )

            project_id = project["projectId"]

        # Save the project id to the experiment tags
        self.store.set_experiment_tag(exp.experiment_id, RunTag(tag_name, project_id))

        return project_name

    def get_or_create_comet_project(self, exp):
        # Check if the mlflow experiment has already a project ID for this workspace
        tag_name = "comet-project-{}".format(self.workspace)

        project_id = None

        if tag_name in exp.tags:
            project_id = exp.tags[tag_name]

            # Check if the project exists
            try:
                project = self.api_client.get_project_by_id(project_id)
                if not project:
                    raise NotFound("POST", {})
                return project["projectName"]
            except (NotFound):
                # A previous project ID has been saved but don't exists anymore (at
                # least in this environment), recreate it
                return self.create_and_save_comet_project(exp, tag_name)
        else:
            return self.create_and_save_comet_project(exp, tag_name)

    def create_or_login(self):
        auth_api_client = get_comet_api_client(None)
        LOGGER.info("Please create a free Comet account with your email.")
        if self.email is None:
            email = input("Email: ")
            print("")
        else:
            email = self.email

        # Check if the email exists in the system
        try:
            status = auth_api_client.check_email(email, "comet-for-mlflow")
        except CometRestApiException as e:
            status = e.response.status_code

        # We hit rate-limitting
        if status == 429:
            # We hit rate-limitting
            LOGGER.error(
                "Too many user login requests, please try again in one minute."
            )
            sys.exit(1)

        new_account = status != 200

        if new_account:
            LOGGER.info("Please enter a username for your new account.")
            username = input("Username: ")
            print("")
            new_account = auth_api_client.create_user(
                email, username, "comet-for-mlflow"
            )

            Reporting.report("mlflow_new_user", api_key=new_account["apiKey"])

            LOGGER.info(
                "A Comet.ml account has been created for you and an email was sent to you to setup your password later."
            )
            save_api_key(new_account["apiKey"])
            LOGGER.info(
                "Your Comet API Key has been saved to ~/.comet.ini, it is also available on your Comet.ml dashboard."
            )
            return (
                new_account["apiKey"],
                new_account["token"],
            )
        else:
            LOGGER.info(
                "An account already exists for this account, please input your API Key below (you can find it in your Settings page, https://comet.ml/docs/quick-start/#getting-your-comet-api-key):"
            )
            api_key = input("API Key: ")

            Reporting.report("mlflow_existing_user", api_key=api_key)
            return (
                api_key,
                None,
            )

    def get_api_key_or_login(self, api_key):
        # ok
        api_key = get_api_key(api_key, self.config)

        if api_key is None:
            return self.create_or_login()

        Reporting.report("mlflow_existing_user", api_key=api_key)

        return (api_key, None)
