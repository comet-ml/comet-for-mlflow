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

import configparser
import hashlib
import json
import os.path
import re
from zipfile import ZipFile


def get_store_id(store):
    if hasattr(store, "root_directory"):
        return store.root_directory
    elif hasattr(store, "db_uri"):
        return store.db_uri
    elif hasattr(store, "get_host_creds"):
        return store.get_host_creds().host
    else:
        return None


def get_store_hash(store):
    store_id = get_store_id(store)
    return hashlib.sha1(store_id.encode("utf-8")).hexdigest()[:6]


def clean_project_name(project_name):
    project_name = re.sub("[^A-Za-z0-9]", "-", project_name)
    project_name = re.sub("-+", "-", project_name)

    return project_name.rstrip("-").lstrip("-").lower()


def get_comet_project_name(store, exp_name):
    store_hash = get_store_hash(store)
    return clean_project_name("mlflow-{}-{}".format(exp_name, store_hash))


def walk_run_artifacts(artifact_store):
    # None is for the root
    nodes = [None]

    while nodes:
        current_node = nodes.pop()

        artifact_entities = artifact_store.list_artifacts(current_node)

        for artifact in artifact_entities:
            if artifact.is_dir:
                nodes.append(artifact.path)
            else:
                yield artifact


def write_comet_experiment_metadata_file(
    mlflow_run, project_name, archive_path, workspace=None
):
    run_start_time = mlflow_run.info.start_time
    run_end_time = mlflow_run.info.end_time

    # MLFlow run_id are also GUID so simply reuse them
    data = {
        "auto_metric_logging": True,
        "auto_output_logging": None,  # MLFlow doesn't log output
        "auto_param_logging": True,
        "disabled": False,
        "feature_toggles_overrides": {},
        "log_code": False,  # MLFlow doesn't log code
        "log_env_details": False,  # MLFlow doesn't log env
        "log_git_metadata": True,
        "log_graph": True,  # MLFlow doesn't log graph
        "parse_args": True,  # MLFlow doesn't log args
        "project_name": project_name,
        "start_time": run_start_time,
        "stop_time": run_end_time,
        "tags": [],
        "workspace": workspace,
        "offline_id": mlflow_run.info.run_id,
    }

    zipfile = ZipFile(archive_path, "a")
    zipfile.writestr("experiment.json", json.dumps(data))
    zipfile.close()


def save_api_key(api_key):
    config_path = os.path.expanduser(os.path.join("~", ".comet.config"))

    with open(config_path, "wt") as config_file:
        config_file.write("# Config file for Comet.ml\n")
        config_file.write(
            "# For help see https://www.comet.com/docs/python-sdk/getting-started/\n"
        )
        config_file.write("")

    config = configparser.ConfigParser()
    if not config.has_section("comet"):
        config.add_section("comet")
    config.set("comet", "api_key", api_key)

    with open(config_path, "a") as config_file:
        config.write(config_file)


def validate_step_consistency(metrics_history):
    """
    Checks if all Metric objects with the same 'step' have the same 'value'.
    Returns `True` if consistent, `False` if not.

    This can happen pretty commonly with the huggingface Trainer (depending on logging_strategy and eval_strategy).
    However, the value still remains the same.

    Args:
        metrics_history `List[mlflow.entities.metric.Metric]`: A list of Metric objects.
    """
    step_map = {}

    for metric in metrics_history:
        if metric.step in step_map:
            if step_map[metric.step] != metric.value:
                return False
        else:
            step_map[metric.step] = metric.value
            
    return True