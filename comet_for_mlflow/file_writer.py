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

"""File writer."""

from __future__ import print_function

import json
import logging
import os.path
import shutil
import tempfile
import uuid

LOGGER = logging.getLogger()


def generate_guid():
    """ Generate a GUID
    """
    return uuid.uuid4().hex


class JsonLinesFile(object):
    """ A context manager to write a JSON Lines file, also called newline-delimited JSON.
    """

    def __init__(self, filepath, tmpdir):
        self.filepath = filepath
        self.tmpdir = tmpdir
        self._file = None

    def __enter__(self):
        self._file = open(self.filepath, "w")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._file.flush()
        self._file.close()
        self._file = None
        return False

    def write_line_data(self, data):
        json.dump(data, self._file)
        self._file.write("\n")

    def write_filename_msg(self, source, timestamp):
        data = {
            "payload": {
                "code": None,
                "context": None,
                "env_details": None,
                "fileName": source,
                "git_meta": None,
                "gpu_static_info": None,
                "graph": None,
                "html": None,
                "htmlOverride": None,
                "installed_packages": None,
                "local_timestamp": timestamp,
                "log_dependency": None,
                "log_other": None,
                "log_system_info": None,
                "metric": None,
                "os_packages": None,
                "param": None,
                "params": None,
                "stderr": None,
                "stdout": None,
            },
            "type": "ws_msg",
        }

        self.write_line_data(data)

    def write_user_msg(self, user, timestamp):
        data = {
            "payload": {
                "code": None,
                "context": None,
                "env_details": {
                    "command": None,
                    "hostname": None,
                    "ip": None,
                    "network_interfaces_ips": None,
                    "os": None,
                    "os_type": None,
                    "pid": None,
                    "python_exe": None,
                    "python_version": None,
                    "python_version_verbose": None,
                    "user": user,
                },
                "fileName": None,
                "git_meta": None,
                "gpu_static_info": None,
                "graph": None,
                "html": None,
                "htmlOverride": None,
                "installed_packages": None,
                "local_timestamp": timestamp,
                "log_dependency": None,
                "log_other": None,
                "log_system_info": None,
                "metric": None,
                "os_packages": None,
                "param": None,
                "params": None,
                "stderr": None,
                "stdout": None,
            },
            "type": "ws_msg",
        }
        self.write_line_data(data)

    def write_git_meta_msg(self, git_commit, git_origin, timestamp):
        data = {
            "payload": {
                "code": None,
                "context": None,
                "env_details": None,
                "fileName": None,
                "git_meta": {
                    "branch": None,
                    "origin": git_origin,
                    "parent": git_commit,
                    "repo_name": None,
                    "root": None,
                    "status": None,
                    "user": None,
                },
                "gpu_static_info": None,
                "graph": None,
                "html": None,
                "htmlOverride": None,
                "installed_packages": None,
                "local_timestamp": timestamp,
                "log_dependency": None,
                "log_other": None,
                "log_system_info": None,
                "metric": None,
                "os_packages": None,
                "param": None,
                "params": None,
                "stderr": None,
                "stdout": None,
            },
            "type": "ws_msg",
        }

        self.write_line_data(data)

    def write_log_other_msg(self, other_name, other_value, timestamp):
        data = {
            "payload": {
                "code": None,
                "context": None,
                "env_details": None,
                "fileName": None,
                "git_meta": None,
                "gpu_static_info": None,
                "graph": None,
                "html": None,
                "htmlOverride": None,
                "installed_packages": None,
                "local_timestamp": timestamp,
                "log_dependency": None,
                "log_other": {"key": other_name, "val": other_value},
                "log_system_info": None,
                "metric": None,
                "os_packages": None,
                "param": None,
                "params": None,
                "stderr": None,
                "stdout": None,
            },
            "type": "ws_msg",
        }

        self.write_line_data(data)

    def write_param_msg(self, param_name, param_value, timestamp):
        data = {
            "payload": {
                "code": None,
                "context": None,
                "env_details": None,
                "fileName": None,
                "git_meta": None,
                "gpu_static_info": None,
                "graph": None,
                "html": None,
                "htmlOverride": None,
                "installed_packages": None,
                "local_timestamp": timestamp,
                "log_dependency": None,
                "log_other": None,
                "log_system_info": None,
                "metric": None,
                "os_packages": None,
                "param": {
                    "paramName": param_name,
                    "paramValue": param_value,
                    "step": None,
                },
                "params": None,
                "stderr": None,
                "stdout": None,
            },
            "type": "ws_msg",
        }

        self.write_line_data(data)

    def write_metric_msg(self, metric_name, step, timestamp, metric_value):
        data = {
            "payload": {
                "code": None,
                "context": None,
                "env_details": None,
                "fileName": None,
                "git_meta": None,
                "gpu_static_info": None,
                "graph": None,
                "html": None,
                "htmlOverride": None,
                "installed_packages": None,
                "local_timestamp": timestamp,
                "log_dependency": None,
                "log_other": None,
                "log_system_info": None,
                "metric": {
                    "epoch": 0,
                    "metricName": metric_name,
                    "metricValue": metric_value,
                    "step": step,
                },
                "os_packages": None,
                "param": None,
                "params": None,
                "stderr": None,
                "stdout": None,
            },
            "type": "ws_msg",
        }

        self.write_line_data(data)

    def log_artifact_as_visualization(
        self, artifact_path, artifact_name, timestamp, figure_counter
    ):
        image_id = generate_guid()

        upload_file = self.get_temp_filename(artifact_path)

        data = {
            "payload": {
                "additional_params": {
                    "context": None,
                    "figCounter": figure_counter,
                    "figName": artifact_name,
                    "imageId": image_id,
                    "overwrite": False,
                    "runId": None,
                    "step": None,
                },
                "clean": True,
                "file_path": os.path.basename(upload_file),
                "local_timestamp": timestamp,
                "upload_type": "visualization",
            },
            "type": "file_upload",
        }

        self.write_line_data(data)

    def log_artifact_as_model(
        self, artifact_path, artifact_name, timestamp, model_name
    ):

        _, extension = os.path.splitext(
            artifact_path
        )  # TODO: Support extension less file names?

        asset_id = generate_guid()

        upload_file = self.get_temp_filename(artifact_path)

        data = {
            "payload": {
                "additional_params": {
                    "assetId": asset_id,
                    "context": None,
                    "extension": extension[1:],
                    "fileName": artifact_name,
                    "groupingName": model_name,
                    "overwrite": False,
                    "runId": None,
                    "step": None,
                    "type": "model-element",
                },
                "clean": True,
                "file_path": os.path.basename(upload_file),
                "local_timestamp": timestamp,
                "metadata": {},
                "upload_type": "model-element",
            },
            "type": "file_upload",
        }

        self.write_line_data(data)

    def log_artifact_as_asset(self, artifact_path, artifact_name, timestamp):

        _, extension = os.path.splitext(
            artifact_path
        )  # TODO: Support extension less file names?

        asset_id = generate_guid()

        upload_file = self.get_temp_filename(artifact_path)

        data = {
            "payload": {
                "additional_params": {
                    "assetId": asset_id,
                    "context": None,
                    "extension": extension[1:],
                    "fileName": artifact_name,
                    "overwrite": False,
                    "runId": None,
                    "step": None,
                },
                "clean": True,
                "file_path": os.path.basename(upload_file),
                "local_timestamp": timestamp,
                "upload_type": "asset",
            },
            "type": "file_upload",
        }

        self.write_line_data(data)

    def get_temp_filename(self, artifact_path):
        tmpfile = tempfile.NamedTemporaryFile(delete=False, dir=self.tmpdir)
        shutil.copyfile(artifact_path, tmpfile.name)
        upload_file = tmpfile.name
        return upload_file

    def log_artifact_as_audio(self, artifact_path, artifact_name, timestamp):
        asset_id = generate_guid()

        upload_file = self.get_temp_filename(artifact_path)

        data = {
            "payload": {
                "additional_params": {
                    "assetId": asset_id,
                    "context": None,
                    "fileName": artifact_name,
                    "overwrite": False,
                    "runId": None,
                    "step": None,
                    "type": "audio",
                },
                "clean": True,
                "file_path": os.path.basename(upload_file),
                "local_timestamp": timestamp,
                "upload_type": "audio",
            },
            "type": "file_upload",
        }

        self.write_line_data(data)
