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

"""
Contains code to support multiple versions of MLFlow
"""
from mlflow.entities.view_type import ViewType

try:
    # MLFLOW version 1.4.0
    from mlflow.store.artifact.artifact_repository_registry import (  # noqa
        get_artifact_repository,
    )
except ImportError:
    # MLFLOW version < 1.4.0
    from mlflow.store.artifact_repository_registry import (  # noqa
        get_artifact_repository,
    )


def search_mlflow_store_experiments(mlflow_store):
    if hasattr(mlflow_store, "search_experiments"):
        # MLflow supports search for up to 50000 experiments, defined in
        # mlflow/store/tracking/__init__.py
        mlflow_experiments = mlflow_store.search_experiments(max_results=50000)
        # TODO: Check if there are more than 50000 experiments
        return list(mlflow_experiments)
    else:
        return list(mlflow_store.list_experiments())


def search_mlflow_store_runs(mlflow_store, experiment_id):
    if hasattr(mlflow_store, "search_runs"):
        # MLflow supports search for up to 50000 experiments, defined in
        # mlflow/store/tracking/__init__.py
        return mlflow_store.search_runs(
            [experiment_id],
            filter_string="",
            run_view_type=ViewType.ALL,
            max_results=50000,
        )
    else:
        return mlflow_store.list_run_infos(experiment_id, ViewType.ALL)


def get_mlflow_run_id(mlflow_run):
    if hasattr(mlflow_run, "info"):
        return mlflow_run.info.run_id
    else:
        return mlflow_run.run_id


def get_mlflow_model_name(mlflow_model):
    if hasattr(mlflow_model, "name"):
        return mlflow_model.name
    else:
        return mlflow_model.registered_model.name
