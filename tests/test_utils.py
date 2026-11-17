# -*- coding: utf-8 -*-

"""Tests for `comet_for_mlflow.utils`."""

import hashlib

from comet_for_mlflow.utils import get_comet_project_name, get_store_hash


class StoreWithoutId(object):
    """A store exposing none of the attributes used to identify it."""


class FileStoreLike(object):
    root_directory = "/data/mlruns"


def test_get_store_hash_without_store_id():
    assert get_store_hash(StoreWithoutId()) is None


def test_get_store_hash_with_store_id():
    expected = hashlib.sha1(b"/data/mlruns").hexdigest()[:6]

    assert get_store_hash(FileStoreLike()) == expected


def test_get_comet_project_name_without_store_id():
    assert get_comet_project_name(StoreWithoutId(), "My Exp") == "mlflow-my-exp"


def test_get_comet_project_name_with_store_id():
    expected_hash = hashlib.sha1(b"/data/mlruns").hexdigest()[:6]

    assert get_comet_project_name(
        FileStoreLike(), "My Exp"
    ) == "mlflow-my-exp-{}".format(expected_hash)
