# -*- coding: utf-8 -*-

"""Tests for the `comet_for_mlflow` console script."""

import sys
from unittest import mock

from comet_for_mlflow import cli


def test_email_flag_is_accepted_and_ignored(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "comet_for_mlflow",
            "--email",
            "someone@example.com",
            "--no-upload",
            "--api-key",
            "XXX",
        ],
    )

    with mock.patch.object(cli, "Translator") as translator_class:
        assert cli.main() == 0

    # The email is not forwarded anywhere
    translator_class.assert_called_once_with(False, "XXX", None, False, None, None)
    translator_class.return_value.prepare.assert_called_once_with()
