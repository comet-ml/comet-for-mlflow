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

"""Console script for comet_for_mlflow."""
import argparse
import sys

from .comet_for_mlflow import Translator


def main():
    """Console script for comet_for_mlflow."""
    parser = argparse.ArgumentParser()

    # Upload or not
    upload_parser = parser.add_mutually_exclusive_group(required=False)
    upload_parser.add_argument(
        "--upload",
        dest="upload",
        action="store_true",
        help="Automatically upload the prepared experiments to comet.ml; defaults to True.",
    )
    upload_parser.add_argument(
        "--no-upload",
        dest="upload",
        action="store_false",
        help="Do not upload the prepared experiments to comet.ml; will not create comet.ml projects",
    )
    parser.set_defaults(upload=True)

    parser.add_argument(
        "--api-key",
        help="Set the Comet API key; required with --upload (the default); can also be configured in the usual places",
    )
    parser.add_argument(
        "--mlflow-store-uri",
        help="Set the MLFlow store uri. The MLFlow store uri to used to retrieve MLFlow runs, given directly to MLFlow, and supports all MLFlow schemes (file:// or SQLAlchemy-compatible database connection strings). If not set, reads MLFLOW_TRACKING_URI environment variable",
    )

    parser.add_argument(
        "--output-dir",
        help="set the directory to store prepared runs; only relevant with --no-upload",
    )
    parser.add_argument(
        "--force-reupload",
        action="store_true",
        default=False,
        help="Force reupload of prepared experiments that were previously uploaded",
    )
    command_group = parser.add_mutually_exclusive_group()
    command_group.add_argument(
        "-y",
        "--yes",
        action="store_true",
        dest="answer",
        default=None,
        help="Answer all yes/no questions automatically with 'yes'",
    )
    command_group.add_argument(
        "-n",
        "--no",
        action="store_false",
        dest="answer",
        default=None,
        help="Answer all yes/no questions automatically with 'no'",
    )
    parser.add_argument(
        "--email", help="Set email address if needed for creating a comet.ml account",
    )

    args = parser.parse_args()

    converter = Translator(
        args.upload,
        args.api_key,
        args.output_dir,
        args.force_reupload,
        args.mlflow_store_uri,
        args.answer,
        args.email,
    )
    converter.prepare()
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
