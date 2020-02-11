#!/usr/bin/env python
# -*- coding: utf-8 -*-

import mlflow

# Log some specialized artifacts
mlflow.log_artifact("files/test.svg")
mlflow.log_artifact("files/test.wav")
mlflow.log_artifact("files/test.txt")
mlflow.log_artifact("files/myfile")

# Log some directories too
mlflow.log_artifacts("files/")
