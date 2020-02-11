#!/usr/bin/env python
# -*- coding: utf-8 -*-

import mlflow

mlflow.set_experiment("My custom experiment")

mlflow.log_metric("loss", 0.42)
mlflow.log_param("foo", "bar")
