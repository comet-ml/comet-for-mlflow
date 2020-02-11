#!/bin/bash
env comet_for_mlflow --mlflow-store-uri sqlite:///db.sqlite --output-dir . $@
