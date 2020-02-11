#!/bin/bash
env MLFLOW_S3_ENDPOINT_URL=http://localhost:9001 AWS_ACCESS_KEY_ID=minio AWS_SECRET_ACCESS_KEY=minio123 comet_for_mlflow --output-dir . --mlflow-store-uri http://localhost:5000 $@
