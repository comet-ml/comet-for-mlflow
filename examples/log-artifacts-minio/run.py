#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import mlflow
from minio import Minio
from minio.error import BucketAlreadyOwnedByYou

os.environ.update(
    {
        "MLFLOW_S3_ENDPOINT_URL": "http://localhost:9001",
        "AWS_ACCESS_KEY_ID": "minio",
        "AWS_SECRET_ACCESS_KEY": "minio123",
    }
)


minioClient = Minio(
    "localhost:9001",
    access_key=os.environ["AWS_ACCESS_KEY_ID"],
    secret_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    secure=False,
)

try:
    minioClient.make_bucket("mlflow")
except BucketAlreadyOwnedByYou as err:
    print(err)

policy = {
    "Statement": [
        {
            "Action": [
                "s3:GetBucketLocation",
                "s3:ListBucket",
                "s3:ListBucketMultipartUploads",
                "s3:ListObjects",
            ],
            "Effect": "Allow",
            "Principal": {"AWS": ["*"]},
            "Resource": ["arn:aws:s3:::mlflow"],
        },
        {
            "Action": [
                "s3:AbortMultipartUpload",
                "s3:DeleteObject",
                "s3:GetObject",
                "s3:ListMultipartUploadParts",
                "s3:PutObject",
            ],
            "Effect": "Allow",
            "Principal": {"AWS": ["*"]},
            "Resource": ["arn:aws:s3:::mlflow/*"],
        },
    ],
    "Version": "2012-10-17",
}

# minioClient.set_bucket_policy('mlflow', json.dumps(policy))

# Then log a run

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Minio Example")

# Log some specialized artifacts
mlflow.log_artifact("files/test.svg")
mlflow.log_artifact("files/test.wav")
mlflow.log_artifact("files/test.txt")
mlflow.log_artifact("files/myfile")

# Log some directories too
mlflow.log_artifacts("files/")
