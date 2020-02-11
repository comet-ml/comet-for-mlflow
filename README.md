# Comet-For-MLFlow Extension

[![image](https://img.shields.io/pypi/v/comet-for-mlflow.svg)](https://pypi.org/project/comet-for-mlflow/)
[![CI Build](https://github.com/comet-ml/comet-for-mlflow/workflows/CI%20Build/badge.svg)](https://github.com/comet-ml/comet-for-mlflow/actions) [![Updates](https://pyup.io/repos/github/comet-ml/comet-for-mlflow/shield.svg)](https://pyup.io/repos/github/comet-ml/comet-for-mlflow/)


The Comet-For-MLFlow extension is a CLI that maps MLFlow experiment runs to Comet experiments. This extension allows you to see your existing experiments in the Comet.ml UI which provides authenticated access to experiment results, dramatically improves the performance for high volume experiment runs, and provides richer charting and visualization options.

This extension will synchronize previous MLFlow experiment runs with all runs tracked with [Comet's Python SDK with MLFlow support](https://comet.ml/docs/python-sdk/mlflow/), for deeper experiment instrumentation and improved logging, visibility, project organization and access management.

The Comet-For-MLFlow Extension is available as free open-source software, released under GNU General Public License v3. The extension can be used with existing Comet.ml accounts or with a new, free Individual account.

# Installation

```bash
pip install comet-for-mlflow
```

If you install `comet-for-mlflow` in a different Python environment than the one you used to generate mlflow runs, please ensure that you use the same mlflow version in both environments.

# Basic usage

For automatically synchronizing MLFlow runs in their default storage location (`./mlruns`) with Comet.ml, run:

```bash
comet_for_mlflow --api-key $COMET_API_KEY --rest-api-key $COMET_REST_API_KEY
```

If you'd like to review the mapping of MLFlow runs in their default storage location without synchronizing them with Comet.ml automatically, you can run:


```bash
comet_for_mlflow --no-upload
```

After review, you can upload the mapped MLFlow runs with:

```bash
comet upload /path/to/archive.zip
```

# Example

```
 __   __         ___ ___     ___  __   __                 ___       __
/  ` /  \  |\/| |__   |  __ |__  /  \ |__) __  |\/| |    |__  |    /  \ |  |
\__, \__/  |  | |___  |     |    \__/ |  \     |  | |___ |    |___ \__/ |/\|


Please create a free Comet account with your email.
Email: kristen.stewart@example.com

Please enter a username for your new account.
Username: kstewart

A Comet.ml account has been created for you and an email was sent to you to setup your password later.
Your Comet API Key has been saved to ~/.comet.ini, it is also available on your Comet.ml dashboard.
Starting Comet Extension for MLFlow

Preparing data locally from: '/home/ks/project/mlruns'
You will have an opportunity to review.

# Preparing experiment 1/3: Default

# Preparing experiment 2/3: Keras Experiment
## Preparing run 1/4 [2e02df92025044669701ed6e6dd300ca]
## Preparing run 2/4 [93fb285da7cf4c4a93e279ab7ff19fc5]
## Preparing run 3/4 [2e8a1aed22544549b2b6b6b2c5976ed9]
## Preparing run 4/4 [82f584bad7604289af61bc505935599b]

# Preparing experiment 3/3: Tensorflow Keras Experiment
## Preparing run 1/2 [99550a7ce4c24677aeb6a1ae4e7444cb]
## Preparing run 2/2 [88ca5c4262f44176b576b54e0b24731a]

 MLFlow name:   | Comet.ml name:   |   Prepared count:
----------------+------------------+-------------------
 Experiments    | Projects         |                 3
 Runs           | Experiments      |                 6
 Tags           | Others           |                39
 Parameters     | Parameters       |                51
 Metrics        | Metrics          |                60
 Artifacts      | Assets           |                27

All prepared data has been saved to: /tmp/tmpjj74z8bf

Upload prepared data to Comet.ml? [y/N] y

# Start uploading data to Comet.ml
100%|███████████████████████████████████████████████████████████████████████| 6/6 [01:00<00:00, 15s/it]
Explore your experiment data on Comet.ml with the following links:
	- https://www.comet.ml/kstewart/mlflow-default-2bacc9?loginToken=NjKgD6f9ZuZWeudP76sDPHx9j
	- https://www.comet.ml/kstewart/mlflow-keras-experiment-2bacc9?loginToken=NjKgD6f9ZuZWeudP76sDPHx9j
	- https://www.comet.ml/kstewart/mlflow-tensorflow-keras-experiment-2bacc9?loginToken=NjKgD6f9ZuZWeudP76sDPHx9j
Get deeper instrumentation by adding Comet SDK to your project: https://comet.ml/docs/python-sdk/mlflow/


If you need support, you can contact us at http://chat.comet.ml/ or https://comet.ml/docs/quick-start/#getting-support
```


# Advanced use

## Importing MLFlow runs in a database store or in the MLFLow server store

If your MLFlow runs are not located in the default local store (`./mlruns`), you can either set the CLI flag `--mlflow-store-uri` or the environment variable `MLFLOW_TRACKING_URI` to point to the right store.

For example, with a different local store path:

```bash
comet_for_mlflow --mlflow-store-uri /data/mlruns/
```

With a SQL store:

```bash
comet_for_mlflow --mlflow-store-uri sqlite:///path/to/file.db
```

Or with a MLFlow server:

```bash
comet_for_mlflow --mlflow-store-uri http://localhost:5000
```

## Importing MLFlow artifacts stored remotely

If your MLFlow runs have artifacts stored remotely (in any of supported remote artifact stores https://www.mlflow.org/docs/latest/tracking.html#artifact-stores), you need to configure your environment the same way as when you ran those experiments. For example, with a local Minio server:

```bash
env MLFLOW_S3_ENDPOINT_URL=http://localhost:9001 \
    AWS_ACCESS_KEY_ID=minio \
    AWS_SECRET_ACCESS_KEY=minio123 \
    comet_for_mlflow
```

# FAQ

## How can I configure my API Key or Rest API Key?

You can either pass your Comet.ml API Key or Rest API Key as command-line flags or through the [usual configuration options](https://www.comet.ml/docs/python-sdk/advanced/#python-configuration).

## How are MLFlow experiments mapped to Comet.ml projects?

Each MLFlow experiment is mapped to a unique Comet.ml project ID. This way even if you rename the Comet.ml project or the MLFlow experiment, new runs will be imported in the correct Comet.ml project. The name for newly created Comet.ml is `mlflow-$MLFLOW_EXPERIMENT_NAME`. The original MLFlow experiment name is also saved as an Other field named `mlflow.experimentName`.

Below is a complete list of MLFlow experiment and run fields mapped to Comet.ml equivalent concepts:

* MLFlow Experiments are mapped as Comet.ml projects
* MLFlow Runs are mapped as Comet.ml experiments
* MLFlow Runs fields are imported according to following table:

| MLFlow Run Field 	| Comet.ml Experiment Field 	|
|------------------	|---------------------------	|
| File name        	| File name                 	|
| Tags             	| Others                    	|
| User             	| Git User + System User    	|
| Git parent       	| Git parent                	|
| Git origin       	| Git Origin                	|
| Params           	| Params                    	|
| Metrics          	| Metrics                   	|
| Artifacts        	| Assets                    	|

## Do I have to run this for future experiments?

No, the common pattern is to import [Comet's Python SDK with MLFlow support](https://comet.ml/docs/python-sdk/mlflow/) in your MLFlow projects, which will keep all future experiment runs synchronized.


# Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage) project template.
