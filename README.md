# ML pipeline lab - starter

Train a scikit-learn model locally, register it in Azure Machine Learning, and deploy it to a managed online endpoint from GitHub Actions. Follow `lab-02-ml-pipeline-github-azureml.md`.

```
.github/workflows/ml-pipeline.yml   train -> register -> deploy on push to main
aml/endpoint.yml                    managed online endpoint
aml/deployment.yml                  blue deployment (model + VM size)
aml/sample-request.json             smoke-test request (MLflow input_data format)
src/train.py                        trains and saves an MLflow model to outputs/model
src/register_model_sdk.py           SDK v2 alternative to `az ml model create`
requirements.txt                    pinned Python dependencies
```

Quick start:

```bash
python -m venv .venv && source .venv/bin/activate   # Git Bash on Windows: source .venv/Scripts/activate
pip install -r requirements.txt
python src/train.py --output outputs/model
az ml model create --name diabetes-regressor --type mlflow_model --path outputs/model -g <rg> -w <workspace>
```

GitHub configuration: secret `AZURE_CREDENTIALS` (output of `az ad sp create-for-rbac ... --json-auth`) and variables `AML_RESOURCE_GROUP`, `AML_WORKSPACE`, `AML_ENDPOINT_NAME`.
