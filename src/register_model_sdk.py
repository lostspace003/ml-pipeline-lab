"""Register outputs/model in Azure Machine Learning using the Python SDK v2.

Equivalent to:
    az ml model create --name diabetes-regressor --type mlflow_model --path outputs/model

Requires:  pip install azure-ai-ml azure-identity
Env vars:  AZURE_SUBSCRIPTION_ID, AML_RESOURCE_GROUP, AML_WORKSPACE
Sign in first with `az login` (DefaultAzureCredential picks up the CLI session).
"""
import os

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential

client = MLClient(
    DefaultAzureCredential(),
    os.environ["AZURE_SUBSCRIPTION_ID"],
    os.environ["AML_RESOURCE_GROUP"],
    os.environ["AML_WORKSPACE"],
)

model = Model(
    path="outputs/model",
    type=AssetTypes.MLFLOW_MODEL,
    name="diabetes-regressor",
    description="Gradient boosting regressor, scikit-learn diabetes dataset (registered with SDK v2)",
)

registered = client.models.create_or_update(model)
print(f"Registered {registered.name} version {registered.version}")
