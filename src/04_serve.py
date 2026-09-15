# Databricks notebook source
# MAGIC %md
# MAGIC # ModernBERT — Deploy to Databricks Model Serving
# MAGIC
# MAGIC Creates (or updates) a **GPU Model Serving** endpoint that serves the fine-tuned ModernBERT
# MAGIC model registered to Unity Catalog by `01`/`02`, waits until it is ready, and sends a test
# MAGIC request for real-time topic classification.
# MAGIC
# MAGIC ## Note on execution mode
# MAGIC Unlike `01`–`03`, this is a **control-plane** step (it calls the Serving REST API), not a
# MAGIC GPU training job. Run it as:
# MAGIC 1. **A Databricks notebook** — open and *Run All* (uses the notebook's ambient auth), or
# MAGIC 2. **Locally / in CI** — `DATABRICKS_CONFIG_PROFILE=DEFAULT python src/04_serve.py`.
# MAGIC
# MAGIC It does **not** need the AI Runtime CLI, because it provisions a serving endpoint rather
# MAGIC than running on a GPU node.

# COMMAND ----------

# MAGIC %pip install -U "databricks-sdk>=0.30.0" "mlflow>=2.15.0"

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

import os
from dataclasses import dataclass


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Config:
    uc_catalog: str = _env("UC_CATALOG", "hiroshi")
    uc_schema: str = _env("UC_SCHEMA", "air_samples")
    registered_model_name: str = _env("REGISTERED_MODEL_NAME", "modernbert_agnews")
    endpoint_name: str = _env("ENDPOINT_NAME", "modernbert-agnews")
    model_version: str = _env("MODEL_VERSION", "")  # empty -> latest
    workload_type: str = _env("WORKLOAD_TYPE", "GPU_SMALL")
    workload_size: str = _env("WORKLOAD_SIZE", "Small")
    scale_to_zero: bool = _env("SCALE_TO_ZERO", "true").lower() == "true"

    @property
    def uc_model_fqn(self) -> str:
        return f"{self.uc_catalog}.{self.uc_schema}.{self.registered_model_name}"


CFG = Config()
print(CFG)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resolve the model version and create/update the endpoint

# COMMAND ----------

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
)


def latest_version(cfg: Config) -> str:
    mlflow.set_registry_uri("databricks-uc")
    from mlflow.tracking import MlflowClient

    client = MlflowClient(registry_uri="databricks-uc")
    versions = client.search_model_versions(f"name='{cfg.uc_model_fqn}'")
    if not versions:
        raise RuntimeError(f"No versions found for {cfg.uc_model_fqn}. Run 01/02 first.")
    return str(max(int(v.version) for v in versions))


def deploy(cfg: Config):
    w = WorkspaceClient()
    version = cfg.model_version or latest_version(cfg)
    print(f"Deploying {cfg.uc_model_fqn} v{version} to endpoint '{cfg.endpoint_name}'")

    served = ServedEntityInput(
        entity_name=cfg.uc_model_fqn,
        entity_version=version,
        workload_type=cfg.workload_type,
        workload_size=cfg.workload_size,
        scale_to_zero_enabled=cfg.scale_to_zero,
    )
    config = EndpointCoreConfigInput(served_entities=[served])

    existing = [e.name for e in w.serving_endpoints.list()]
    if cfg.endpoint_name in existing:
        print("Endpoint exists — updating served entity (this may take several minutes).")
        w.serving_endpoints.update_config_and_wait(
            name=cfg.endpoint_name, served_entities=[served]
        )
    else:
        print("Creating endpoint (this may take several minutes).")
        w.serving_endpoints.create_and_wait(name=cfg.endpoint_name, config=config)
    print("Endpoint is ready.")
    return w, version

# COMMAND ----------

# MAGIC %md
# MAGIC ## Send a test request

# COMMAND ----------

def query(w, cfg: Config):
    examples = [
        "The national team clinched the championship in overtime last night.",
        "The central bank raised interest rates to curb rising inflation.",
        "Researchers unveiled a new quantum chip that doubles qubit stability.",
    ]
    response = w.serving_endpoints.query(name=cfg.endpoint_name, inputs=examples)
    for text, pred in zip(examples, response.predictions):
        print(f"{pred} <- {text}")
    return response.predictions

# COMMAND ----------

# MAGIC %md
# MAGIC ## Entry point

# COMMAND ----------

def main():
    w, version = deploy(CFG)
    query(w, CFG)
    print(
        f"Done. Endpoint '{CFG.endpoint_name}' serving {CFG.uc_model_fqn} v{version}."
    )


# COMMAND ----------

if __name__ == "__main__":
    main()
