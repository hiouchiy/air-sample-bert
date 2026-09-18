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
# MAGIC GPU training job — so it does **not** use the AI Runtime CLI. Run it either way:
# MAGIC 1. **As a Databricks notebook** — open and *Run All* (the `%pip` cell installs `mlflow`; auth
# MAGIC    is the notebook's own), **or**
# MAGIC 2. **Locally / in CI** — install the dependency first, then run with your profile:
# MAGIC    ```bash
# MAGIC    pip install -r requirements.txt        # or: pip install "mlflow>=2.15.0"
# MAGIC    DATABRICKS_CONFIG_PROFILE=<your-profile> python 02_cli/04_serve.py
# MAGIC    ```
# MAGIC
# MAGIC It uses the MLflow **Deployments** client (`get_deploy_client("databricks")`), whose dict
# MAGIC config is the stable REST shape — so it does not break across `databricks-sdk` versions.

# COMMAND ----------

# MAGIC %pip install -U "mlflow>=2.15.0"

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

import os
import time
from dataclasses import dataclass


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Config:
    uc_catalog: str = _env("UC_CATALOG", "main")
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
from mlflow.deployments import get_deploy_client


def latest_version(cfg: Config) -> str:
    mlflow.set_registry_uri("databricks-uc")
    from mlflow.tracking import MlflowClient

    client = MlflowClient(registry_uri="databricks-uc")
    versions = client.search_model_versions(f"name='{cfg.uc_model_fqn}'")
    if not versions:
        raise RuntimeError(f"No versions found for {cfg.uc_model_fqn}. Run 01/02 first.")
    return str(max(int(v.version) for v in versions))


def _served_config(cfg: Config, version: str) -> dict:
    return {
        "served_entities": [
            {
                "entity_name": cfg.uc_model_fqn,
                "entity_version": version,
                "workload_type": cfg.workload_type,
                "workload_size": cfg.workload_size,
                "scale_to_zero_enabled": cfg.scale_to_zero,
            }
        ],
        "traffic_config": {
            "routes": [
                {"served_model_name": f"{cfg.registered_model_name}-{version}", "traffic_percentage": 100}
            ]
        },
    }


def _endpoint_exists(client, name: str) -> bool:
    try:
        client.get_endpoint(name)
        return True
    except Exception:
        return False


def _wait_ready(client, name: str, timeout_s: int = 2400):
    """Poll until state.ready == READY and config_update == NOT_UPDATING."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        state = (client.get_endpoint(name) or {}).get("state", {})
        ready, updating = state.get("ready"), state.get("config_update")
        print(f"  endpoint state: ready={ready} config_update={updating}")
        if ready == "READY" and updating in ("NOT_UPDATING", None):
            return
        time.sleep(30)
    raise TimeoutError(f"Endpoint {name} not ready within {timeout_s}s")


def deploy(cfg: Config):
    client = get_deploy_client("databricks")
    version = cfg.model_version or latest_version(cfg)
    config = _served_config(cfg, version)
    print(f"Deploying {cfg.uc_model_fqn} v{version} to endpoint '{cfg.endpoint_name}' "
          f"(this provisions GPU serving and may take several minutes).")

    if _endpoint_exists(client, cfg.endpoint_name):
        print("Endpoint exists — updating served entity.")
        client.update_endpoint(endpoint=cfg.endpoint_name, config=config)
    else:
        print("Creating endpoint.")
        client.create_endpoint(name=cfg.endpoint_name, config=config)

    _wait_ready(client, cfg.endpoint_name)
    print("Endpoint is ready.")
    return client, version

# COMMAND ----------

# MAGIC %md
# MAGIC ## Send a test request

# COMMAND ----------

def query(client, cfg: Config):
    examples = [
        "The national team clinched the championship in overtime last night.",
        "The central bank raised interest rates to curb rising inflation.",
        "Researchers unveiled a new quantum chip that doubles qubit stability.",
    ]
    response = client.predict(endpoint=cfg.endpoint_name, inputs={"inputs": examples})
    predictions = response["predictions"] if isinstance(response, dict) else response.predictions
    for text, pred in zip(examples, predictions):
        print(f"{pred} <- {text}")
    return predictions

# COMMAND ----------

# MAGIC %md
# MAGIC ## Entry point

# COMMAND ----------

def main():
    client, version = deploy(CFG)
    query(client, CFG)
    print(f"Done. Endpoint '{CFG.endpoint_name}' serving {CFG.uc_model_fqn} v{version}.")


# COMMAND ----------

if __name__ == "__main__":
    main()
