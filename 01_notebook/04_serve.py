# Databricks notebook source
# MAGIC %md
# MAGIC # ModernBERT — (Optional) Deploy to Databricks Model Serving
# MAGIC
# MAGIC Creates (or updates) a **GPU Model Serving** endpoint that serves the fine-tuned ModernBERT
# MAGIC model registered to Unity Catalog by `01`/`02`, waits until it is ready, and sends a test
# MAGIC request for real-time topic classification.
# MAGIC
# MAGIC ## Optional — and it's Model Serving, not AI Runtime
# MAGIC This repo's core value is **AI Runtime** (serverless-GPU training, multi-GPU, batch inference).
# MAGIC Real-time serving is a nice add-on, so this step is **optional**. It uses **Databricks Model
# MAGIC Serving** — the general-purpose GPU serving product, which is **independent of AI Runtime**.
# MAGIC (AI Runtime has its own custom/LLM serving, but that is oriented to **LLMs** and does not cover
# MAGIC BERT-class encoder classification — verify current AIR serving support — so we serve ModernBERT
# MAGIC via standard Model Serving.)
# MAGIC
# MAGIC ## How Model Serving works (mental model)
# MAGIC Register model in UC → create a **serving endpoint** for it → Databricks **builds a container
# MAGIC and provisions a GPU** (several minutes, one-time) → the endpoint reaches **READY** → you send
# MAGIC **JSON requests** over HTTPS → it **scales to zero** when idle (the first call after idle pays a
# MAGIC cold start).
# MAGIC
# MAGIC ## Note on execution mode
# MAGIC Unlike `01`–`03`, this is a **control-plane** step (it calls the Serving REST API), not a
# MAGIC GPU training job — so it needs **no GPU attach** (any compute works) and does **not** use the
# MAGIC AI Runtime CLI. Run it either way:
# MAGIC 1. **As a Databricks notebook** — run the cells top to bottom (the `%pip` cell installs
# MAGIC    `mlflow`; auth is the notebook's own), **or**
# MAGIC 2. **Locally / in CI** — install the dependency first, then run with your profile:
# MAGIC    ```bash
# MAGIC    pip install -r requirements.txt        # or: pip install "mlflow>=2.15.0"
# MAGIC    DATABRICKS_CONFIG_PROFILE=<your-profile> python 02_cli/04_serve.py
# MAGIC    ```
# MAGIC
# MAGIC It uses the MLflow **Deployments** client (`get_deploy_client("databricks")`), whose dict
# MAGIC config is the stable REST shape — so it does not break across `databricks-sdk` versions.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install dependencies
# MAGIC The `%pip` cell installs `mlflow`. **`%restart_python`** (a Databricks magic) then restarts the
# MAGIC notebook's Python process so that freshly installed version is the one imported below.

# COMMAND ----------

# MAGIC %pip install "mlflow>=2.15.0"

# COMMAND ----------

# MAGIC # Restarts the Python interpreter so the version just installed above is the one imported below.
# MAGIC # Databricks-specific magic; it clears in-memory state, so continue from the next cell.
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configuration
# MAGIC The serving knobs (one line each — these are the ones people trip on):
# MAGIC - **`endpoint_name`** — the REST endpoint's name (its URL path).
# MAGIC - **`model_version` / `MODEL_VERSION`** — which UC version to serve (empty = latest).
# MAGIC - **`workload_type`** — the **serving GPU tier** (`GPU_SMALL`, `GPU_MEDIUM`, …). **These names
# MAGIC   differ from AI Runtime's** (`GPU_1xA10` / `GPU_8xH100`): `GPU_SMALL` is a single small GPU
# MAGIC   (e.g. T4/A10-class), which fits ModernBERT with SDPA comfortably.
# MAGIC - **`workload_size`** — `Small`/`Medium`/`Large` = provisioned **concurrency** (how many
# MAGIC   simultaneous requests), **not** the GPU.
# MAGIC - **`scale_to_zero`** — scale to **0 replicas when idle (no cost)**; the next request then pays a
# MAGIC   **cold start**. Trade cost for tail latency.

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
# MAGIC ## 3. Create (or update) the endpoint
# MAGIC The endpoint config has two parts that confuse people:
# MAGIC - **`served_entities`** — *which model/version* to serve (plus its GPU tier + concurrency).
# MAGIC - **`traffic_config.routes`** — how to split traffic: send `traffic_percentage` to a
# MAGIC   `served_model_name` (here `"<model>-<version>"`). With one model it's just 100% to it, but
# MAGIC   this is how you'd do canary / A-B across versions.
# MAGIC
# MAGIC **Create vs update is idempotent:** if the endpoint already exists we *update* it instead of
# MAGIC failing, so re-running this notebook is safe. `_wait_ready` polls until `state.ready == READY`
# MAGIC and `config_update == NOT_UPDATING`; the **first** deploy takes several minutes because
# MAGIC Databricks builds the serving container and provisions the GPU.

# COMMAND ----------

import mlflow
from mlflow.deployments import get_deploy_client


def latest_version(cfg: Config) -> str:
    mlflow.set_registry_uri("databricks-uc")
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
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
    # Membership check instead of a broad try/except (which could mask auth/network errors).
    return any(e.get("name") == name for e in (client.list_endpoints() or []))


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


# Run it: create/update the endpoint and wait until it is ready (first deploy takes several minutes).
client, version = deploy(CFG)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Send a test request
# MAGIC The **request** body for a `transformers` text-classification pipeline is
# MAGIC `{"inputs": ["text1", "text2", ...]}`; the **response** is `{"predictions": [{"label", "score"},
# MAGIC ...]}` — one label + score per input row. The **first** call after idle is a **cold start**
# MAGIC (slower while a replica spins up); subsequent calls are fast.

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


# Run it: send a few real-time requests to the endpoint.
query(client, CFG)
print(f"Done. Endpoint '{CFG.endpoint_name}' serving {CFG.uc_model_fqn} v{version}.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Call it from outside the notebook (plain REST / curl)
# MAGIC The endpoint is a standard HTTPS REST service — call it from anywhere with a bearer token:
# MAGIC ```bash
# MAGIC curl -s -X POST \
# MAGIC   -H "Authorization: Bearer $DATABRICKS_TOKEN" \
# MAGIC   -H "Content-Type: application/json" \
# MAGIC   -d '{"inputs": ["The central bank raised interest rates."]}' \
# MAGIC   https://<workspace-host>/serving-endpoints/modernbert-agnews/invocations
# MAGIC ```
# MAGIC Replace `<workspace-host>` with your workspace URL and `modernbert-agnews` with `ENDPOINT_NAME`.
# MAGIC The response is the same `{"predictions": [{"label", "score"}, ...]}` shape as above.
