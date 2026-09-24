# Databricks notebook source
# MAGIC %md
# MAGIC # ModernBERT — GPU batch inference on AI Runtime
# MAGIC
# MAGIC Loads the fine-tuned ModernBERT model that `01`/`02` registered to **Unity Catalog**
# MAGIC (the `@champion` version) and runs **GPU batch inference** over the AG News test set on an
# MAGIC AI Runtime GPU. It reports accuracy and throughput and writes the scored rows to a CSV on a
# MAGIC **Unity Catalog Volume** (a plain file write — no Spark).

# COMMAND ----------

# MAGIC %md
# MAGIC ## ▶ Before you start — attach a serverless GPU
# MAGIC AI Runtime GPUs are **serverless** — there is no cluster to create. This notebook needs a
# MAGIC **single-GPU `GPU_1xA10`**. Attach one from the notebook itself:
# MAGIC 1. Open the **compute** drop-down at the top of the notebook → **Serverless GPU**.
# MAGIC 2. Click the **environment** icon to open the **Environment** side panel.
# MAGIC 3. Set **Accelerator** to a **single A10** (`GPU_1xA10`); leave the default **Base environment**.
# MAGIC 4. Click **Apply**, then **Confirm**.
# MAGIC
# MAGIC Run `01` (or `02`) first — it registers the model and sets the `@champion` alias this step
# MAGIC loads — then **run the cells one at a time, top to bottom**, reviewing each step's output.
# MAGIC (Run All works too, but stepping through is recommended for a sample you're evaluating.)
# MAGIC Docs: [Connect to serverless GPU compute](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/connecting#gpu-compute).
# MAGIC
# MAGIC > Prefer submitting from a terminal? The CLI equivalent is `02_cli/03_batch_inference.py` — run
# MAGIC > it with `air run --file 02_cli/batch_inference.yaml --watch`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install dependencies
# MAGIC The `%pip` cell installs the dependencies. **`%restart_python`** (a Databricks magic) then
# MAGIC restarts the notebook's Python process so those freshly installed versions are the ones
# MAGIC imported below — run both once, at the top. (The CLI copy in `02_cli/` gets its dependencies
# MAGIC from the workload YAML instead.)

# COMMAND ----------

# MAGIC %pip install -U "transformers>=4.48,<5" "datasets>=2.19,<4" "hf_transfer"

# COMMAND ----------

# MAGIC # Restarts the Python interpreter so the versions just installed above are the ones imported
# MAGIC # below. Databricks-specific magic; it clears in-memory state, so continue from the next cell.
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configuration
# MAGIC By default we load the registered UC model's **`@champion`** alias and score the AG News
# MAGIC test split. Point `MODEL_URI` at a specific version or alias to score with a different model.
# MAGIC
# MAGIC This step loads the **`@champion`** version of `<catalog>.<schema>.modernbert_agnews`. Both
# MAGIC `01` (single-GPU) and `02` (multi-GPU) set `@champion` to the model they just trained, so
# MAGIC **03 scores whichever you ran last**. To score a specific model: re-run `01` or `02` (it
# MAGIC re-promotes `@champion`), or set `MODEL_URI` to a version/alias, e.g.
# MAGIC `models:/<catalog>.<schema>.modernbert_agnews/3` (empty `MODEL_URI` = `@champion`).

# COMMAND ----------

import logging
import os
from dataclasses import dataclass

# Serverless/AI Runtime enforces a py4j method whitelist, so MLflow's optional run-context tag
# lookup logs a benign `Py4JSecurityException ... extraContext ... not whitelisted` warning during
# logging. It's harmless (MLflow skips a couple of optional tags and continues) — quiet just that
# logger so it doesn't look like a failure.
logging.getLogger("mlflow.tracking.context.registry").setLevel(logging.ERROR)


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Config:
    uc_catalog: str = _env("UC_CATALOG", "main")
    uc_schema: str = _env("UC_SCHEMA", "air_samples")
    registered_model_name: str = _env("REGISTERED_MODEL_NAME", "modernbert_agnews")
    # Empty -> use models:/<catalog>.<schema>.<name>@champion (else falls back to latest version).
    model_uri: str = _env("MODEL_URI", "")

    dataset_name: str = _env("DATASET_NAME", "fancyzhx/ag_news")
    max_samples: int = int(_env("MAX_SAMPLES", "7600"))  # AG News test size
    batch_size: int = int(_env("BATCH_SIZE", "128"))
    max_length: int = int(_env("MAX_LENGTH", "256"))

    output_name: str = _env("OUTPUT_NAME", "modernbert_agnews_predictions")

    @property
    def uc_model_fqn(self) -> str:
        return f"{self.uc_catalog}.{self.uc_schema}.{self.registered_model_name}"

    @property
    def resolved_model_uri(self) -> str:
        return self.model_uri or f"models:/{self.uc_model_fqn}@champion"


CFG = Config()
print(CFG)

LABELS = ["World", "Sports", "Business", "Sci/Tech"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load the model from Unity Catalog onto the GPU
# MAGIC We first try the `@champion` alias; if it is not set we fall back to the latest version.

# COMMAND ----------

import mlflow
import torch


def load_model(cfg: Config):
    mlflow.set_registry_uri("databricks-uc")
    device = 0 if torch.cuda.is_available() else -1
    uri = cfg.resolved_model_uri
    try:
        clf = mlflow.transformers.load_model(uri, device=device)
    except Exception as exc:
        print(f"Could not load {uri} ({exc}); falling back to latest version.")
        from mlflow.tracking import MlflowClient

        client = MlflowClient(registry_uri="databricks-uc")
        versions = client.search_model_versions(f"name='{cfg.uc_model_fqn}'")
        latest = max(int(v.version) for v in versions)
        uri = f"models:/{cfg.uc_model_fqn}/{latest}"
        print(f"Loading {uri}")
        clf = mlflow.transformers.load_model(uri, device=device)
    return clf, uri


# Run it: load the @champion model onto the GPU.
print(f"CUDA available: {torch.cuda.is_available()}")
clf, uri = load_model(CFG)
print("Loaded model from", uri)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Load input rows
# MAGIC Load the AG News test split (texts + true labels) to score.

# COMMAND ----------

def load_inputs(cfg: Config):
    """Return (texts, labels). Scores the AG News test split."""
    from datasets import load_dataset

    ds = load_dataset(cfg.dataset_name)["test"]
    if cfg.max_samples and cfg.max_samples > 0:
        ds = ds.select(range(min(cfg.max_samples, ds.num_rows)))
    return ds["text"], ds["label"]


# Run it.
texts, labels = load_inputs(CFG)
print(f"Loaded {len(texts)} rows to score.")

# Peek at a couple of input rows (text + true class) before scoring.
_preview = [{"text": t, "true_label": LABELS[y]} for t, y in list(zip(texts, labels))[:3]]
for row in _preview:
    print(row)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Run GPU batch inference
# MAGIC Score the texts on the GPU in batches, and log latency/throughput (and accuracy vs. the true
# MAGIC labels) to an MLflow run.

# COMMAND ----------

import time
from sklearn.metrics import accuracy_score


def run_inference(cfg: Config, clf, texts):
    t0 = time.time()
    preds = clf(
        texts,
        batch_size=cfg.batch_size,
        truncation=True,
        max_length=cfg.max_length,
    )
    elapsed = time.time() - t0
    throughput = len(texts) / elapsed if elapsed > 0 else float("nan")
    print(f"Scored {len(texts)} rows in {elapsed:.1f}s ({throughput:.0f} rows/s)")
    return preds, elapsed, throughput


# Run it: score inside an MLflow run and log metrics.
nested = mlflow.active_run() is not None
with mlflow.start_run(run_name="modernbert-agnews-batch-inference", nested=nested):
    preds, elapsed, throughput = run_inference(CFG, clf, texts)
    mlflow.log_params({"model_uri": uri, "batch_size": CFG.batch_size, "n_rows": len(texts)})
    mlflow.log_metric("inference_seconds", elapsed)
    mlflow.log_metric("rows_per_second", throughput)
    if labels is not None:
        pred_ids = [LABELS.index(p["label"]) for p in preds]
        acc = accuracy_score(labels, pred_ids)
        mlflow.log_metric("accuracy", acc)
        print(f"Batch inference accuracy: {acc:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Write predictions to a Unity Catalog Volume
# MAGIC A plain file write to the UC Volume — no Spark involved.

# COMMAND ----------

def persist(cfg: Config, texts, preds, labels):
    import pandas as pd

    rows = {
        "text": texts,
        "predicted_label": [p["label"] for p in preds],
        "score": [float(p["score"]) for p in preds],
    }
    if labels is not None:
        rows["true_label"] = [LABELS[i] for i in labels]
    pdf = pd.DataFrame(rows)

    out_dir = f"/Volumes/{cfg.uc_catalog}/{cfg.uc_schema}/predictions"
    # A UC Volume can't be created by mkdir on the /Volumes FUSE mount (that raises a cryptic
    # Errno 95). If the Volume is missing, tell the user exactly how to create it.
    if not os.path.isdir(out_dir):
        raise FileNotFoundError(
            f"UC Volume {out_dir} not found. Create it once:\n"
            f"  databricks volumes create {cfg.uc_catalog} {cfg.uc_schema} predictions MANAGED\n"
            f"(or run setup.sh with CATALOG={cfg.uc_catalog}), or set UC_CATALOG/UC_SCHEMA to an "
            f"existing Volume."
        )
    path = f"{out_dir}/{cfg.output_name}.csv"
    pdf.to_csv(path, index=False)
    print(f"Wrote {len(pdf)} predictions to UC Volume: {path}")
    return path


# Run it.
target = persist(CFG, texts, preds, labels)
print("Output:", target)
