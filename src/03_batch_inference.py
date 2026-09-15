# Databricks notebook source
# MAGIC %md
# MAGIC # ModernBERT — GPU batch inference on AI Runtime
# MAGIC
# MAGIC Loads the fine-tuned ModernBERT model that `01`/`02` registered to **Unity Catalog** and
# MAGIC runs **GPU batch inference** over the AG News test set on an AI Runtime GPU. It reports
# MAGIC accuracy and throughput, and writes the scored rows to a **Unity Catalog Delta table**
# MAGIC (falling back to a CSV on a UC Volume if no Spark session is available).
# MAGIC
# MAGIC ## Runs two ways, without code changes
# MAGIC 1. **Notebook** — open and *Run All* on AI Runtime.
# MAGIC 2. **AI Runtime CLI** — `air run --file air/batch_inference.yaml --watch`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install dependencies (notebook only)

# COMMAND ----------

# MAGIC %pip install -U "transformers>=4.48,<5" "datasets>=2.19,<4" "hf_transfer"

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configuration
# MAGIC By default we load the latest version of the registered UC model and score the AG News
# MAGIC test split. Point `MODEL_URI` at a specific version/alias, or `INPUT_TABLE` at your own
# MAGIC Unity Catalog table (must have a `text` column) to score real data.

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
    # Empty -> use models:/<catalog>.<schema>.<name>@champion or latest version.
    model_uri: str = _env("MODEL_URI", "")

    dataset_name: str = _env("DATASET_NAME", "fancyzhx/ag_news")
    input_table: str = _env("INPUT_TABLE", "")          # optional UC table with a `text` column
    max_samples: int = int(_env("MAX_SAMPLES", "7600"))  # AG News test size
    batch_size: int = int(_env("BATCH_SIZE", "128"))
    max_length: int = int(_env("MAX_LENGTH", "256"))

    output_table: str = _env("OUTPUT_TABLE", "modernbert_agnews_predictions")

    @property
    def uc_model_fqn(self) -> str:
        return f"{self.uc_catalog}.{self.uc_schema}.{self.registered_model_name}"

    @property
    def resolved_model_uri(self) -> str:
        return self.model_uri or f"models:/{self.uc_model_fqn}@champion"

    @property
    def output_table_fqn(self) -> str:
        return f"{self.uc_catalog}.{self.uc_schema}.{self.output_table}"


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

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Load input rows

# COMMAND ----------

def load_inputs(cfg: Config):
    """Return (texts, labels_or_None). Labels are present only for the AG News eval path."""
    if cfg.input_table:
        spark = get_spark()
        if spark is None:
            raise RuntimeError("INPUT_TABLE set but no Spark session is available.")
        pdf = spark.table(cfg.input_table).limit(cfg.max_samples).toPandas()
        return pdf["text"].tolist(), None

    from datasets import load_dataset

    ds = load_dataset(cfg.dataset_name)["test"]
    if cfg.max_samples and cfg.max_samples > 0:
        ds = ds.select(range(min(cfg.max_samples, ds.num_rows)))
    return ds["text"], ds["label"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Run GPU batch inference

# COMMAND ----------

import time


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

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Persist predictions to Unity Catalog (with graceful fallback)

# COMMAND ----------

def get_spark():
    """Best-effort Spark session. AI Runtime notebooks expose `spark`; scripts may not."""
    try:
        from pyspark.sql import SparkSession

        s = SparkSession.getActiveSession()
        if s is not None:
            return s
        return SparkSession.builder.getOrCreate()
    except Exception:
        try:
            from databricks.connect import DatabricksSession

            return DatabricksSession.builder.getOrCreate()
        except Exception:
            return None


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

    spark = get_spark()
    if spark is not None:
        try:
            spark.createDataFrame(pdf).write.mode("overwrite").saveAsTable(cfg.output_table_fqn)
            print(f"Wrote {len(pdf)} predictions to UC table {cfg.output_table_fqn}")
            return cfg.output_table_fqn
        except Exception as exc:
            print(f"Spark write failed ({exc}); falling back to a UC Volume CSV.")

    # Fallback when no Spark session is available (AI Runtime GPU nodes have no Spark):
    # write the scored rows to a UC Volume instead of a Delta table.
    out_dir = f"/Volumes/{cfg.uc_catalog}/{cfg.uc_schema}/predictions"
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/{cfg.output_table}.csv"
    pdf.to_csv(path, index=False)
    print(f"Wrote {len(pdf)} predictions to UC Volume: {path}")
    return path

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Entry point

# COMMAND ----------

def main():
    from sklearn.metrics import accuracy_score

    mlflow.set_registry_uri("databricks-uc")
    print(f"CUDA available: {torch.cuda.is_available()}")
    clf, uri = load_model(CFG)
    texts, labels = load_inputs(CFG)

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
        target = persist(CFG, texts, preds, labels)
        print(f"Output: {target}")


# COMMAND ----------

if __name__ == "__main__":
    main()
