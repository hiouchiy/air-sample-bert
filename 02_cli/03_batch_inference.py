"""ModernBERT GPU batch inference on AG News — AI Runtime CLI script.

Loads the @champion model from Unity Catalog, scores the AG News test set on the GPU, and writes
predictions as a CSV to a UC Volume. Requires a model registered by 01/02 first.
    COPYFILE_DISABLE=1 air run --file 02_cli/batch_inference.yaml --watch --profile <your-profile>
The notebook equivalent is 01_notebook/03_batch_inference.py.
"""

import logging
import os
from dataclasses import dataclass

# Serverless/AI Runtime enforces a py4j method whitelist, so MLflow's optional run-context tag
# lookup logs a benign `Py4JSecurityException ... extraContext ... not whitelisted` warning. It's
# harmless (MLflow skips a couple of optional tags and continues) — quiet just that logger.
logging.getLogger("mlflow.tracking.context.registry").setLevel(logging.ERROR)
# Serverless also emits benign pyspark-connect / py4j chatter during MLflow logging; quiet it too.
logging.getLogger("pyspark.sql.connect").setLevel(logging.ERROR)
logging.getLogger("py4j").setLevel(logging.ERROR)


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Config:
    uc_catalog: str = _env("UC_CATALOG", "main")
    uc_schema: str = _env("UC_SCHEMA", "air_samples")
    registered_model_name: str = _env("REGISTERED_MODEL_NAME", "modernbert_agnews")
    # Empty -> use models:/<catalog>.<schema>.<name>@champion or latest version.
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


import mlflow
import torch


def load_model(cfg: Config):
    mlflow.set_registry_uri("databricks-uc")
    device = 0 if torch.cuda.is_available() else -1
    uri = cfg.resolved_model_uri
    try:
        clf = mlflow.transformers.load_model(uri, device=device)
    except mlflow.exceptions.MlflowException as exc:
        print(f"Could not load {uri} ({exc}); falling back to latest version.")
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        versions = client.search_model_versions(f"name='{cfg.uc_model_fqn}'")
        if not versions:
            raise RuntimeError(f"No versions for {cfg.uc_model_fqn}; run 01/02 first.")
        latest = max(int(v.version) for v in versions)
        uri = f"models:/{cfg.uc_model_fqn}/{latest}"
        print(f"Loading {uri}")
        clf = mlflow.transformers.load_model(uri, device=device)
    return clf, uri


def load_inputs(cfg: Config):
    """Return (texts, labels). Scores the AG News test split."""
    from datasets import load_dataset

    ds = load_dataset(cfg.dataset_name)["test"]
    if cfg.max_samples and cfg.max_samples > 0:
        ds = ds.select(range(min(cfg.max_samples, ds.num_rows)))
    return ds["text"], ds["label"]


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
            label_to_id = {l: i for i, l in enumerate(LABELS)}
            unknown = {p["label"] for p in preds if p["label"] not in label_to_id}
            if unknown:
                raise ValueError(f"Predicted labels not in LABELS {LABELS}: {unknown}. "
                                 "Check the model's id2label mapping.")
            pred_ids = [label_to_id[p["label"]] for p in preds]
            acc = accuracy_score(labels, pred_ids)
            mlflow.log_metric("accuracy", acc)
            print(f"Batch inference accuracy: {acc:.4f}")
        target = persist(CFG, texts, preds, labels)
        print(f"Output: {target}")


if __name__ == "__main__":
    main()
