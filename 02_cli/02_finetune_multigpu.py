"""ModernBERT multi-GPU (8xH100) DDP fine-tuning on AG News — AI Runtime CLI script.

Launched by torchrun (see 02_cli/finetune_multigpu.yaml: `torchrun --standalone
--nproc_per_node=gpu ...`), which starts one process per GPU. Hugging Face Trainer reads the
torchrun env vars and runs PyTorch DDP.
    COPYFILE_DISABLE=1 air run --file 02_cli/finetune_multigpu.yaml --watch --profile <your-profile>
The notebook equivalent (01_notebook/02_finetune_multigpu.py) uses serverless_gpu's
`@distributed` / `run_train.distributed()` instead of torchrun.
"""

import logging
import os
from dataclasses import dataclass


def _logmodel_model_kw():
    """Cross-version: MLflow >= 3 takes name=, MLflow 2.x (AIR CLI env) requires artifact_path=."""
    import mlflow
    return {"name": "model"} if int(mlflow.__version__.split(".")[0]) >= 3 else {"artifact_path": "model"}

# Serverless/AI Runtime enforces a py4j method whitelist, so MLflow's optional run-context tag
# lookup logs a benign `Py4JSecurityException ... extraContext ... not whitelisted` warning. It's
# harmless (MLflow skips a couple of optional tags and continues) — quiet just that logger.
logging.getLogger("mlflow.tracking.context.registry").setLevel(logging.ERROR)
# Serverless also emits benign pyspark-connect / py4j chatter during MLflow logging; quiet it too.
logging.getLogger("pyspark.sql.connect").setLevel(logging.ERROR)
logging.getLogger("py4j").setLevel(logging.ERROR)


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _ensure_uc(catalog, schema, volume=None):
    """Create the UC schema (and optionally a MANAGED volume) if missing, so a fresh catalog runs
    top-to-bottom with no manual setup. Falls back to an actionable message without CREATE rights."""
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.catalog import VolumeType

    w = WorkspaceClient()

    def _create(fn, what):
        try:
            fn()
            print(f"Created {what}")
        except Exception as e:
            m = str(e).lower()
            if "already exists" in m:
                return
            if any(t in m for t in ("permission", "denied", "does not have", "unauthorized")):
                raise RuntimeError(
                    f"Cannot create {what}: {e}\nGrant CREATE on catalog '{catalog}', or pre-create "
                    f"it (see setup.sh / the README), or set UC_CATALOG/UC_SCHEMA to an existing one."
                ) from e
            raise

    _create(lambda: w.schemas.create(name=schema, catalog_name=catalog), f"schema {catalog}.{schema}")
    if volume:
        _create(lambda: w.volumes.create(catalog_name=catalog, schema_name=schema, name=volume,
                                         volume_type=VolumeType.MANAGED), f"volume {catalog}.{schema}.{volume}")


@dataclass
class Config:
    # Distributed --------------------------------------------------------
    num_gpus: int = int(_env("NUM_GPUS", "8"))
    gpu_type: str = _env("GPU_TYPE", "H100")

    # Model & data -------------------------------------------------------
    model_name: str = _env("MODEL_NAME", "answerdotai/ModernBERT-base")
    dataset_name: str = _env("DATASET_NAME", "fancyzhx/ag_news")
    max_length: int = int(_env("MAX_LENGTH", "256"))
    # Default to the FULL training set — the point of multi-GPU is throughput at scale.
    max_train_samples: int = int(_env("MAX_TRAIN_SAMPLES", "-1"))
    max_eval_samples: int = int(_env("MAX_EVAL_SAMPLES", "-1"))

    # Attention backend: "sdpa" (default) or "flash_attention_2" (needs a matching flash-attn build).
    attn_implementation: str = _env("ATTN_IMPLEMENTATION", "sdpa")

    # Training hyper-parameters -----------------------------------------
    epochs: float = float(_env("EPOCHS", "2"))
    train_batch_size: int = int(_env("TRAIN_BATCH_SIZE", "64"))  # per-device
    eval_batch_size: int = int(_env("EVAL_BATCH_SIZE", "128"))
    learning_rate: float = float(_env("LEARNING_RATE", "5e-5"))
    weight_decay: float = float(_env("WEIGHT_DECAY", "0.01"))
    warmup_ratio: float = float(_env("WARMUP_RATIO", "0.1"))
    seed: int = int(_env("SEED", "42"))

    # Unity Catalog (model registry) ------------------------------------
    uc_catalog: str = _env("UC_CATALOG", "main")
    uc_schema: str = _env("UC_SCHEMA", "air_samples")
    registered_model_name: str = _env("REGISTERED_MODEL_NAME", "modernbert_agnews")
    register_model: bool = _env("REGISTER_MODEL", "true").lower() == "true"

    output_dir: str = _env("OUTPUT_DIR", "/tmp/modernbert_agnews_ddp")

    @property
    def uc_model_fqn(self) -> str:
        return f"{self.uc_catalog}.{self.uc_schema}.{self.registered_model_name}"


CFG = Config()
print(CFG)

LABELS = ["World", "Sports", "Business", "Sci/Tech"]
ID2LABEL = {i: l for i, l in enumerate(LABELS)}
LABEL2ID = {l: i for i, l in enumerate(LABELS)}


def _train_impl():
    import time

    import numpy as np
    import torch
    from datasets import load_dataset
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    cfg = CFG
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_main = rank == 0

    def log(msg):
        print(f"[rank {rank}/{world_size}] {msg}")

    set_seed(cfg.seed)
    log(f"CUDA devices visible: {torch.cuda.device_count()}")

    # --- Data ---------------------------------------------------------
    ds = load_dataset(cfg.dataset_name)
    if cfg.max_train_samples and cfg.max_train_samples > 0:
        ds["train"] = ds["train"].shuffle(seed=cfg.seed).select(
            range(min(cfg.max_train_samples, ds["train"].num_rows))
        )
    if cfg.max_eval_samples and cfg.max_eval_samples > 0:
        ds["test"] = ds["test"].shuffle(seed=cfg.seed).select(
            range(min(cfg.max_eval_samples, ds["test"].num_rows))
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=cfg.max_length)

    tokenized = ds.map(tokenize, batched=True, remove_columns=["text"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # --- Model --------------------------------------------------------
    log(f"attn_implementation={cfg.attn_implementation}")
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        attn_implementation=cfg.attn_implementation,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro"),
        }

    # --- Trainer (handles DDP automatically) --------------------------
    args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        bf16=True,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        report_to=["mlflow"],  # HF's MLflow callback logs loss/eval curves (rank 0 only) -> active run
        ddp_find_unused_parameters=False,
        seed=cfg.seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Rank 0 opens the MLflow run BEFORE training so the Trainer's MLflow callback logs the
    # loss/eval curves into it (the callback only logs from rank 0, so this is DDP-safe).
    if is_main:
        import mlflow

        mlflow.set_registry_uri("databricks-uc")
        _ensure_uc(cfg.uc_catalog, cfg.uc_schema)
        mlflow.start_run(run_name="modernbert-agnews-multigpu")

    t0 = time.time()
    trainer.train()
    train_secs = time.time() - t0
    metrics = trainer.evaluate()
    log(f"train_seconds={train_secs:.1f} eval_metrics={metrics}")

    # --- Add final metrics + model to the SAME run, from rank 0 only --
    if is_main:
        from transformers import pipeline

        # Package on CPU so the logged model loads on any hardware, and so MLflow's signature
        # inference (it runs the pipeline once at log time) doesn't hit a GPU/CPU device mismatch.
        # Training ran on GPU; only this serialization step uses CPU.
        clf = pipeline("text-classification", model=trainer.model.to("cpu"), tokenizer=tokenizer)
        example = ["Wall Street stocks rallied as tech earnings beat expectations."]

        # HF's MLflow callback already logged the Trainer args + model config; add our extra config
        # under distinct keys, skipping any that would collide (avoids a duplicate-key
        # RestException, e.g. the model config's own `max_length`).
        for _k, _v in {
            "base_model": cfg.model_name,
            "dataset": cfg.dataset_name,
            "tokenizer_max_length": cfg.max_length,
            "effective_batch_size": cfg.train_batch_size * world_size,
            "ddp_world_size": world_size,
            "training_mode": f"ddp-{world_size}x{cfg.gpu_type}",
        }.items():
            try:
                mlflow.log_param(_k, _v)
            except Exception:
                pass
        mlflow.log_metric("train_seconds", train_secs)
        mlflow.log_metrics(
            {k.replace("eval_", ""): float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        )
        info = mlflow.transformers.log_model(
            transformers_model=clf,
            **_logmodel_model_kw(),
            task="text-classification",
            input_example=example,
            registered_model_name=cfg.uc_model_fqn if cfg.register_model else None,
        )
        log(f"logged model: {info.model_uri}")
        if cfg.register_model:
            from mlflow.tracking import MlflowClient

            v = info.registered_model_version
            MlflowClient().set_registered_model_alias(
                cfg.uc_model_fqn, "champion", v)
            log(f"registered {cfg.uc_model_fqn} version {v} and set alias @champion")
        mlflow.end_run()

    return metrics


def main():
    # This script is launched by torchrun (see 02_cli/finetune_multigpu.yaml), which starts one
    # process per GPU and sets RANK/WORLD_SIZE/LOCAL_RANK. Each process runs the training directly.
    return _train_impl()


if __name__ == "__main__":
    main()
