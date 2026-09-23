# Databricks notebook source
# MAGIC %md
# MAGIC # ModernBERT — Multi-GPU (8×H100) distributed fine-tuning on AG News
# MAGIC
# MAGIC This example fine-tunes [`answerdotai/ModernBERT-base`](https://huggingface.co/answerdotai/ModernBERT-base)
# MAGIC on **AG News** across **8× H100 GPUs on a single AI Runtime node** using **PyTorch DDP**.
# MAGIC It uses the `serverless_gpu` Python API's `@distributed` decorator, which launches one
# MAGIC process per GPU and wires up the `torch.distributed` environment. Hugging Face `Trainer`
# MAGIC then performs data-parallel training automatically.
# MAGIC
# MAGIC ## Why show multi-GPU for a BERT-class model?
# MAGIC ModernBERT-base (149M) and -large (395M) both fit comfortably on a single GPU, so the
# MAGIC realistic reason to use multiple GPUs here is **throughput**: DDP replicates the model on
# MAGIC every GPU and splits each batch across them, letting you train on the full 120k-row AG News
# MAGIC corpus in a fraction of the wall-clock time. (Model-parallel / FSDP / DeepSpeed are only
# MAGIC needed when a model does not fit on one GPU — not the case for BERT-class encoders.)
# MAGIC
# MAGIC ## ▶ Before you Run All — attach a serverless 8×H100 GPU
# MAGIC AI Runtime GPUs are **serverless** — there is no cluster to create. This notebook needs a
# MAGIC **`GPU_8xH100`** node (the `@distributed` decorator runs in local mode and requires the
# MAGIC attached GPU type to match `gpu_type="H100"`). Attach one from the notebook itself:
# MAGIC 1. Open the **compute** drop-down at the top of the notebook → **Serverless GPU**.
# MAGIC 2. Click the **environment** icon to open the **Environment** side panel.
# MAGIC 3. Set **Accelerator** to **8xH100** (`GPU_8xH100`); leave the default **Base environment**.
# MAGIC 4. Click **Apply**, then **Confirm**.
# MAGIC
# MAGIC Then **Run All** — the final cell calls `run_train.distributed()`, which `serverless_gpu` fans
# MAGIC out across the node's 8 GPUs (one process per GPU).
# MAGIC Docs: [Connect to serverless GPU compute](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/connecting#gpu-compute).
# MAGIC
# MAGIC > The CLI equivalent (`02_cli/02_finetune_multigpu.py`, launched with `torchrun`) trains the
# MAGIC > same model; see that folder to submit it as an AI Runtime CLI job.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install dependencies
# MAGIC These `%pip` cells install the dependencies when you Run All. (The CLI copy in `02_cli/`
# MAGIC gets them from its workload YAML instead.)

# COMMAND ----------

# MAGIC %pip install -U "transformers>=4.48,<5" "datasets>=2.19,<4" "accelerate>=0.30,<2" "hf_transfer"

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configuration
# MAGIC Same env-var-driven config as the single-GPU example. `NUM_GPUS` controls how many GPUs
# MAGIC the `@distributed` decorator requests (8 for `GPU_8xH100`).

# COMMAND ----------

import os
from dataclasses import dataclass


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


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

    # Attention backend: "sdpa" (default) or "flash_attention_2" (see §3).
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

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. The distributed training function
# MAGIC Everything the workers need lives **inside** `_train_impl()` — imports, data loading, model
# MAGIC init and the training loop — because `serverless_gpu` ships this function to each GPU worker
# MAGIC process. Hugging Face `Trainer` reads the `torch.distributed` env vars (`RANK`, `WORLD_SIZE`,
# MAGIC `LOCAL_RANK`, ...) that `@distributed` sets and runs DDP; we only log/register from rank 0.
# MAGIC
# MAGIC The attention backend defaults to **PyTorch SDPA** (built in, runs anywhere). To use
# MAGIC **Flash Attention 2** (faster on H100, but it compiles a CUDA extension to match your exact
# MAGIC PyTorch/CUDA/GPU and can be slow to install), add `flash-attn` to the deps and set
# MAGIC `ATTN_IMPLEMENTATION=flash_attention_2`.

# COMMAND ----------

from serverless_gpu import distributed


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
        report_to=[],
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

    t0 = time.time()
    trainer.train()
    train_secs = time.time() - t0
    metrics = trainer.evaluate()
    log(f"train_seconds={train_secs:.1f} eval_metrics={metrics}")

    # --- Log & register from rank 0 only ------------------------------
    if is_main:
        import mlflow
        from transformers import pipeline

        mlflow.set_registry_uri("databricks-uc")
        clf = pipeline("text-classification", model=trainer.model.to("cpu"), tokenizer=tokenizer)
        example = ["Wall Street stocks rallied as tech earnings beat expectations."]

        nested = mlflow.active_run() is not None
        with mlflow.start_run(run_name="modernbert-agnews-multigpu", nested=nested):
            mlflow.log_params(
                {
                    "model_name": cfg.model_name,
                    "dataset": cfg.dataset_name,
                    "max_length": cfg.max_length,
                    "epochs": cfg.epochs,
                    "per_device_train_batch_size": cfg.train_batch_size,
                    "effective_batch_size": cfg.train_batch_size * world_size,
                    "learning_rate": cfg.learning_rate,
                    "world_size": world_size,
                    "training_mode": f"ddp-{world_size}x{cfg.gpu_type}",
                }
            )
            mlflow.log_metric("train_seconds", train_secs)
            mlflow.log_metrics(
                {k.replace("eval_", ""): float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
            )
            info = mlflow.transformers.log_model(
                transformers_model=clf,
                artifact_path="model",
                task="text-classification",
                input_example=example,
                registered_model_name=cfg.uc_model_fqn if cfg.register_model else None,
            )
            log(f"logged model: {info.model_uri}")
            if cfg.register_model:
                from mlflow.tracking import MlflowClient

                v = info.registered_model_version
                MlflowClient(registry_uri="databricks-uc").set_registered_model_alias(
                    cfg.uc_model_fqn, "champion", v)
                log(f"registered {cfg.uc_model_fqn} version {v} and set alias @champion")

    return metrics


# The notebook launch handle: `serverless_gpu` ships `_train_impl` to `num_gpus` workers.
run_train = distributed(gpus=CFG.num_gpus, gpu_type=CFG.gpu_type)(_train_impl)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Launch the distributed training
# MAGIC `run_train.distributed()` fans `_train_impl` out across all `num_gpus` GPUs of the attached
# MAGIC node (one process per GPU) and runs PyTorch DDP. Each call also creates an MLflow run.

# COMMAND ----------

result = run_train.distributed()
print("Distributed training finished.")
