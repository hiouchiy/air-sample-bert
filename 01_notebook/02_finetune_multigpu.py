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
# MAGIC ## ▶ Before you start — attach a serverless 8×H100 GPU
# MAGIC AI Runtime GPUs are **serverless** — there is no cluster to create. This notebook needs a
# MAGIC **`GPU_8xH100`** node (the `@distributed` decorator runs in local mode and requires the
# MAGIC attached GPU type to match `gpu_type="H100"`). Attach one from the notebook itself:
# MAGIC 1. Open the **compute** drop-down at the top of the notebook → **Serverless GPU**.
# MAGIC 2. Click the **environment** icon to open the **Environment** side panel.
# MAGIC 3. Set **Accelerator** to **8xH100** (`GPU_8xH100`); leave the default **Base environment**.
# MAGIC 4. Click **Apply**, then **Confirm**.
# MAGIC
# MAGIC Then **run the cells one at a time, top to bottom**, reviewing each step's output; the final
# MAGIC cell calls `run_train.distributed()`, which fans out across the node's 8 GPUs. (Run All works
# MAGIC too, but stepping through is recommended for a sample you're evaluating.)
# MAGIC Docs: [Connect to serverless GPU compute](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/connecting#gpu-compute).
# MAGIC
# MAGIC > The CLI equivalent (`02_cli/02_finetune_multigpu.py`, launched with `torchrun`) trains the
# MAGIC > same model; see that folder to submit it as an AI Runtime CLI job.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install dependencies
# MAGIC The `%pip` cell installs the dependencies. **`%restart_python`** (a Databricks magic) then
# MAGIC restarts the notebook's Python process so those freshly installed versions are the ones
# MAGIC imported below — run both once, at the top. (The CLI copy in `02_cli/` gets its dependencies
# MAGIC from the workload YAML instead.)

# COMMAND ----------

# MAGIC %pip install "transformers>=4.48" "datasets>=2.19" "accelerate>=0.30" "hf_transfer"

# COMMAND ----------

# MAGIC # Restarts the Python interpreter so the versions just installed above are the ones imported
# MAGIC # below. Databricks-specific magic; it clears in-memory state, so continue from the next cell.
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configuration
# MAGIC Same env-var-driven config as the single-GPU example. `NUM_GPUS` controls how many GPUs
# MAGIC the `@distributed` decorator requests (8 for `GPU_8xH100`).

# COMMAND ----------

import os
from dataclasses import dataclass


def _logmodel_model_kw():
    """Cross-version: MLflow >= 3 takes name=, MLflow 2.x (AIR CLI env) requires artifact_path=."""
    import mlflow
    return {"name": "model"} if int(mlflow.__version__.split(".")[0]) >= 3 else {"artifact_path": "model"}


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
# MAGIC
# MAGIC ### What `serverless_gpu` / `@distributed` is (the Databricks-unique part)
# MAGIC `serverless_gpu` is a **Databricks AI Runtime library** (preinstalled in the AIR environment,
# MAGIC currently Beta) — **not** a generic PyPI package. It runs a plain Python function across
# MAGIC multiple **serverless** GPUs with **no cluster to create, no `torchrun`, no shell launcher**.
# MAGIC See [Distributed training on serverless GPU](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/).
# MAGIC
# MAGIC ### What `distributed(gpus=N, gpu_type="H100")` does, step by step
# MAGIC 1. Claims **N GPUs of that type** and launches **one worker process per GPU**.
# MAGIC 2. Sets up the `torch.distributed` rendezvous env (`RANK`, `WORLD_SIZE`, `LOCAL_RANK`,
# MAGIC    `MASTER_ADDR/PORT`) on every worker.
# MAGIC 3. **Ships `_train_impl` (and its closure) to every worker** and runs them in parallel — so
# MAGIC    HF `Trainer` sees a DDP world and data-parallelizes automatically.
# MAGIC 4. Returns rank 0's result.
# MAGIC
# MAGIC ### Why `_train_impl` must be self-contained
# MAGIC It is serialized and shipped to separate worker processes, so its imports, data loading and
# MAGIC model init all live **inside** the function — top-level notebook state isn't available there.
# MAGIC
# MAGIC ### The two-step dispatch (easy to miss)
# MAGIC ```python
# MAGIC run_train = distributed(gpus=CFG.num_gpus, gpu_type=CFG.gpu_type)(_train_impl)  # wrap into a launcher
# MAGIC result = run_train.distributed()                                               # trigger the fan-out
# MAGIC ```
# MAGIC The first line *wraps* the function into a distributed launcher; the second line (step 4)
# MAGIC *actually runs* it across the GPUs. The `02_cli/` copy launches the same `_train_impl` with
# MAGIC `torchrun --nproc_per_node=gpu` instead — same DDP, a different launcher.
# MAGIC
# MAGIC ### Attention backend
# MAGIC Defaults to **PyTorch SDPA** (built in, runs anywhere). To use **Flash Attention 2** (faster on
# MAGIC H100, but it compiles a CUDA extension to match your exact PyTorch/CUDA/GPU and can be slow to
# MAGIC install), add `flash-attn` to the deps and set `ATTN_IMPLEMENTATION=flash_attention_2`.
# MAGIC
# MAGIC We set `report_to=["mlflow"]`, and rank 0 opens the MLflow run **before** `trainer.train()`,
# MAGIC so the training/eval curves stream into the same run that later gets the final metrics + model.

# COMMAND ----------

from serverless_gpu import distributed


def _train_impl():
    import logging
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

        # Quiet the benign serverless py4j-whitelist warning MLflow logs while resolving tags.
        logging.getLogger("mlflow.tracking.context.registry").setLevel(logging.ERROR)
        # Serverless also emits benign pyspark-connect / py4j chatter during MLflow logging.
        logging.getLogger("pyspark.sql.connect").setLevel(logging.ERROR)
        logging.getLogger("py4j").setLevel(logging.ERROR)
        mlflow.set_registry_uri("databricks-uc")
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


# Two-step dispatch (see §3): first WRAP _train_impl into a distributed launcher...
run_train = distributed(gpus=CFG.num_gpus, gpu_type=CFG.gpu_type)(_train_impl)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Launch the distributed training
# MAGIC `run_train.distributed()` is the second dispatch step: it fans `_train_impl` out across all
# MAGIC `num_gpus` GPUs of the attached node (one process per GPU) and runs PyTorch DDP. Rank 0 logs a
# MAGIC single MLflow run (training curves → final metrics → registered `@champion` model).

# COMMAND ----------

result = run_train.distributed()
print("Distributed training finished.")
