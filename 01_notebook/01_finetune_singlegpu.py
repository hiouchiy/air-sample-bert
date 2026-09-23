# Databricks notebook source
# MAGIC %md
# MAGIC # ModernBERT — Single-GPU fine-tuning on AG News (Databricks AI Runtime)
# MAGIC
# MAGIC This example fine-tunes [`answerdotai/ModernBERT-base`](https://huggingface.co/answerdotai/ModernBERT-base)
# MAGIC for **topic classification** on the public **AG News** dataset (4 classes:
# MAGIC `World`, `Sports`, `Business`, `Sci/Tech`), using a **single GPU** on Databricks
# MAGIC AI Runtime. It tracks the run with **MLflow** and registers the fine-tuned model
# MAGIC to **Unity Catalog**.
# MAGIC
# MAGIC ## Why ModernBERT?
# MAGIC ModernBERT (Dec 2024) is the current mainstream BERT-class encoder: RoPE positional
# MAGIC embeddings, alternating local/global attention, Flash Attention 2 support, an 8192-token
# MAGIC context window, and training on a large English + code corpus. It is a drop-in, more
# MAGIC efficient replacement for classic BERT / RoBERTa for classification, retrieval and reranking.

# COMMAND ----------

# MAGIC %md
# MAGIC ## ▶ Before you Run All — attach a serverless GPU
# MAGIC AI Runtime GPUs are **serverless** — there is no cluster to create. This notebook needs a
# MAGIC **single-GPU `GPU_1xA10`**. Attach one from the notebook itself:
# MAGIC 1. Open the **compute** drop-down at the top of the notebook → **Serverless GPU**.
# MAGIC 2. Click the **environment** icon to open the **Environment** side panel.
# MAGIC 3. Set **Accelerator** to a **single A10** (`GPU_1xA10`); leave the default **Base environment**.
# MAGIC 4. Click **Apply**, then **Confirm**.
# MAGIC
# MAGIC Then **Run All** — the steps below execute top to bottom and show their output as you go.
# MAGIC Docs: [Connect to serverless GPU compute](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/connecting#gpu-compute).
# MAGIC
# MAGIC > Prefer submitting from a terminal? The CLI equivalent is `02_cli/01_finetune_singlegpu.py`
# MAGIC > — run it with `air run --file 02_cli/finetune_singlegpu.yaml`.

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
# MAGIC Every knob is an environment variable with a sensible default. In the notebook you can set
# MAGIC one before running, e.g. `import os; os.environ["EPOCHS"] = "3"`. (The CLI form sets them in
# MAGIC the workload YAML — see `02_cli/finetune_singlegpu.yaml`.)

# COMMAND ----------

import os
from dataclasses import dataclass


def _env(name: str, default: str) -> str:
    """Read an env var, falling back to a default. Keeps notebook and CLI in sync."""
    return os.environ.get(name, default)


@dataclass
class Config:
    # Model & data ---------------------------------------------------------
    model_name: str = _env("MODEL_NAME", "answerdotai/ModernBERT-base")
    dataset_name: str = _env("DATASET_NAME", "fancyzhx/ag_news")
    max_length: int = int(_env("MAX_LENGTH", "256"))
    # Sub-sample to keep the single-GPU demo fast. Set to -1 to use the full split.
    max_train_samples: int = int(_env("MAX_TRAIN_SAMPLES", "20000"))
    max_eval_samples: int = int(_env("MAX_EVAL_SAMPLES", "2000"))

    # Attention backend: "sdpa" (default) or "flash_attention_2" (see the Train section).
    attn_implementation: str = _env("ATTN_IMPLEMENTATION", "sdpa")

    # Training hyper-parameters -------------------------------------------
    epochs: float = float(_env("EPOCHS", "1"))
    train_batch_size: int = int(_env("TRAIN_BATCH_SIZE", "32"))
    eval_batch_size: int = int(_env("EVAL_BATCH_SIZE", "64"))
    learning_rate: float = float(_env("LEARNING_RATE", "5e-5"))
    weight_decay: float = float(_env("WEIGHT_DECAY", "0.01"))
    warmup_ratio: float = float(_env("WARMUP_RATIO", "0.1"))
    seed: int = int(_env("SEED", "42"))

    # Unity Catalog (model registry) --------------------------------------
    uc_catalog: str = _env("UC_CATALOG", "main")
    uc_schema: str = _env("UC_SCHEMA", "air_samples")
    registered_model_name: str = _env("REGISTERED_MODEL_NAME", "modernbert_agnews")
    register_model: bool = _env("REGISTER_MODEL", "true").lower() == "true"

    # Output --------------------------------------------------------------
    output_dir: str = _env("OUTPUT_DIR", "/tmp/modernbert_agnews")

    @property
    def uc_model_fqn(self) -> str:
        return f"{self.uc_catalog}.{self.uc_schema}.{self.registered_model_name}"


CFG = Config()
print(CFG)

# AG News label names (index order matches the dataset's integer labels).
LABELS = ["World", "Sports", "Business", "Sci/Tech"]
ID2LABEL = {i: l for i, l in enumerate(LABELS)}
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load & tokenize the dataset
# MAGIC AG News is a public dataset that ships with the `datasets` library.

# COMMAND ----------

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def load_and_tokenize(cfg: Config):
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
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=cfg.max_length,
        )

    tokenized = ds.map(tokenize, batched=True, remove_columns=["text"])
    return tokenized, tokenizer


# Run it: download AG News, sub-sample, and tokenize.
tokenized, tokenizer = load_and_tokenize(CFG)
print(tokenized)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Metrics
# MAGIC Compute accuracy and **macro-F1** (averages the four classes equally, so a weak class can't
# MAGIC hide behind the majority ones).

# COMMAND ----------

from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Train
# MAGIC ModernBERT supports two attention backends:
# MAGIC - **PyTorch SDPA** (`sdpa`) — built into PyTorch, no extra install, runs anywhere. **Default in this sample.**
# MAGIC - **Flash Attention 2** (`flash_attention_2`) — faster / more memory-efficient on A10/H100, but it
# MAGIC   ships as a CUDA extension that must be **compiled to match your exact PyTorch + CUDA + GPU
# MAGIC   architecture**; a matching prebuilt wheel often isn't available, so install can take many
# MAGIC   minutes or fail.
# MAGIC
# MAGIC To use FA2: `%pip install flash-attn --no-build-isolation` and set
# MAGIC `ATTN_IMPLEMENTATION=flash_attention_2` (or edit the `Config` cell).

# COMMAND ----------

import torch
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


def build_model(cfg: Config):
    print(f"Using attn_implementation={cfg.attn_implementation}")
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        attn_implementation=cfg.attn_implementation,
    )
    return model


def train(cfg: Config, tokenized, tokenizer):
    set_seed(cfg.seed)
    model = build_model(cfg)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        bf16=use_bf16,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        report_to=[],  # MLflow logging is handled explicitly in the next cell.
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
    trainer.train()
    metrics = trainer.evaluate()
    print("Eval metrics:", metrics)
    return trainer, metrics


# Run it: fine-tune on the GPU and evaluate.
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
trainer, metrics = train(CFG, tokenized, tokenizer)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Log to MLflow & register to Unity Catalog
# MAGIC We log the fine-tuned model with the MLflow `transformers` flavor (as a
# MAGIC `text-classification` pipeline) and register it directly to the **Unity Catalog Model
# MAGIC Registry**, so it can be loaded for batch inference and deployed to Model Serving with no
# MAGIC extra packaging code. Parameters and metrics are logged to the same MLflow run, and the new
# MAGIC version is promoted to the **`@champion`** alias.

# COMMAND ----------

import mlflow
from transformers import pipeline


def log_and_register(cfg: Config, trainer, tokenizer, metrics):
    mlflow.set_registry_uri("databricks-uc")

    clf = pipeline(
        "text-classification",
        model=trainer.model.to("cpu"),
        tokenizer=tokenizer,
    )
    example = ["Wall Street stocks rallied as tech earnings beat expectations."]

    # Log inside the run AI Runtime already created (nested=... makes this safe in a
    # notebook too, where a run may or may not be active).
    nested = mlflow.active_run() is not None
    with mlflow.start_run(run_name="modernbert-agnews-singlegpu", nested=nested) as run:
        mlflow.log_params(
            {
                "model_name": cfg.model_name,
                "dataset": cfg.dataset_name,
                "max_length": cfg.max_length,
                "epochs": cfg.epochs,
                "train_batch_size": cfg.train_batch_size,
                "learning_rate": cfg.learning_rate,
                "max_train_samples": cfg.max_train_samples,
                "training_mode": "single-gpu",
            }
        )
        mlflow.log_metrics(
            {k.replace("eval_", ""): float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        )
        model_info = mlflow.transformers.log_model(
            transformers_model=clf,
            artifact_path="model",
            task="text-classification",
            input_example=example,
            registered_model_name=cfg.uc_model_fqn if cfg.register_model else None,
        )
        print("Logged model:", model_info.model_uri)
        if cfg.register_model:
            _promote_to_champion(cfg, model_info)
        return run.info.run_id


def _promote_to_champion(cfg: Config, model_info):
    """Tag the just-registered version with the @champion alias — this is what batch inference
    (03) and serving (04) load by default, so version promotion is explicit and governed."""
    from mlflow.tracking import MlflowClient

    version = model_info.registered_model_version
    client = MlflowClient(registry_uri="databricks-uc")
    client.set_registered_model_alias(cfg.uc_model_fqn, "champion", version)
    print(f"Registered {cfg.uc_model_fqn} as version {version} and set alias @champion "
          f"(this is the version 03/04 will load).")


# Run it: log params/metrics/model and promote to @champion.
run_id = log_and_register(CFG, trainer, tokenizer, metrics)
print("MLflow run_id:", run_id)
