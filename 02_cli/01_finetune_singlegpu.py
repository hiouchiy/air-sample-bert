"""ModernBERT single-GPU fine-tuning on AG News — AI Runtime CLI script.

Plain Python (no notebook markers). Submit it as an AI Runtime job:
    COPYFILE_DISABLE=1 air run --file 02_cli/finetune_singlegpu.yaml --watch --profile <your-profile>
Dependencies come from the YAML (environment.dependencies); config is via env vars.
The notebook-optimized equivalent is 01_notebook/01_finetune_singlegpu.py.
"""

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


from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


import torch
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


def _pick_attn_implementation() -> str:
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except Exception:
        return "sdpa"


def build_model(cfg: Config):
    attn = _pick_attn_implementation()
    print(f"Using attn_implementation={attn}")
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        attn_implementation=attn,
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
        report_to=[],  # MLflow logging is handled explicitly in main().
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


def main():
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    tokenized, tokenizer = load_and_tokenize(CFG)
    trainer, metrics = train(CFG, tokenized, tokenizer)
    run_id = log_and_register(CFG, trainer, tokenizer, metrics)
    print(f"Done. MLflow run_id={run_id}")
    return metrics


if __name__ == "__main__":
    main()
