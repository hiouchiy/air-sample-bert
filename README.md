# AI Runtime sample: ModernBERT on AG News

End-to-end **BERT-class fine-tuning, batch inference, and serving** on **Databricks AI Runtime**
(serverless NVIDIA GPUs). It fine-tunes **[ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base)**
— the current mainstream BERT-class encoder (RoPE, Flash Attention 2, 8k context, released Dec 2024)
— for **topic classification** on the public **AG News** dataset (4 classes: World, Sports, Business,
Sci/Tech), tracks everything with **MLflow**, registers the model to **Unity Catalog**, runs **GPU
batch inference**, and deploys a **Model Serving** endpoint.

> **Repo name is tentative** (`air-sample-bert`). AIR = AI Runtime.

## What it demonstrates

| Step | File | AIR compute | Shows |
|------|------|-------------|-------|
| 1. Single-GPU fine-tune | [`src/01_finetune_singlegpu.py`](src/01_finetune_singlegpu.py) | `GPU_1xA10` | Fine-tuning, MLflow tracking, UC registration |
| 2. Multi-GPU fine-tune | [`src/02_finetune_multigpu.py`](src/02_finetune_multigpu.py) | `GPU_8xH100` | 8-GPU **DDP** via `serverless_gpu`, throughput at scale |
| 3. GPU batch inference | [`src/03_batch_inference.py`](src/03_batch_inference.py) | `GPU_1xA10` | Loading the UC model, batched GPU scoring, writing to UC |
| 4. Model Serving | [`src/04_serve.py`](src/04_serve.py) | Serving (GPU) | Real-time endpoint from the registered model |

## Every script runs two ways, with no code changes

This is a hard requirement for these samples:

1. **As a Databricks notebook** — import the `.py` into the workspace (it carries
   `# Databricks notebook source` markers) and **Run All** on AI Runtime. The `# MAGIC %pip`
   cells install dependencies in the notebook.
2. **As an AI Runtime CLI job** — `air run --file air/<step>.yaml`. The `# MAGIC` lines are plain
   Python comments (ignored); dependencies come from the YAML `environment.dependencies`.

The same file works in both because logic lives in functions, the entry point is a single
`if __name__ == "__main__": main()` (Databricks notebooks also expose `__name__ == "__main__"`),
and all settings are environment variables with sensible defaults.

## Prerequisites

- **Databricks CLI** authenticated to the workspace (this repo was validated on
  `e2-demo-field-eng`, profile `DEFAULT`).
- **AI Runtime CLI**: `uv tool install --force databricks-air --python 3.12` (reuses your Databricks profile).
- A **Unity Catalog schema** for the registered model and outputs. Default:
  `main.air_samples` — override via env vars (see below).

## Quickstart (CLI)

```bash
# macOS: COPYFILE_DISABLE=1 keeps AppleDouble (._*) files out of the code snapshot.

# 1) Fine-tune on one A10, log to MLflow, register to Unity Catalog
COPYFILE_DISABLE=1 air run --file air/finetune_singlegpu.yaml --watch --profile DEFAULT

# 2) (Optional) Full-dataset fine-tune across 8x H100 with DDP
COPYFILE_DISABLE=1 air run --file air/finetune_multigpu.yaml --watch --profile DEFAULT

# 3) GPU batch inference over the AG News test set, written to a UC Delta table
COPYFILE_DISABLE=1 air run --file air/batch_inference.yaml --watch --profile DEFAULT

# 4) Deploy a real-time serving endpoint (control-plane; run locally or as a notebook)
DATABRICKS_CONFIG_PROFILE=DEFAULT python src/04_serve.py
```

## Quickstart (notebook)

Import any `src/*.py` into the workspace, attach it to AI Runtime, and **Run All**. Start with
`01_finetune_singlegpu.py`, then `03_batch_inference.py`, then `04_serve.py`.

## Configuration (env vars, with defaults)

| Variable | Default | Meaning |
|----------|---------|---------|
| `MODEL_NAME` | `answerdotai/ModernBERT-base` | Base model to fine-tune |
| `DATASET_NAME` | `fancyzhx/ag_news` | HuggingFace dataset |
| `UC_CATALOG` / `UC_SCHEMA` | `main` / `air_samples` | Unity Catalog target |
| `REGISTERED_MODEL_NAME` | `modernbert_agnews` | UC registered model name |
| `MAX_TRAIN_SAMPLES` / `MAX_EVAL_SAMPLES` | `20000` / `2000` (single-GPU); `-1` (multi-GPU) | Sub-sampling; `-1` = full split |
| `EPOCHS`, `TRAIN_BATCH_SIZE`, `LEARNING_RATE`, `MAX_LENGTH` | see scripts | Training hyper-parameters |

Override from the CLI by editing the YAML `command:` line, e.g.
`command: EPOCHS=3 MAX_TRAIN_SAMPLES=-1 python $CODE_SOURCE_PATH/src/01_finetune_singlegpu.py`.

## Repo layout

```
air-sample-bert/
├── src/                       # notebook-and-CLI dual-mode Python scripts
│   ├── 01_finetune_singlegpu.py
│   ├── 02_finetune_multigpu.py
│   ├── 03_batch_inference.py
│   └── 04_serve.py
├── air/                       # AI Runtime CLI workload specs (one per step)
│   ├── finetune_singlegpu.yaml
│   ├── finetune_multigpu.yaml
│   └── batch_inference.yaml
├── docs/                      # deeper docs (architecture, AIR notes)
├── requirements.txt
└── README.md
```

See [`docs/`](docs/) for the architecture walkthrough and AI Runtime specifics.
