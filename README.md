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

- A **Databricks workspace where AI Runtime is enabled.** AI Runtime is currently available in
  **US regions on AWS/Azure** (EU/APJ/GCP were not yet GA as of this writing) — confirm with your
  Databricks contact if unsure.
- Permission to **create a Unity Catalog schema and volume** in some catalog (ask your admin which
  catalog you can write to, or use one you own).
- macOS/Linux/WSL with a terminal. (This repo was validated on `e2-demo-field-eng`.)

## Setup — one time, ~10 minutes (no Databricks experience needed)

Run these on your laptop. Replace `<workspace-url>` and pick a profile name (here `air`).

```bash
# a) Install the Databricks CLI
brew install databricks            # macOS; else: curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh
databricks --version               # need v0.230+

# b) Log in — this opens a browser and saves an auth "profile" named `air`
databricks auth login --host https://<workspace-url>.cloud.databricks.com --profile air
databricks current-user me --profile air     # should print your email

# c) Install the AI Runtime CLI (`air`); it reuses the Databricks profiles above
curl -LsSf https://astral.sh/uv/install.sh | sh      # installs `uv` if you don't have it
uv tool install --force databricks-air --python 3.12
air --version

# d) Create the Unity Catalog schema + volume this demo writes to (one time).
#    Pick a catalog you can write to (e.g. `main`, or your own). Everything else is created for you.
export CATALOG=main                                   # <-- change to YOUR catalog
databricks schemas create air_samples $CATALOG --profile air
databricks volumes create $CATALOG air_samples predictions MANAGED --profile air
```

> Prefer one command? Run `CATALOG=main PROFILE=air ./setup.sh` (see [`setup.sh`](setup.sh)).

## Point the demo at your catalog

The scripts default to catalog **`hiroshi`**, schema **`air_samples`** (the environment this was
built in). **Set them to the `$CATALOG` you created above** in either of two ways:

- **Easiest:** edit the two `UC_CATALOG` / `UC_SCHEMA` default lines near the top of each
  `src/*.py` (search for `UC_CATALOG`), or
- **Per run:** prefix the YAML `command:` line, e.g.
  `command: UC_CATALOG=main python $CODE_SOURCE_PATH/src/01_finetune_singlegpu.py`.

Use the **same profile name** you created (`air`) in every `air run --profile ...` below.

## Run it (CLI) — do the steps in order

Step 3 and 4 need the model that step 1 (or 2) registers, so **run 01 first.**

```bash
# macOS: the COPYFILE_DISABLE=1 prefix is REQUIRED — it keeps macOS ._* files out of the
# uploaded code snapshot (otherwise the job dies immediately). Harmless on Linux.

# 1) Fine-tune on one A10 GPU → MLflow → register to Unity Catalog  (~10-15 min incl. GPU wait)
COPYFILE_DISABLE=1 air run --file air/finetune_singlegpu.yaml --watch --profile air

# 2) (Optional) Fine-tune across 8× H100 with DDP  (larger GPU request; may wait for capacity)
COPYFILE_DISABLE=1 air run --file air/finetune_multigpu.yaml --watch --profile air

# 3) GPU batch inference over the AG News test set → UC table (or a CSV on the UC Volume)
COPYFILE_DISABLE=1 air run --file air/batch_inference.yaml --watch --profile air

# 4) Deploy a real-time GPU serving endpoint, then query it  (run locally; needs Model Serving)
DATABRICKS_CONFIG_PROFILE=air python src/04_serve.py
```

Each `air run` ends with `Job status: SUCCESS` on success. The first run of each waits a few
minutes for a GPU to be provisioned — that is normal. (Note: `air logs` sometimes prints
"No logs available" even for successful runs; trust `Job status` and the MLflow links.)

## Run it (notebook)

Import any `src/*.py` into your Databricks workspace (**Workspace → Import → File**), attach it to
**AI Runtime**, and **Run All**. The `%pip` cells install dependencies automatically. Start with
`01_finetune_singlegpu.py`, then `03_batch_inference.py`, then `04_serve.py`.

## Configuration (env vars, with defaults)

| Variable | Default | Meaning |
|----------|---------|---------|
| `UC_CATALOG` / `UC_SCHEMA` | `hiroshi` / `air_samples` | **Unity Catalog target — set to your catalog (see above)** |
| `REGISTERED_MODEL_NAME` | `modernbert_agnews` | UC registered model name |
| `MODEL_NAME` | `answerdotai/ModernBERT-base` | Base model to fine-tune |
| `DATASET_NAME` | `fancyzhx/ag_news` | HuggingFace dataset |
| `MAX_TRAIN_SAMPLES` / `MAX_EVAL_SAMPLES` | `20000` / `2000` (single-GPU); `-1` (multi-GPU) | Sub-sampling; `-1` = full split |
| `EPOCHS`, `TRAIN_BATCH_SIZE`, `LEARNING_RATE`, `MAX_LENGTH` | see scripts | Training hyper-parameters |

Override any of these per run by prefixing the YAML `command:` line, e.g.
`command: EPOCHS=3 MAX_TRAIN_SAMPLES=-1 python $CODE_SOURCE_PATH/src/01_finetune_singlegpu.py`.

## Troubleshooting

| Symptom | Cause & fix |
|---------|-------------|
| Job dies in seconds, `cd: .../._xxx: Not a directory` | macOS AppleDouble files in the snapshot — always run with `COPYFILE_DISABLE=1` (see above). |
| `RESOURCE_DOES_NOT_EXIST` / schema or volume not found | Run the Setup step (d); make sure `UC_CATALOG`/`UC_SCHEMA` match what you created. |
| Job fails ~50s in with no logs | Usually a dependency-install issue — keep the pinned versions in the YAML; don't add heavy/unpinned packages. |
| `air logs` says "No logs available" | Known quirk; the run may still have succeeded. Check `Job status` and the MLflow run link. |
| Step 3/4 can't find the model | Run step 1 first (it registers the model), then set the `@champion` alias or let step 3 fall back to the latest version. |
| Long "waiting for GPU capacity" | Normal for H100; retry later or use the A10 steps. AI Runtime is US-region only for now. |

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
├── setup.sh                   # one-time UC schema + volume creation
├── requirements.txt
└── README.md
```

See [`docs/`](docs/) for the architecture walkthrough and AI Runtime specifics.
