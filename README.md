# AI Runtime sample: ModernBERT on AG News

End-to-end **BERT-class fine-tuning and batch inference** on **Databricks AI Runtime**
(serverless NVIDIA GPUs), then **optionally** deploy to **Databricks Model Serving**. It fine-tunes
**[ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base)**
— the current mainstream BERT-class encoder (RoPE, Flash Attention 2, 8k context, released Dec 2024)
— for **topic classification** on the public **AG News** dataset (4 classes: World, Sports, Business,
Sci/Tech), tracks everything with **MLflow**, registers the model to **Unity Catalog**, runs **GPU
batch inference**, and (optionally) deploys a **Model Serving** endpoint.

> AIR = AI Runtime.

## Two ways to run — one folder each

Each step ships in **two forms**, so you can pick whichever fits and neither needs editing:

- **`01_notebook/`** — Databricks notebooks (`# Databricks notebook source` `.py`). **Import and step
  through the cells** (top to bottom). Rich per-cell markdown; `%pip` cells install dependencies.
- **`02_cli/`** — plain Python scripts + AI Runtime CLI workload YAMLs. **Submit with `air run`.**
  No notebook markers; dependencies come from the YAML.

Both forms share the same logic and are driven by environment variables with sensible defaults.
`03_docs/` has the architecture write-up.

## What it demonstrates

| Step | Notebook / CLI script | AIR compute | Shows |
|------|-----------------------|-------------|-------|
| 1. Single-GPU fine-tune | `01_finetune_singlegpu.py` | `GPU_1xA10` | Fine-tuning, MLflow tracking, UC registration + `@champion` |
| 2. Multi-GPU fine-tune | `02_finetune_multigpu.py` | `GPU_8xH100` | 8-GPU **DDP** (notebook: `serverless_gpu`; CLI: `torchrun`) |
| 3. GPU batch inference | `03_batch_inference.py` | `GPU_1xA10` | Loading the `@champion` UC model, batched GPU scoring, CSV to a UC Volume |
| 4. Model Serving *(Optional)* | `04_serve.py` | Serving (GPU) | Real-time endpoint via **Databricks Model Serving** — a separate product from AI Runtime (control-plane) |

## What lands in the Databricks platform (both forms)

Beyond running on AI Runtime GPUs, every step is wired into the wider Databricks platform:

- **MLflow experiment tracking** — 01/02/03 log params, metrics and the model to an MLflow run
  (02 logs each config as its own run). View them in the workspace **Experiments** UI.
- **Unity Catalog Model Registry + versioning** — 01/02 register the model to
  `main.air_samples.modernbert_agnews`, creating a new **version** each run and promoting it to the
  **`@champion`** alias. 03 (batch inference) and 04 (serving) load `@champion`, so version
  promotion is explicit and governed — no manual step.
- **Unity Catalog Volumes** — 03 writes its prediction CSV to a UC Volume; 04 serves the registered
  model as a **Model Serving** endpoint.

## Prerequisites

- A **Databricks workspace where AI Runtime is enabled** — see the
  [AI Runtime documentation](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/) for
  current cloud/region availability.
- Permission to **create a Unity Catalog schema and volume** in some catalog (ask your admin which
  catalog you can write to, or use one you own).
- macOS/Linux/WSL with a terminal. (Validated on a workspace with AI Runtime enabled.)

## Setup — one time, ~10 minutes

Run these on your laptop. Replace `<workspace-url>` and pick a profile name (here `air`).

```bash
# a) Install the Databricks CLI
brew install databricks            # macOS; else: curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh
databricks --version               # need v0.230+

# b) Log in — this opens a browser and saves an auth "profile" named `air`
databricks auth login --host https://<workspace-url>.cloud.databricks.com --profile air
databricks current-user me --profile air     # should print your email

# c) (CLI users only — SKIP if you'll run the 01_notebook/ notebooks) Install the AI Runtime CLI (`air`)
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

The scripts default to catalog **`main`**, schema **`air_samples`** — the same values the Setup
step creates. **If you ran Setup with `CATALOG=main`, everything runs unchanged.** To use a
different catalog, set `UC_CATALOG` either way:

- edit the `UC_CATALOG` / `UC_SCHEMA` default lines near the top of each script, or
- prefix the YAML `command:` line, e.g.
  `command: UC_CATALOG=mycat python $CODE_SOURCE_PATH/02_cli/01_finetune_singlegpu.py`.

Use the **same profile name** you created (`air`) in every `air run --profile ...` below.

## Run it via the CLI (`02_cli/`) — do the steps in order

Steps 3 and 4 need the model that step 1 (or 2) registers, so **run 01 first.**

```bash
# macOS: the COPYFILE_DISABLE=1 prefix is REQUIRED — it keeps macOS ._* files out of the
# uploaded code snapshot (otherwise the job dies immediately). Harmless on Linux.

# 1) Fine-tune on one A10 GPU → MLflow → register to Unity Catalog  (~10-15 min incl. GPU wait)
COPYFILE_DISABLE=1 air run --file 02_cli/finetune_singlegpu.yaml --watch --profile air

# 2) (Optional) Fine-tune across 8× H100 with DDP (torchrun)  (larger GPU request; may wait for capacity)
COPYFILE_DISABLE=1 air run --file 02_cli/finetune_multigpu.yaml --watch --profile air

# 3) GPU batch inference over the AG News test set → predictions CSV on the UC Volume
COPYFILE_DISABLE=1 air run --file 02_cli/batch_inference.yaml --watch --profile air

# 4) (Optional) Deploy a real-time GPU serving endpoint, then query it. This uses Databricks Model
#    Serving (a separate product from AI Runtime). 04 is a control-plane script (not a GPU job), so
#    run it locally — install its one dependency first:
pip install -r requirements.txt          # or: pip install "mlflow>=2.15.0"
DATABRICKS_CONFIG_PROFILE=air python 02_cli/04_serve.py
```

Each `air run` ends with `Job status: SUCCESS` on success. The first run of each waits a few
minutes for a GPU to be provisioned — that is normal. (Note: `air logs` sometimes prints
"No logs available" even for successful runs; trust `Job status` and the MLflow links.)

## Run it as a notebook (`01_notebook/`)

Import a file from `01_notebook/` into your Databricks workspace (**Workspace → Import → File**),
then **attach a serverless AI Runtime GPU** — there is no cluster to create:

1. Open the **compute** drop-down at the top of the notebook → **Serverless GPU**.
2. Click the **environment** icon to open the **Environment** side panel.
3. Set **Accelerator** (`GPU_1xA10` for 01/03; **`GPU_8xH100`** for `02_finetune_multigpu.py`) and
   leave the default **Base environment**.
4. Click **Apply**, then **Confirm**.

Then **run the cells one at a time, top to bottom**, reviewing each step's output (Run All works
too). The `%pip` cells install dependencies automatically. Start with `01_finetune_singlegpu.py`,
then `03_batch_inference.py`, then (optionally) `04_serve.py`. See
[Connect to serverless GPU compute](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/connecting#gpu-compute).

- **`02_finetune_multigpu.py` must be attached to a `GPU_8xH100` AI Runtime compute** — not a
  generic/A10 one. As a notebook it runs `serverless_gpu` in local mode, which requires the attached
  GPU to match `gpu_type="H100"` (otherwise it raises `GPUTypeError`).
- The **first** run of each notebook waits several minutes (~5 min on A10, ~7 min on 8×H100) for GPU
  capacity before any cell executes — that's normal cold start, not a hang.
- `04_serve.py` is control-plane and runs on any compute (its `%pip` cell installs `mlflow`).

## Configuration (env vars, with defaults)

| Variable | Default | Meaning |
|----------|---------|---------|
| `UC_CATALOG` / `UC_SCHEMA` | `main` / `air_samples` | Unity Catalog target (matches Setup; change only to use another catalog) |
| `REGISTERED_MODEL_NAME` | `modernbert_agnews` | UC registered model name |
| `MODEL_NAME` | `answerdotai/ModernBERT-base` | Base model to fine-tune |
| `DATASET_NAME` | `fancyzhx/ag_news` | HuggingFace dataset |
| `MAX_TRAIN_SAMPLES` / `MAX_EVAL_SAMPLES` | `20000` / `2000` (single-GPU); `-1` (multi-GPU) | Sub-sampling; `-1` = full split |
| `EPOCHS`, `TRAIN_BATCH_SIZE`, `LEARNING_RATE`, `MAX_LENGTH` | see scripts | Training hyper-parameters |

Override per run by prefixing the YAML `command:` line, e.g.
`command: EPOCHS=3 MAX_TRAIN_SAMPLES=-1 python $CODE_SOURCE_PATH/02_cli/01_finetune_singlegpu.py`.

## Troubleshooting

| Symptom | Cause & fix |
|---------|-------------|
| Job dies in seconds, `cd: .../._xxx: Not a directory` | macOS AppleDouble files in the snapshot — always run with `COPYFILE_DISABLE=1` (see above). |
| `RESOURCE_DOES_NOT_EXIST` / schema or volume not found | Run the Setup step (d); make sure `UC_CATALOG`/`UC_SCHEMA` match what you created. |
| Job fails ~50s in with no logs | Usually a dependency-install issue — keep the pinned versions in the YAML; don't add heavy/unpinned packages. |
| `04_serve.py` → `ModuleNotFoundError: No module named 'mlflow'` (local run) | Install deps first: `pip install -r requirements.txt`. (Not needed in notebook mode — the `%pip` cell handles it.) |
| Step 03 log shows `spark-class ... ClassNotFoundException` / `dbconnect` errors | Harmless. AI Runtime GPU nodes have no Spark; these lines come from the runtime's Spark probe during MLflow logging (not from the demo code) and are safe to ignore — 03 writes a CSV to the UC Volume. |
| `02` notebook → `GPUTypeError: ... does not match the requested GPU type H100` | Attach the `02` notebook to a **`GPU_8xH100`** AI Runtime compute (see notebook notes). |
| `air logs` says "No logs available" | Known quirk; the run may still have succeeded. Check `Job status` and the MLflow run link. |
| Step 3/4 can't find the model | Run step 1 (or 2) first — it registers the model and sets the `@champion` alias that 03/04 load. |
| Long "waiting for GPU capacity" | Normal for H100; retry later or use the A10 steps. |

## Repo layout

```
air-sample-bert/
├── 01_notebook/               # Databricks notebooks — import + step through
│   ├── 01_finetune_singlegpu.py
│   ├── 02_finetune_multigpu.py
│   ├── 03_batch_inference.py
│   └── 04_serve.py
├── 02_cli/                    # AI Runtime CLI — plain scripts + workload YAMLs (air run)
│   ├── 01_finetune_singlegpu.py … 04_serve.py
│   ├── finetune_singlegpu.yaml
│   ├── finetune_multigpu.yaml
│   └── batch_inference.yaml
├── 03_docs/                   # architecture walkthrough & AI Runtime notes
│   └── architecture.md
├── setup.sh                   # one-time UC schema + volume creation
├── requirements.txt
└── README.md
```

See [`03_docs/architecture.md`](03_docs/architecture.md) for the architecture walkthrough and AI Runtime specifics.
