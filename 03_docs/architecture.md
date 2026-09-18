# Architecture & AI Runtime notes

## End-to-end flow

```
                    ┌──────────────────────────────────────────────┐
   AG News (HF)  →  │  AI Runtime GPU node (serverless)             │
                    │                                              │
                    │  01 single-GPU  or  02 multi-GPU (8xH100 DDP)│
                    │        │  fine-tune ModernBERT               │
                    │        ▼                                      │
                    │  MLflow run (params, metrics, model)          │
                    │        │                                      │
                    │        ▼  log_model + register                │
                    └────────┼──────────────────────────────────────┘
                             ▼
              Unity Catalog registered model
              main.air_samples.modernbert_agnews  (promoted to @champion)
                     │                                  │
       03 batch inference (AIR GPU)         04 Model Serving (GPU endpoint)
       → predictions CSV on a UC Volume     → real-time topic classification
```

## Model & data

- **ModernBERT** (`answerdotai/ModernBERT-base`, 149M) — the mainstream BERT-class encoder as of
  2026: RoPE, alternating local/global attention, Flash Attention 2, 8192-token context. `-large`
  (395M) is a drop-in for the multi-GPU run.
- **AG News** — 120k train / 7.6k test news headlines+bodies, 4 balanced classes. English, public,
  and long enough to exercise ModernBERT's context window.

## Why the multi-GPU example uses DDP (not FSDP/DeepSpeed)

ModernBERT fits comfortably on a single 24 GB A10, so there is no need to *shard* the model.
The realistic reason to reach for 8× H100 is **throughput**: one process per GPU, each holding a
full replica (PyTorch **DDP**), with the global batch split across them. This is the honest, common
pattern for BERT-class training. FSDP/DeepSpeed only earn their complexity when a model does not fit
on one GPU.

### Two launch modes for multi-GPU (`02_finetune_multigpu.py`)

Multi-GPU has two different launch mechanisms on AI Runtime, and the repo ships a file tuned to each
(`01_notebook/` vs `02_cli/`):

- **Notebook** (`01_notebook/02_finetune_multigpu.py`, *Run All*): calls `run_train.distributed()`,
  and `serverless_gpu`'s `@distributed` decorator fans `_train_impl` out across the node's GPUs. This
  is the notebook-native API and is single-node (≤ 8 GPUs). Attach to a **`GPU_8xH100`** compute.
- **AI Runtime CLI** (`02_cli/02_finetune_multigpu.py`): the workload YAML runs the file under
  `torchrun` (`command: torchrun --standalone --nproc_per_node=gpu
  $CODE_SOURCE_PATH/02_cli/02_finetune_multigpu.py`). `air run` provisions the node and injects
  rendezvous env vars; `torchrun` starts one process per GPU and each runs `_train_impl()` directly.
  Hugging Face `Trainer` reads the env and does DDP. (`run_train.distributed()` does **not** work
  under the CLI — it errors with "cluster_id is required" — which is why the CLI file uses `torchrun`.)

For true **multi-node** (e.g. 16× H100 = 2 nodes) the CLI is the only option: set
`num_accelerators: 16`, and use the injected `--nnodes=$NUM_NODES --node_rank=$NODE_RANK
--nproc_per_node=$LOCAL_WORLD_SIZE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT` form of
the `torchrun` command.

## Notebook and CLI forms

Each step exists as two files that share the same logic:

- **`01_notebook/*.py`** — Databricks notebook source. It carries `# Databricks notebook source`,
  `# COMMAND ----------` cell separators, and `# MAGIC %pip`/`%md` cells. Imported into the
  workspace these become real cells (the `%pip` cells install dependencies) and *Run All* executes it.
- **`02_cli/*.py`** — the same logic as a plain Python script (notebook markers stripped), submitted
  with `air run`; dependencies come from the workload YAML (`environment.dependencies`) instead of
  `%pip`. Each pairs with a `02_cli/*.yaml` workload spec.

All parameters are environment variables with defaults, so neither form needs editing to run.

## AI Runtime operational notes (validated on this workspace)

These are the non-obvious things that make the samples run reliably on AI Runtime:

1. **Environment version 4 preinstalls** Python 3.12, `torch` 2.7.1+cu126, `mlflow`, `scikit-learn`
   and `serverless_gpu`. It does **not** include `transformers`, `datasets`, or `accelerate` — the
   YAML/`%pip` add only those. (Version 5+ drops `torch`; stay on 4.)
2. **Pin the NLP libraries.** `transformers>=4.48,<5` (5.x has breaking API changes) and
   `datasets>=2.19,<4` (unpinned pulls `pyarrow>=25`, which conflicts with the preinstalled
   `databricks-connect` and makes the YAML environment build fail).
3. **Install `hf_transfer`.** AI Runtime presets `HF_HUB_ENABLE_HF_TRANSFER=1`; without the package,
   every HuggingFace Hub download raises `ValueError`.
4. **Model logging/registration needs egress to the MLflow artifact store.** On this workspace
   (`e2-demo-field-eng`) AI Runtime GPU compute can reach the default artifact store, so the
   standard `mlflow.transformers.log_model(..., registered_model_name=...)` writes and registers to
   Unity Catalog directly. **Workspaces with stricter egress may block that host**
   (`*.storage.cloud.databricks.com`); there, save the model to a UC Volume with
   `mlflow.transformers.save_model` and register from a control-plane context that can reach the
   store. Validate on the target workspace before a customer demo.
5. **macOS submitters:** prefix `air run` with `COPYFILE_DISABLE=1` to keep AppleDouble `._*` files
   out of the code snapshot tarball.
6. **`air logs` may report "No logs available" even for successful runs.** Rely on MLflow
   (metrics/params/artifacts) and, for debugging, on writing to a UC Volume — not on `air logs` stdout.

## Files

| File | Role |
|------|------|
| `01_finetune_singlegpu.py` (+ `02_cli/finetune_singlegpu.yaml`) | Single-GPU (A10) fine-tune → MLflow → UC |
| `02_finetune_multigpu.py` (+ `02_cli/finetune_multigpu.yaml`) | 8×H100 DDP fine-tune |
| `03_batch_inference.py` (+ `02_cli/batch_inference.yaml`) | GPU batch inference → predictions CSV on a UC Volume |
| `04_serve.py` | Deploy/query a Model Serving endpoint (control-plane) |

Each of the above exists in both `01_notebook/` (Run All) and `02_cli/` (`air run`) form.
