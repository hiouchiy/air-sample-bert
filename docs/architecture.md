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
              hiroshi.air_samples.modernbert_agnews
                     │                                  │
       03 batch inference (AIR GPU)         04 Model Serving (GPU endpoint)
       → UC Delta predictions table         → real-time topic classification
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

### Two launch modes for the same file (`02_finetune_multigpu.py`)

Multi-GPU has two different launch mechanisms on AI Runtime, and the file supports both with **no
code changes** by detecting `torchrun`'s `RANK`/`WORLD_SIZE` env vars in `main()`:

- **Notebook** (*Run All*): no `torchrun` env → the file calls `run_train.distributed()`, and
  `serverless_gpu`'s `@distributed` decorator fans `_train_impl` out across the node's GPUs. This is
  the notebook-native API and is single-node (≤ 8 GPUs).
- **AI Runtime CLI**: the workload YAML runs the file under `torchrun`
  (`command: torchrun --standalone --nproc_per_node=gpu $CODE_SOURCE_PATH/src/02_finetune_multigpu.py`).
  `air run` provisions the node and injects rendezvous env vars; `torchrun` starts one process per
  GPU and each runs `_train_impl()` directly. Hugging Face `Trainer` reads the env and does DDP.
  (`run_train.distributed()` does **not** work under the CLI — it errors with "cluster_id is
  required" — which is why the CLI path uses `torchrun` instead.)

For true **multi-node** (e.g. 16× H100 = 2 nodes) the CLI is the only option: set
`num_accelerators: 16`, and use the injected `--nnodes=$NUM_NODES --node_rank=$NODE_RANK
--nproc_per_node=$LOCAL_WORLD_SIZE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT` form of
the `torchrun` command.

## Notebook + CLI dual-mode (no code changes)

Each `src/*.py` carries Databricks notebook markers that are also valid Python comments:

- `# Databricks notebook source` — identifies the file as an importable notebook.
- `# COMMAND ----------` — cell separators.
- `# MAGIC %pip ...` / `# MAGIC %md ...` — notebook-only magic cells (install deps, render docs).

When the file is **imported into the workspace**, these become real cells (the `%pip` cells install
dependencies). When it is **run by the AI Runtime CLI** (`python …`), the `# MAGIC` lines are inert
comments and dependencies come from the workload YAML instead. A single
`if __name__ == "__main__": main()` triggers execution in both (Databricks notebooks expose
`__name__ == "__main__"`). All parameters are environment variables with defaults, so nothing needs
editing between modes.

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
| `src/01_finetune_singlegpu.py` + `air/finetune_singlegpu.yaml` | Single-GPU (A10) fine-tune → MLflow → UC |
| `src/02_finetune_multigpu.py` + `air/finetune_multigpu.yaml` | 8×H100 DDP fine-tune |
| `src/03_batch_inference.py` + `air/batch_inference.yaml` | GPU batch inference → UC Delta table |
| `src/04_serve.py` | Deploy/query a Model Serving endpoint (control-plane) |
