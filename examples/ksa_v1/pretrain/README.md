# KSA V1 pretraining

The V1 example trains a 1.9B Hybrid KSA model through 8K, 32K, 64K and 128K
contexts. It uses `recipes/ksa_v1.py`, the shared `muse/` framework and `ksa_attn`
kernels. For C-KSA and S-KSA, use [the V2 example](../../ksa_v2/README.md).

## Environment and data

Build the V1 image from the repository root:

```bash
docker build -t ksa-v1 -f dockerfile/Dockerfile .
```

V1 and V2 share `ksa_attn` and its CuTe dependency. Their HF runtimes use
Transformers 4.57.1 and 5.0.0 respectively; keep those environments separate.

Set the mmap prefixes in `dataset_config/` to your tokenized corpus.
`global_batch_size` must equal the data-parallel world size times
`GRAD_ACC` (one sequence per microbatch). The example JSON defaults to GBS8.
The dataset format is documented in [data](../../../data/).

## Launch

Run the same command on every node, with a different `NODE_RANK`.
The launchers use torchrun in the foreground; they do not configure SSH, install
dependencies, start detached jobs, or stop other processes.

```bash
export NNODES=1
export GPUS_PER_NODE=8
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

bash examples/ksa_v1/pretrain/run_pretrain_8k.sh --dry-run
bash examples/ksa_v1/pretrain/run_pretrain_8k.sh

CHECKPOINT_DIR=output/ksa_v1/1b9_hybrid_8k \
  bash examples/ksa_v1/pretrain/run_pretrain_32k.sh
CHECKPOINT_DIR=output/ksa_v1/1b9_hybrid_32k \
  bash examples/ksa_v1/pretrain/run_pretrain_64k.sh
CHECKPOINT_DIR=output/ksa_v1/1b9_hybrid_64k \
  bash examples/ksa_v1/pretrain/run_pretrain_128k.sh
```

Each context-extension stage loads parent weights and starts a fresh optimizer,
schedule and dataloader. Within a stage, set `RESUME=1` and `CHECKPOINT_DIR` to
that stage's run to restore weights, optimizer and dataloader. Keep the model,
dataset, topology, seed and schedule unchanged.

| Context | Updates | Learning rate | LR warmup / decay | Save interval |
| --- | ---: | ---: | ---: | ---: |
| 8K | 127200 | 4e-4 | 1000 / 1000 | 5000 |
| 32K | 1600 | 1e-5 | 100 / 100 | 250 |
| 64K | 1600 | 1e-5 | 100 / 100 | 250 |
| 128K | 1600 | 1e-5 | 100 / 100 | 250 |

Override `OUTPUT_DIR`, `MODEL_CONFIG`, `DATASET_CONFIG`, `LR`, `TOTAL_STEPS`,
`WARMUP_STEPS`, `DECAY_STEPS` and `SAVE_STEPS` through environment variables.
These are V1 example budgets, not the V2 pretraining schedule.

The model config uses a 3:1 KSA/Full pattern. KSA layers retain 128 text chunks;
Full layers cover the full 128K supported context. The run saves its resolved
`model_config.json` for export.

## Export and inference

```bash
bash examples/ksa_v1/pretrain/convert/convert_muse_to_hf.sh \
  output/ksa_v1/1b9_hybrid_8k \
  global_step5000 \
  examples/ksa_v1/pretrain/hf_template

python examples/ksa_v1/inference/inference.py \
  --model_path output/ksa_v1/1b9_hybrid_8k/global_step5000/hf \
  --prompt "The main idea of long-context modeling is" \
  --greedy
```

The converter derives the HF configuration from the saved training config and
copies the V1 runtime and [shared tokenizer](../../../assets/tokenizer/).
For older runs without a saved config, call `convert.py` with an explicit
`--model-config`; do not substitute a different experiment's config.
The V1 template supports independent summary QKV with shared Q/K normalization.
Use the V2 exporter for shared-QKV, C-KSA or S-KSA models.

Use the versioned training and conversion commands above; the old
`examples/pretrain/` entry points have been removed.
