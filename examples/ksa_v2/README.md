# KSA V2 training

Train the paper's 1.9B C-KSA + S-KSA model from the released KSA V1 1.9B
HF checkpoint. Each stage has one shell script that directly launches
[the training recipe](../../recipes/ksa_v2.py) with torchrun.

## Environment

Use Linux, Python 3.12, CUDA 12.6 and H100/H800 (SM90). From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
bash examples/ksa_v2/scripts/install.sh
```

V1 and V2 share `ksa_attn` and its CuTe dependency, but use different Transformers
versions; keep their environments separate. The installer pins the training
dependencies and installs the shared kernel.

## Data and parent checkpoint

The [demo dataset config](dataset_config/demo_128k.json) directly reads the
tokenized V1 sample in `data/`. No conversion or data copy is required.
It provides seven complete 128K training samples with a real look-ahead token;
the loader cycles over them without padding or modifying the source file.
This is smoke-test data, not the paper's training corpus.

For real training, point `DATASET_CONFIG` at your own weighted mmap mixture;
see [the data format](../../data/README.md). Token IDs must use the KSA tokenizer
in `assets/tokenizer/`, not merely the same vocabulary size.

Download [KSA-1.9B-base](https://huggingface.co/OpenOneRec/KSA-1.9B-base)
into `models/KSA-v1-1.9B`, or set `MODEL_DIR`
to its local directory. Compression warmup loads its weights directly, including
the Summary embedding; no internal training checkpoint is required. Its Adam
state, step counter and data cursor start fresh. The importer checks the backbone
configuration and loads every inherited tensor; only new compression parameters
are initialized.

Later stages read the Muse DCP checkpoints produced by this workflow. They
contain the weights and data cursor needed to continue training. To initialize
from an existing training checkpoint instead, set `CHECKPOINT_DIR` and
`CHECKPOINT_ID` on `compress_warmup.sh`. Match the tokenizer, dimensions, Summary
QKV sharing and RoPE; the separate V1 from-scratch example has different defaults.

## Three training stages

The 1.9B backbone has 24 layers, hidden width 2048, and 16 query / 16 KV heads
of width 128 (MHA), with a 128K context. The model configs use chunk8,
group128 compression and a 3:1 C-KSA/S-KSA layout.
S-KSA uses block4 and a max-pooled teacher. Its Indexer has 8 heads of width
128, query rank 512 and RoPE width 64, without Indexer Q/K normalization.
The remote budget is 4096 text tokens including 8 sink tokens, with an
additional 1024-token local window.

Each complete remote Meta group covers 8 × 128 = 1024 text tokens. This
1024× compression applies to those groups, not the total hybrid-model KV cache.
Separately released checkpoints may use different compression and normalization
settings; follow each model's own configuration and model card.

| Script | Trainable parameters | Objective | Steps | LR / warmup |
| --- | --- | --- | ---: | --- |
| [compress_warmup.sh](scripts/compress_warmup.sh) | C-KSA aggregation | CE + 5 × attention MSE | 1600 | 1e-4 / 200 |
| [full_cpt.sh](scripts/full_cpt.sh) | All | CE | 800 | 1e-4 / 400 |
| [indexer_warmup.sh](scripts/indexer_warmup.sh) | S-KSA aggregation and Indexer | Dense-teacher KL | 4400 | 1e-3 / 200 |

GBS32, 128K, warmup then constant LR; save every 200 updates, plus update 10
and the final update. The budgets are 6.7B / 3.4B / 18.5B text tokens.
Indexer warmup freezes all inherited parameters and keeps dense Full-layer
attention for supervision. Export that checkpoint directly for sparse inference.

Edit paths and hyperparameters at the top of each shell. Paths are relative to
the repository root. On each node, set its rank and run the same stage:

```bash
export NNODES=4
export GPUS_PER_NODE=8
export NODE_RANK=0
export MASTER_ADDR=YOUR_NODE0_ADDRESS
export MASTER_PORT=29500

bash examples/ksa_v2/scripts/compress_warmup.sh --dry-run
bash examples/ksa_v2/scripts/compress_warmup.sh
```

Use `NNODES=1` and `MASTER_ADDR=127.0.0.1` for one node. Gradient accumulation
is `GBS / (NNODES * GPUS_PER_NODE)`; context parallelism is 1.

Starting from HF, checkpoints are numbered `0 → 1600 → 2400 → 6800`:

```bash
bash examples/ksa_v2/scripts/full_cpt.sh
bash examples/ksa_v2/scripts/indexer_warmup.sh
```

Stages never auto-chain. Select `CHECKPOINT_DIR` and `CHECKPOINT_ID` explicitly
when using a different parent. Set `SOURCE_GBS` to the parent's global batch size
when changing batch size across stages. Stage transitions restore
weights and data but start new Adam/LR states. Only newly added modules may be
randomly initialized.

## Few-step check

With the HF parent downloaded, try the bundled demo data. The full model and
128K length are unchanged:

```bash
TOTAL_STEPS=3 \
WARMUP_STEPS=0 \
SAVE_EVERY=3 \
SAVE_STEPS=3 \
OUTPUT_DIR=output/ksa_v2/demo_compress \
bash examples/ksa_v2/scripts/compress_warmup.sh
```

Then use the new checkpoint as the next stage's parent:

```bash
CHECKPOINT_DIR=output/ksa_v2/demo_compress \
CHECKPOINT_ID=global_step3 \
TOTAL_STEPS=3 \
WARMUP_STEPS=0 \
SAVE_EVERY=3 \
SAVE_STEPS=3 \
OUTPUT_DIR=output/ksa_v2/demo_full_cpt \
bash examples/ksa_v2/scripts/full_cpt.sh
```

Then run Indexer warmup:

```bash
CHECKPOINT_DIR=output/ksa_v2/demo_full_cpt \
CHECKPOINT_ID=global_step6 \
TOTAL_STEPS=3 \
WARMUP_STEPS=0 \
SAVE_EVERY=3 \
SAVE_STEPS=3 \
OUTPUT_DIR=output/ksa_v2/demo_indexer \
bash examples/ksa_v2/scripts/indexer_warmup.sh
```

Export `output/ksa_v2/demo_indexer/global_step9` with that run's
`model_config.json` using the export command below. Short checks verify
execution, not convergence; the bundled samples are not sufficient for training
a useful model.

## Resume

To resume within a stage, restore weights, Adam, LR/step schedulers and data:

```bash
RESUME=1 \
CHECKPOINT_DIR=output/ksa_v2/indexer \
CHECKPOINT_ID=global_step3600 \
OUTPUT_DIR=output/ksa_v2/indexer_resume \
bash examples/ksa_v2/scripts/indexer_warmup.sh
```

Keep model, dataset, seed, topology, GBS and LR recipe unchanged.
`TOTAL_STEPS` is the total budget for the stage, not additional steps.
Always use a new output directory. Fused kernels do not guarantee bitwise
identical weights after resume.

## Export and inference

The recipe saves `model_config.json` and `dataset_config.json` in each run.
Export with that model configuration:

```bash
python examples/ksa_v2/tools/export_hf.py \
  --checkpoint output/ksa_v2/indexer/global_step6800 \
  --model-config output/ksa_v2/indexer/model_config.json \
  --output-dir output/ksa_v2/hf

python output/ksa_v2/hf/generate.py \
  --prompt "The main idea of long-context modeling is" \
  --max-new-tokens 64
```

The single `hf_template` uses the infra-optimized runtime for shared-QKV,
chunk8 KSA + Full, intermediate C-KSA + Full, and final C-KSA + block4 S-KSA.
For compression warmup or Full CPT, pass that stage's checkpoint and saved
`model_config.json`; Full layers remain dense and no Indexer is created.
Exports contain their tokenizer, kernels and inference requirements and run
outside this repository; use the bundled `generate.py` to select those kernels.
Install an export's requirements in a separate inference environment.

## Tests

```bash
python -m pip install pytest==8.4.1
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/ksa_v2 \
  -q \
  -p no:cacheprovider \
  --basetemp output/pytest

CUDA_VISIBLE_DEVICES=0 python tests/ksa_v2/smoke_test.py \
  --output output/smoke
```

The GPU check uses a tiny model, exercises HF initialization, all three public
shell scripts, checkpoint restore, frozen parameters, export and cached decoding.
Run it only on a reserved idle Hopper GPU. Use `--prepare-only` to inspect
the test fixtures without training.
