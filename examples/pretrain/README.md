# Pretrain

Progressive-length pretraining pipeline for Qwen3-1.6B Summary Attention
(hybrid variant). Train from scratch at 8k, then extend context to 32k → 64k
→ 128k, each stage resuming weights from the previous one.

## Layout

```
examples/pretrain/
├── model_config/                       # model architecture JSON
│   └── model_config_1b6_hybrid.json
├── dataset_config/                     # per-seq-length dataset JSONs
│   ├── pretrain_kai_mmap_8k.json
│   ├── pretrain_kai_mmap_32k.json
│   ├── pretrain_kai_mmap_64k.json
│   └── pretrain_kai_mmap_128k.json
├── run_pretrain_8k.sh                  # from-scratch 8k
├── run_pretrain_32k.sh                 # 32k, resumes 8k weights
├── run_pretrain_64k.sh                 # 64k, resumes 32k weights
├── run_pretrain_128k.sh                # 128k, resumes 64k weights
├── convert/                            # DCP → HF safetensors
│   ├── convert.py
│   ├── checkpoint.py
│   └── convert_muse_to_hf.sh
└── hf_template/                        # HF template copied into converted ckpt
    └── (populate on server, see hf_template/README.md)
```

## Pipeline

```
  run_pretrain_8k.sh   →  1b6_sa_hybrid_8k/   (from scratch)
          │
          ▼  resume weights
  run_pretrain_32k.sh  →  1b6_sa_hybrid_32k/
          │
          ▼  resume weights
  run_pretrain_64k.sh  →  1b6_sa_hybrid_64k/
          │
          ▼  resume weights
  run_pretrain_128k.sh →  1b6_sa_hybrid_128k/
          │
          ▼  convert
  global_stepN/hf/     (HF-inferable directory)
```

Each stage's output dir name matches the pattern
`1b6_sa_hybrid_{8,32,64,128}k`. The next stage's `CHECKPOINT_DIR` is set to
the previous stage's `OUTPUT_DIR` by default — edit the paths at the top of
each script if you store outputs elsewhere.

## Launching a stage

```bash
bash examples/pretrain/run_pretrain_8k.sh      # 1. from scratch
bash examples/pretrain/run_pretrain_32k.sh     # 2. after 8k finishes
bash examples/pretrain/run_pretrain_64k.sh     # 3. after 32k finishes
bash examples/pretrain/run_pretrain_128k.sh    # 4. after 64k finishes
```

Each script launches via `mpirun` + `nohup` and writes logs to
`$OUTPUT_DIR/stdout.log` / `stderr.log`.

## Key knobs (top of each `run_pretrain_*.sh`)

| Variable | Meaning |
|---|---|
| `CHECKPOINT_DIR` | Previous stage's output (or this stage's own output when resuming) |
| `OUTPUT_DIR` | Where this stage's DCP checkpoints + logs land |
| `TOTAL_STEPS` | Training step budget for this stage |
| `LR` / `MIN_LR` / `WARMUP_STEPS` / `DECAY_STEPS` | WSD schedule |
| `USE_CHUNKED_CE` | `1` = chunked CE loss (memory-efficient); `0` = standard CE |
| `RESUME_DATALOADER` | `0` = first launch (fresh loader); `1` = mid-run resume of this stage |

## Resuming a stage mid-run

If a 128k run dies halfway:

```bash
# edit run_pretrain_128k.sh
CHECKPOINT_DIR=/.../muse_outputs/1b6_sa_hybrid_128k   # point at THIS stage's output
RESUME_DATALOADER=1                                    # resume weights + dataloader
# then relaunch
bash examples/pretrain/run_pretrain_128k.sh
```

`--enable-dataset-checkpointing` is always on, so dataloader state is dumped
alongside each saved checkpoint and replayed when `RESUME_DATALOADER=1`.

## Converting to HuggingFace for eval

```bash
bash examples/pretrain/convert/convert_muse_to_hf.sh \
    /path/to/muse_outputs/1b6_sa_hybrid_8k \
    global_step5000 \
    examples/pretrain/hf_template
```

Populate `hf_template/` on the server first — see
[hf_template/README.md](hf_template/README.md).

Converted weights land at `<OUTPUT_DIR>/<STEP>/hf/`.
