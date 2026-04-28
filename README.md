<div align="center">
  <h1>Kwai Summary Attention (KSA)</h1>
  <p align="center">
    <strong>Efficient long-context modeling via learnable summary tokens</strong>
  </p>
  <p align="center">
    <a href="https://arxiv.org/abs/2604.24432">
        <img alt="Paper" src="https://img.shields.io/badge/Paper-arXiv%3A2604.24432-b31b1b?logo=arxiv" />
    </a>
    <a href="https://github.com/Kuaishou-OneRec/KSA">
        <img alt="GitHub" src="https://img.shields.io/badge/GitHub-Kuaishou--OneRec-black?logo=github" />
    </a>
    <a href="#-license">
        <img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-green" />
    </a>
  </p>
  <p align="center">
    <a href="README.md">English</a> | <a href="README_zh.md">中文</a>
  </p>
</div>
<br>

## 📖 Introduction

**Kwai Summary Attention (KSA)** is an efficient attention mechanism that compresses historical context into a small set of *learnable summary tokens* inserted at regular chunk boundaries. Unlike GQA/MLA, which keep one cache entry per token, and unlike sliding-window or linear attention, which discard or lossily compress distant history, KSA takes an **intermediate path**: the KV cache scales as **O(N/R)** with a semantic-level compression ratio R, trading a small amount of memory for *complete, referential, and interpretable* retention of long-range dependencies.

This repository contains:

- **Muse** training framework with the Qwen3 + Summary Attention model.
- A block-sparse training / prefill **kernel** for Summary Attention.
- A ring-buffer **KV cache** implementation for decoding, packaged as a HuggingFace `trust_remote_code` template.
- A full end-to-end **pretraining recipe** that progressively extends a Qwen3-1.9B base from 8k to 128k context.
- Weight conversion utilities (DCP → HuggingFace safetensors) and an inference sanity-check script.

<p align="center"><img src="./assets/figures/mainmodel.png" width="80%" alt="KSA hybrid architecture: summary tokens interleaved with text tokens, with Summary Attention layers and Full Attention layers in a 3:1 hybrid ratio." /></p>
<p align="center"><em>Figure: KSA hybrid architecture. Summary tokens interleave with text tokens; Summary Attention and Full Attention layers are stacked in a 3:1 ratio.</em></p>

## 🔥 News

- **2026-04-28** — KSA technical report is released on arXiv: [arXiv:2604.24432](https://arxiv.org/abs/2604.24432).
- **2026-04-28** — Code, training recipes, block-sparse kernel, and HuggingFace `trust_remote_code` template are open-sourced under this repository.

## ✨ Highlights

- **Sequence-level KV compression.** Summary tokens partition the sequence into chunks of size $N$; the summary of each chunk acts as a compressed prior of distant history. KV cache grows as $O(N/R)$ instead of $O(N)$, and is **orthogonal** to GQA / MLA — the compression ratios multiply.
- **Sliding *chunk* attention, not sliding *window*.** Window boundaries are aligned with chunk boundaries so every past chunk is either fully visible (text) or summarized (summary token), never partially both. This avoids the information gap that naive SWA introduces at window edges.
- **Hybrid by default.** The released recipe uses a `3:1` *Summary : Full* layer interleaving. A small dose of full attention serves as a cross-chunk integrator and stabilizes long-context retrieval.
- **Summary KV cache for decoding.** KV states are laid out as a single contiguous buffer `[scratch | current chunk | sliding chunks (ring) | summary buffer]`. Every decode step reads one contiguous slice — no `cat`, no `gather`, no dense mask materialization. See [`examples/pretrain/hf_template/modeling_qwen3sa.py`](examples/pretrain/hf_template/modeling_qwen3sa.py).
- **Block-sparse training / prefill kernel.** Only non-empty block pairs are loaded from HBM to SRAM, avoiding the $O(L^2)$ mask materialization that would otherwise be infeasible at 128k. Distributed as a prebuilt wheel under [`summary_attention_kernel/`](summary_attention_kernel/).
- **Three-stage training recipe.** Attention distillation → parameter annealing → sequence-length extension, all reproducible via the `run_pretrain_{8,32,64,128}k.sh` launchers.

## 🤖 Model Zoo

*Coming soon.* Pretrained checkpoints will be published on Hugging Face once the technical report is released.

| Model         | Backbone    | Parameters | Context | Training              | Link  |
| :------------ | :---------- | :--------- | :------ | :-------------------- | :---- |
| KSA-4B (CPT)  | Qwen3-4B    | 4B         | 128k    | Continual pretraining | *TBD* |

The 1.9B *from-scratch* configuration is provided as a reproducible recipe only; no 1.9B weights will be released.

## 🏗️ Method & Architecture

KSA compresses long context at the *semantic* level by inserting a small number of **learnable summary tokens** at fixed chunk boundaries, then treating the past as a sequence of chunks — each exposed either as full text or as its summary state.

### 1. Sliding Chunk Attention

<p align="center"><img src="./assets/figures/sca_vs_swa.png" width="75%" alt="Sliding-window attention may cut through a chunk and lose boundary information; sliding-chunk attention aligns with chunk boundaries and guarantees clean information routing." /></p>
<p align="center"><em>Figure: Sliding-chunk attention aligns windows to chunk boundaries. Naive sliding windows cut through chunks and drop boundary information.</em></p>

If the window boundary cuts through a chunk, that chunk is neither fully covered by text tokens nor wholly summarized — its information falls through the cracks. KSA aligns windows to chunks so every past chunk is *exclusively* accessed either as full text (inside the window) or via its summary token (outside), with no double-counting and no gaps.

### 2. Ring-buffer KV Cache

<p align="center"><img src="./assets/figures/buffer_layout.png" width="82%" alt="Contiguous KV cache layout for KSA decoding: scratch slot, current chunk, sliding-chunk ring, and summary token buffer all share a single physical tensor." /></p>
<p align="center"><em>Figure: Decoding KV cache layout. Every logical region is a contiguous slice of a single physical tensor.</em></p>

Every logical region — scratch, current chunk, sliding ring, summary buffer — is a contiguous slice of a single tensor. Text attention and summary attention each read one span. RoPE is applied *before* caching, so physical position in the ring is independent of logical position. Chunk eviction is an in-place copy into the oldest ring slot; no reallocation, no concatenation, no dense mask.

### 3. Sub-linear KV Scaling

<p align="center"><img src="./assets/figures/kv_cache_comparison.png" width="65%" alt="KV cache growth vs. sequence length: Full attention grows linearly, SWA is flat but loses distant history, KSA grows sub-linearly while preserving a compressed trace of all history." /></p>
<p align="center"><em>Figure: KV cache growth vs. sequence length.</em></p>

### 4. Training Recipe

Three stages, repeated at each target sequence length (8k → 32k → 64k → 128k):

1. **Attention distillation** — warm up the summary-attention parameters against a Full-Attention teacher.
2. **Parameter annealing** — unfreeze the full model and jointly optimize.
3. **Sequence-length extension** — scale `max_position_embeddings` and resume with adjusted RoPE base.

See [`examples/pretrain/README.md`](examples/pretrain/README.md) for per-stage hyperparameters.

### Released model configuration

The release ships two recipes: a 1.9B hybrid model trained from scratch (recipe only — no weights released) and a 4B continual-pretraining variant.

| Configuration                 | From Scratch (1.9B) | Continual Pretraining (4B) |
| :---------------------------- | :------------------ | :------------------------- |
| Number of layers              | 24                  | 36                         |
| Hidden size                   | 2048                | 2560                       |
| Intermediate size             | 6144                | 9728                       |
| Attention heads (Q / KV)      | 16 / 16             | 32 / 8                     |
| Head dimension                | 128                 | 128                        |
| Hybrid ratio (Summary : Full) | 3 : 1               | 3 : 1                      |
| Summary chunk size            | 8                   | 8                          |
| Sliding chunk number          | 128                 | 128                        |
| Tied embeddings               | False               | True                       |

The config lives at [`examples/pretrain/model_config/model_config_1b9_hybrid.json`](examples/pretrain/model_config/model_config_1b9_hybrid.json) and is loaded via the `Qwen3SummaryAttentionConfig` / `Qwen3SummaryModel` registered in `muse/models/`.

## 📈 Performance

We evaluate KSA under two settings — **Continual Pretraining (CPT)** from a Qwen3-4B-base checkpoint (85B tokens), and **Train-from-Scratch** at 1.9B (400B tokens). Full results are in the [technical report](https://arxiv.org/abs/2604.24432); the highlights below are taken directly from its tables.

### Long-context retrieval — RULER (CPT, 4B)

| Benchmark   | Full      | Hybrid-SWA | Hybrid-SCA | Hybrid-Linear | KSA   | **Hybrid-KSA** |
| :---------- | :-------- | :--------- | :--------- | :------------ | :---- | :------------- |
| RULER-4K    | 92.88     | 91.30      | 86.02      | 86.39         | 91.55 | **92.97**      |
| RULER-8K    | **91.38** | 88.03      | 84.28      | 83.86         | 86.78 | 90.53          |
| RULER-16K   | **89.12** | 82.87      | 80.67      | 78.06         | 84.78 | 88.86          |
| RULER-32K   | 84.74     | 78.94      | 76.89      | 76.48         | 80.30 | **86.65**      |
| RULER-64K   | **78.16** | 73.88      | 68.88      | 73.50         | 76.09 | 76.04          |
| RULER-128K  | 65.86     | 66.27      | 60.94      | 67.98         | 66.81 | **71.67**      |

Hybrid-KSA leads at 4K, 32K, and 128K, and at **128K it surpasses Full attention by +5.81 points** while operating with a substantially smaller KV cache. Across all RULER lengths it is the strongest sub-quadratic alternative to Full attention.

### General benchmarks (CPT, 4B)

| Benchmark | Full      | Hybrid-SWA | Hybrid-SCA | Hybrid-Linear | KSA   | **Hybrid-KSA** |
| :-------- | :-------- | :--------- | :--------- | :------------ | :---- | :------------- |
| MMLU      | **71.83** | 70.57      | 69.83      | 64.33         | 70.73 | 70.50          |
| CMMLU     | **75.00** | 73.69      | 72.59      | 68.41         | 73.29 | 72.63          |
| C-Eval    | **73.66** | 72.36      | 71.66      | 67.42         | 72.14 | 72.66          |
| MMLU-Pro  | **46.36** | 45.23      | 45.11      | 38.83         | 45.70 | 45.39          |
| CMath     | 83.41     | **84.84**  | 83.16      | 79.09         | 84.58 | 84.25          |
| GSM8K     | **82.75** | 81.92      | 80.10      | 72.44         | 81.09 | 79.50          |
| MATH      | 47.48     | **48.24**  | 47.45      | 42.57         | 48.15 | 47.56          |
| MBPP      | 61.30     | 61.70      | 59.60      | 55.30         | 61.50 | **62.20**      |
| HumanEval | 58.54     | 61.89      | 61.89      | 54.58         | 60.97 | **62.50**      |
| **Avg.**  | 73.50     | 72.12      | 69.94      | 67.28         | 72.30 | **73.59**      |

KSA preserves full general capability under CPT — Hybrid-KSA's average **(73.59) edges out Full attention (73.50)**, with the smallest gap-to-Full of any sub-quadratic alternative.

### Train-from-scratch headlines (1.9B, 400B tokens)

- **RULER-128K**: Hybrid-KSA **65.35** vs. Full attention **48.75** ( **+16.60** ). Hybrid-KSA stays robust as length grows (80.65 → 65.35 from 4K to 128K), while Full attention collapses (76.08 → 48.75).
- **GSM8K**: Hybrid-KSA **59.14** vs. Full **48.29** ( **+10.85** ). **MATH**: **36.92** vs. **23.38** ( **+13.54** ).
- **MBPP / HumanEval**: best of all configurations at **36.40 / 31.71**.
- **Training loss**: Hybrid-KSA reaches the lowest final loss (**1.524**), below Hybrid-GDN (1.534), Hybrid-SWA (1.550), and Full (1.572).

### Needle-in-a-Haystack & RULER-128K subtasks (CPT)

Hybrid-KSA achieves **near-perfect single-needle retrieval across 4K–128K** at all needle depths, with only a minor dip at 128K. On RULER-128K subtasks it leads on **NIAH-Multivalue (98.75, +10.63 over Full)**, **VT (90.50, +30.0 over Full)**, **FWE (65.84)**, and **SQuAD (42.50)**.

### Inference efficiency (4B, 128K context)

- **KV cache**: 7.5 GB vs. 18.6 GB for Full attention — a **2.5× reduction**.
- **Decode throughput** at 16K prefill: **1.06× of Full attention**, vs. 0.73× for Hybrid-SWA and 0.81× for Hybrid-Ring-Linear.

## 🚀 Quick Start

### 1. Build the reference image

Ubuntu 24.04 + CUDA 12.6 + Python 3.12 + PyTorch 2.6.0 + FlashAttention 2.7.4.post1, with the block-sparse kernel preinstalled:

```bash
docker build -t ksa-train -f dockerfile/Dockerfile .
```

Versions are pinned from an actual training-host snapshot; see [`dockerfile/requirements.txt`](dockerfile/requirements.txt) for the full list. If you prefer bare-metal installation, mirror the same pins.

### 2. Configure environment variables

```bash
cp .env.example .env      # then edit paths
bash set_env.sh
```

The run scripts auto-export `PYTHONPATH=$PWD:$PYTHONPATH`, so keeping the repo root on `PYTHONPATH` is sufficient.

### 3. Pretrain (progressive length extension)

Four stages, each resuming weights from the previous:

```bash
bash examples/pretrain/run_pretrain_8k.sh     # 1. from scratch at 8k
bash examples/pretrain/run_pretrain_32k.sh    # 2. extend to 32k
bash examples/pretrain/run_pretrain_64k.sh    # 3. extend to 64k
bash examples/pretrain/run_pretrain_128k.sh   # 4. extend to 128k
```

Edit `CHECKPOINT_DIR` / `OUTPUT_DIR` at the top of each script to match your storage layout. Each stage launches via `mpirun` and writes DCP checkpoints + dataloader state to `$OUTPUT_DIR/global_stepN/`. See [`examples/pretrain/README.md`](examples/pretrain/README.md) for mid-run resume, chunked-CE toggles, and per-stage hyperparameters.

### 4. Convert a trained checkpoint to HuggingFace

```bash
bash examples/pretrain/convert/convert_muse_to_hf.sh \
     /path/to/muse_outputs/1b9_sa_hybrid_128k \
     global_step5000 \
     examples/pretrain/hf_template
```

The converted HF directory lands at `<OUTPUT_DIR>/<STEP>/hf/` and contains the remapped safetensors plus the `modeling_qwen3sa.py` / `summary_context.py` / tokenizer files from `hf_template/`. See [`examples/pretrain/hf_template/README.md`](examples/pretrain/hf_template/README.md) for the expected template contents.

### 5. Inference — sanity-check a converted model

```bash
python examples/inference/inference.py \
     --model_path /path/to/global_step5000/hf \
     --prompt "介绍一下你自己" \
     --device cuda:0
```

The inference path uses HuggingFace's `AutoModelForCausalLM` with `trust_remote_code=True` and goes through the ring-buffer KV cache defined in `hf_template/modeling_qwen3sa.py` — no framework-specific glue required.

## 📁 Repository Layout

```
.
├── muse/                           # Training framework (models, layers, training loop)
│   ├── models/qwen3_sa/            # Qwen3 + Summary Attention model
│   ├── layers/summary_context.py   # SummaryBatchContext + mask helpers
│   └── ...
├── recipes/
│   └── pretrain_kai_summary_unified.py   # Main pretrain entry
├── summary_attention_kernel/
│   ├── summary_attn-*.whl          # Block-sparse SA kernel (training + prefill)
│   └── flash_attn_cute-*.whl       # CuTe-based FlashAttention build used by the kernel
├── examples/
│   ├── pretrain/                   # Progressive 8k→128k recipe
│   │   ├── model_config/           # model_config_1b9_hybrid.json
│   │   ├── dataset_config/         # per-seq-length mmap dataset specs
│   │   ├── run_pretrain_{8,32,64,128}k.sh
│   │   ├── convert/                # DCP → HF safetensors
│   │   └── hf_template/            # HF-compatible modeling + config template
│   └── inference/
│       └── inference.py            # Quick chat-style sanity check
├── data/                           # (User-populated) mmap corpora
├── dockerfile/                     # Reference Dockerfile + requirements.txt
└── README.md / README_zh.md
```

## 🛣️ Roadmap

We are actively working on:

- [x] Technical report on arXiv ([arXiv:2604.24432](https://arxiv.org/abs/2604.24432)).
- [ ] Publish pretrained 1.9B checkpoints on Hugging Face.
- [ ] Release the 4B continual-pretraining recipe and checkpoint.
- [ ] Expanded evaluation scripts for RULER / NIAH / LongBench v2 reproduction.
- [ ] A reference serving stack with the ring-buffer KV cache.
- [ ] Additional ablations and tutorials.

Contributions are welcome — feel free to open an issue or PR.

## 📜 Citation

If you find KSA useful, please cite our technical report:

```bibtex
@techreport{kwai2026ksa,
  title       = {Kwai Summary Attention Technical Report},
  author      = {OneRec Team},
  year        = {2026},
  institution = {Kuaishou Technology},
  url         = {https://arxiv.org/abs/2604.24432}
}
```

## 🛡️ License

The code in this repository is licensed under the **Apache 2.0 License** (see [`LICENSE`](LICENSE)). Model weights, when released, will be subject to their own license agreements.

## 🙏 Acknowledgements

KSA is built upon and inspired by the open-source ecosystem. We would like to thank:

- **Qwen3** — for the base architecture and tokenizer that KSA extends.
- **FlashAttention** — for the dense-attention primitives our block-sparse kernel composes with.
- **HuggingFace Transformers** — for the model / tokenizer / generation abstractions that make `trust_remote_code` deployment painless.
- **PyTorch distributed training** — for FSDP, DCP, and the communication primitives that make large-scale pretraining tractable.

We sincerely thank these projects for their outstanding work.
