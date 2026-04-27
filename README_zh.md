<div align="center">
  <h1>Kwai Summary Attention (KSA)</h1>
  <p align="center">
    <strong>基于可学习 Summary Token 的高效长上下文建模</strong>
  </p>
  <p align="center">
    <a href="#-引用">
        <img alt="Paper" src="https://img.shields.io/badge/Paper-Technical%20Report-b31b1b?logo=arxiv" />
    </a>
    <a href="https://github.com/Kuaishou-OneRec">
        <img alt="GitHub" src="https://img.shields.io/badge/GitHub-Kuaishou--OneRec-black?logo=github" />
    </a>
    <a href="#-许可证">
        <img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-green" />
    </a>
  </p>
  <p align="center">
    <a href="README.md">English</a> | <a href="README_zh.md">中文</a>
  </p>
</div>
<br>

## 📖 简介

**Kwai Summary Attention（KSA）** 是一种高效注意力机制，通过在固定 chunk 边界插入少量 *可学习 Summary Token* 把历史上下文压缩成紧凑状态。相比 GQA/MLA 仍保留每 token 一份 KV cache，以及 SWA/线性注意力完全丢弃或有损压缩远距离历史，KSA 选择一条 **中间路线**：KV cache 以语义级压缩比 R 按 **O(N/R)** 缩放，用少量显存换取 *完整、可回溯、可解释* 的长程依赖。

本仓库提供：

- **Muse** 训练框架与 Qwen3 + Summary Attention 模型实现。
- 训练 / prefill 阶段的块稀疏 **kernel**（已打包为 wheel）。
- 解码阶段的 **ring-buffer KV cache**，以 HuggingFace `trust_remote_code` 模板形式发布。
- 一份把 Qwen3-1.6B 基座从 8k → 32k → 64k → 128k 渐进式训到位的 **端到端 pretraining recipe**。
- DCP → HuggingFace safetensors 的 **权重转换脚本** 与推理自测脚本。

<p align="center"><img src="./assets/figures/mainmodel.png" width="80%" alt="KSA 混合架构：Summary Token 与 Text Token 交织,Summary Attention 层与 Full Attention 层按 3:1 比例堆叠。" /></p>
<p align="center"><em>图：KSA 混合架构。Summary Token 与 Text Token 交织；Summary Attention 与 Full Attention 层按 3:1 比例堆叠。</em></p>

## 🔥 新闻

*Coming soon.* 技术报告和模型权重公开后将补上带日期的条目。

## ✨ 核心特性

- **序列级 KV 压缩。** Summary Token 将序列切分成大小为 $N$ 的 chunk，每个 chunk 的 Summary 作为远端历史的压缩先验。KV cache 增长从 $O(N)$ 变为 $O(N/R)$,并且与 GQA/MLA **正交** —— 各路压缩比相乘。
- **滑动 *Chunk* 而非滑动 *Window*。** 窗口边界对齐 chunk 边界,保证每个历史 chunk 要么整块可见(文本),要么以 Summary 形式可见,永不出现"部分重叠"的夹缝区 —— 这是朴素 SWA 在边界处丢信息的根源。
- **默认 Hybrid 结构。** 发布的 recipe 采用 `3:1` 的 *Summary : Full* 层间交织。少量 Full Attention 层充当跨 chunk 的信息整合通道,对长上下文检索稳定性至关重要。
- **解码阶段 Summary KV Cache。** KV 状态以单一连续 buffer 布局 `[scratch | current chunk | sliding chunks (ring) | summary buffer]`,每步解码只读一段连续切片 —— 不需要 `cat`、不需要 `gather`、不需要显式构造稠密 mask。详见 [`examples/pretrain/hf_template/modeling_qwen3sa.py`](examples/pretrain/hf_template/modeling_qwen3sa.py)。
- **训练/Prefill 用块稀疏 kernel。** 只把非零 block pair 从 HBM 搬到 SRAM,避开 $O(L^2)$ 的稠密 mask(否则 128k 根本跑不起)。已打包为 wheel,见 [`summary_attention_kernel/`](summary_attention_kernel/)。
- **三阶段训练 recipe。** Attention 蒸馏 → 参数退火 → 长度扩展,可通过 `run_pretrain_{8,32,64,128}k.sh` 直接复现。

## 🤖 Model Zoo

*Coming soon.* 技术报告正式发布时,将在 Hugging Face 同步放出预训练 checkpoint。

| 模型          | Backbone    | 参数量 | Context | 训练方式              | 链接  |
| :------------ | :---------- | :----- | :------ | :-------------------- | :---- |
| KSA-4B (CPT)  | Qwen3-4B    | 4B     | 128k    | Continual Pretraining | *TBD* |

1.6B *from-scratch* 配置只作为可复现的训练 recipe 提供,不会发布对应权重。

## 🏗️ 方法与架构

KSA 在 *语义* 层面压缩长上下文 —— 在固定 chunk 边界插入少量 **可学习 Summary Token**,然后把历史作为 chunk 序列处理,每个 chunk 要么以原始文本暴露,要么只留下它的 Summary 状态。

### 1. Sliding Chunk Attention

<p align="center"><img src="./assets/figures/sca_vs_swa.png" width="75%" alt="滑动窗口注意力可能切断一个 chunk,在边界处丢信息;滑动 chunk 注意力让窗口边界与 chunk 边界对齐,信息路径干净无冲突。" /></p>
<p align="center"><em>图:滑动 chunk 注意力让窗口边界与 chunk 边界对齐。朴素滑窗会切穿 chunk 丢边界信息。</em></p>

当窗口边界切穿某个 chunk 时,这个 chunk 既没被文本 token 完整覆盖,也不能算作"已被 Summary 代表"的远端 chunk —— 信息就从缝里漏掉了。KSA 让窗口按 chunk 粒度滑动,每个历史 chunk *要么* 整块以文本形式可见(在窗口内),*要么* 只以 Summary 形式可见(在窗口外),不重复也不遗漏。

### 2. Ring-buffer KV Cache

<p align="center"><img src="./assets/figures/buffer_layout.png" width="82%" alt="KSA 解码时的连续 KV cache 布局:scratch 槽、当前 chunk、sliding chunk 环形区、Summary buffer 共享一块物理 tensor。" /></p>
<p align="center"><em>图:解码阶段 KV cache 布局。每个逻辑分区都是同一块物理 tensor 的连续切片。</em></p>

scratch / current chunk / sliding ring / summary buffer 都是同一块物理 tensor 的连续切片。文本 attention 和 summary attention 各读一段即可。由于 RoPE 在进 cache 之前就已经施加,环形 buffer 里的物理位置与逻辑位置解耦。chunk eviction 是一次原地 copy,不需要重分配、不需要 concat、不需要稠密 mask。

### 3. 次线性 KV 增长

<p align="center"><img src="./assets/figures/kv_cache_comparison.png" width="65%" alt="不同机制下 KV cache 随序列长度的增长:Full attention 线性增长;SWA 虽然持平但丢失远端信息;KSA 次线性增长且保留对全部历史的压缩访问。" /></p>
<p align="center"><em>图:KV cache 随序列长度的增长曲线。</em></p>

### 4. 训练 Recipe

每个目标长度(8k → 32k → 64k → 128k)下,都循环执行三阶段:

1. **Attention 蒸馏** —— 用 Full-Attention teacher warm-up Summary Attention 参数。
2. **参数退火** —— 解冻全模型联合优化。
3. **长度扩展** —— 放大 `max_position_embeddings`,调整 RoPE base 继续训练。

每阶段具体超参见 [`examples/pretrain/README.md`](examples/pretrain/README.md)。

### 发布版模型配置

本次发布包含两套 recipe:1.6B hybrid 从零训练(仅配方,不发布权重)、以及 4B 连续预训练版本。

| 配置项                         | From Scratch (1.6B)  | Continual Pretraining (4B) |
| :----------------------------- | :------------------- | :------------------------- |
| 层数                           | 24                   | 36                         |
| Hidden size                    | 2048                 | 2560                       |
| Intermediate size              | 6144                 | 9728                       |
| Attention heads (Q / KV)       | 16 / 16              | 32 / 8                     |
| Head dimension                 | 128                  | 128                        |
| Hybrid 比例 (Summary : Full)   | 3 : 1                | 3 : 1                      |
| Summary chunk size             | 8                    | 8                          |
| Sliding chunk number           | 128                  | 128                        |
| Tied embeddings                | False                | True                       |

配置文件位于 [`examples/pretrain/model_config/model_config_1b6_hybrid.json`](examples/pretrain/model_config/model_config_1b6_hybrid.json),通过 `muse/models/` 中注册的 `Qwen3SummaryAttentionConfig` / `Qwen3SummaryModel` 加载。

## 📈 实验结果

*具体数字将在技术报告公开后补上。* 论文在以下维度上评估 KSA:

- **长上下文检索** —— RULER (4k–128k)、NIAH 单针/多针、LongBench v2。
- **通用知识与推理** —— MMLU、CMMLU、GSM8k、CMath、MBPP。
- **效率** —— KV cache 随序列长度的占用、不同 prefill 下的解码吞吐。

128k 场景下的关键结论(CPT 设置):

- **Hybrid-Summary** 在 RULER-128k 上超过 Full attention,KV 占用显著更小。
- **Hybrid-Summary** 在 4K–128K 全部 context 长度、全部 needle 插入深度下均接近满分的 NIAH 单针检索。
- **Hybrid-Summary** 在 16k prefill 下解码速度高于 Full attention,且优于 Hybrid-SWA 与 Hybrid-Ring-Linear。

## 🚀 快速开始

### 1. 构建参考镜像

Ubuntu 24.04 + CUDA 12.6 + Python 3.12 + PyTorch 2.6.0 + FlashAttention 2.7.4.post1,并预装块稀疏 kernel:

```bash
docker build -t ksa-train -f dockerfile/Dockerfile .
```

版本号来自真实训练机的 snapshot,完整列表见 [`dockerfile/requirements.txt`](dockerfile/requirements.txt)。偏好裸机部署时对齐同一份 pin 即可。

### 2. 配置环境变量

```bash
cp .env.example .env      # 修改路径
bash set_env.sh
```

run 脚本会自动 `export PYTHONPATH=$PWD:$PYTHONPATH`,把仓库根目录挂到 `PYTHONPATH` 就够了。

### 3. Pretrain(渐进式长度扩展)

四个阶段,每阶段在上一阶段权重基础上续训:

```bash
bash examples/pretrain/run_pretrain_8k.sh     # 1. 从零训 8k
bash examples/pretrain/run_pretrain_32k.sh    # 2. 扩到 32k
bash examples/pretrain/run_pretrain_64k.sh    # 3. 扩到 64k
bash examples/pretrain/run_pretrain_128k.sh   # 4. 扩到 128k
```

每个脚本顶部的 `CHECKPOINT_DIR` / `OUTPUT_DIR` 需要按你的存储路径调整。阶段通过 `mpirun` 启动,DCP checkpoint 和 dataloader 状态都落到 `$OUTPUT_DIR/global_stepN/`。中途 resume、chunked-CE 开关、各阶段超参详见 [`examples/pretrain/README.md`](examples/pretrain/README.md)。

### 4. 转换为 HuggingFace 格式

```bash
bash examples/pretrain/convert/convert_muse_to_hf.sh \
     /path/to/muse_outputs/1b6_sa_hybrid_128k \
     global_step5000 \
     examples/pretrain/hf_template
```

转换结果落在 `<OUTPUT_DIR>/<STEP>/hf/`,除了 remap 后的 safetensors,还会从 `hf_template/` 拷入 `modeling_qwen3sa.py` / `summary_context.py` / tokenizer 等。模板所需内容见 [`examples/pretrain/hf_template/README.md`](examples/pretrain/hf_template/README.md)。

### 5. 推理 —— 跑一次生成验证模型

```bash
python examples/inference/inference.py \
     --model_path /path/to/global_step5000/hf \
     --prompt "介绍一下你自己" \
     --device cuda:0
```

推理直接走 HuggingFace `AutoModelForCausalLM` + `trust_remote_code=True`,底层使用 `hf_template/modeling_qwen3sa.py` 中定义的 ring-buffer KV cache,无需额外框架胶水。

## 📁 仓库结构

```
.
├── muse/                           # 训练框架(models / layers / training loop)
│   ├── models/qwen3_sa/            # Qwen3 + Summary Attention 模型
│   ├── layers/summary_context.py   # SummaryBatchContext + mask 工具
│   └── ...
├── recipes/
│   └── pretrain_kai_summary_unified.py   # Pretrain 主入口
├── summary_attention_kernel/
│   ├── summary_attn-*.whl          # 块稀疏 SA kernel(训练 + prefill)
│   └── flash_attn_cute-*.whl       # kernel 依赖的 CuTe FlashAttention 构建
├── examples/
│   ├── pretrain/                   # 8k → 128k 渐进式 recipe
│   │   ├── model_config/           # model_config_1b6_hybrid.json
│   │   ├── dataset_config/         # 各 seq len 的 mmap 数据集描述
│   │   ├── run_pretrain_{8,32,64,128}k.sh
│   │   ├── convert/                # DCP → HF safetensors
│   │   └── hf_template/            # HF 兼容 modeling + config 模板
│   └── inference/
│       └── inference.py            # chat-style 快速自测
├── data/                           # (自行准备)mmap 语料
├── dockerfile/                     # 参考 Dockerfile + requirements.txt
└── README.md / README_zh.md
```

## 🛣️ Roadmap

正在推进:

- [ ] 技术报告上 arXiv。
- [ ] Hugging Face 发布 1.6B 预训练 checkpoint。
- [ ] 放出 4B 连续预训练 recipe 与 checkpoint。
- [ ] RULER / NIAH / LongBench v2 复现脚本。
- [ ] 内置 ring-buffer KV cache 的参考推理/Serving 栈。
- [ ] 更多消融与教程。

欢迎 issue / PR。

## 📜 引用

*BibTeX 会在技术报告正式公开后确定,现为占位版本:*

```bibtex
@techreport{kwai2026ksa,
  title       = {Kwai Summary Attention Technical Report},
  author      = {OneRec Team},
  year        = {2026},
  institution = {Kuaishou Technology},
  url         = {https://github.com/Kuaishou-OneRec}
}
```

## 🛡️ 许可证

本仓库代码基于 **Apache License 2.0** 发布,详见 [`LICENSE`](LICENSE)。模型权重开放后将遵循各自的 License。

## 🙏 致谢

KSA 构建在开源生态之上。感谢:

- **Qwen3** —— 提供 KSA 扩展所依赖的基座架构与 tokenizer。
- **FlashAttention** —— 我们块稀疏 kernel 背后的 dense attention 原语。
- **HuggingFace Transformers** —— 让 `trust_remote_code` 部署无缝衔接的模型 / tokenizer / generation 抽象。
- **PyTorch 分布式训练** —— FSDP、DCP、通信原语让大规模预训练可行。

由衷感谢这些项目的出色工作。
