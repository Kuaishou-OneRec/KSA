# HF Template

Populate this folder on the training machine with a working HF model snapshot
(Qwen3 + Summary Attention variant) **before** running
`examples/pretrain/convert/convert_muse_to_hf.sh`.

## Expected contents

| File | Purpose |
|---|---|
| `config.json` | HF config with `summary_*` fields matching your trained model |
| `generation_config.json` | Default generation settings |
| `tokenizer.json` / `tokenizer_config.json` / `special_tokens_map.json` | Tokenizer |
| `vocab.json` / `merges.txt` | Tokenizer vocab (if applicable) |
| `modeling_qwen3*.py` | HF-compatible modeling code with SA support |
| `summary_context.py` | Helper module imported by the modeling code |

Only the **weights** come from the Muse DCP — everything else above is copied
verbatim into `<OUTPUT_DIR>/<STEP>/hf/` by the convert script.

## Usage

```bash
bash examples/pretrain/convert/convert_muse_to_hf.sh \
    /path/to/muse_outputs/1b6_sa_hybrid_8k \
    global_step5000 \
    examples/pretrain/hf_template
```
