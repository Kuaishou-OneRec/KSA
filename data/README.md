# Tokenized example data

The KSA V1 and V2 examples share these small mmap samples. They can be used to
check the training pipeline; they are not the complete research corpus.
Dataset terms are separate from the repository's code license.

Each prefix has two files:

- `.bin`: a contiguous little-endian uint32 token stream using the KSA tokenizer.
- `.bin.meta`: the dtype name `uint32`.

The V1 mixture is listed in `train_data_list_8k.txt`. The shared tokenizer is in
[assets/tokenizer](../assets/tokenizer/).

A causal training sample reads `sequence_length + 1` tokens, with a stride of
`sequence_length`. A tail without the extra token is not a complete sample;
the loader skips it without padding or modifying the file. The V2
[dataset config](../examples/ksa_v2/dataset_config/demo_128k.json) reads the
existing fineweb-edu sample directly, yielding seven complete 128K samples.

For your own corpus, use the same tokenizer and edit `data_prefix` in the
dataset JSON: `[weight, prefix, weight, prefix, ...]`. Prefixes omit `.bin`
and resolve from the repository root. Keep `seeded_sampling: true` for
checkpointable sampling and use at least one data worker.
