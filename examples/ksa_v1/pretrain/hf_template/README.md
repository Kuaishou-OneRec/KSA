# KSA V1 Hugging Face runtime

This directory supplies the V1 model code and HF configuration defaults.
The exporter reads architecture fields from the run's saved `model_config.json`
and copies tokenizer files from `assets/tokenizer/`.

Use the [V1 conversion command](../README.md#export-and-inference). The exported
directory contains real copies of all required files; the template itself is
not a pretrained model.
