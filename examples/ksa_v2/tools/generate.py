#!/usr/bin/env python3

"""Generate text with a local KSA V2 export and its bundled kernels."""

import argparse
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactively generate text with a local model.")
    parser.add_argument("--model", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--prompt", help="Omit for interactive text completion.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--apply-chat-template", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = args.model.resolve()
    # Select the export's kernels before importing either runtime or Transformers.
    for directory in (model_dir / "kernel/deps", model_dir / "kernel"):
        sys.path.insert(0, str(directory))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        attn_implementation="sdpa",
    ).eval()

    def complete(prompt: str) -> str:
        if args.apply_chat_template:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    if args.prompt is not None:
        print(complete(args.prompt))
        return
    print("Enter a prompt, or type 'exit' to stop.")
    while True:
        try:
            prompt = input("Prompt: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt.lower() in {"exit", "quit"}:
            break
        if prompt:
            print(complete(prompt))


if __name__ == "__main__":
    main()
