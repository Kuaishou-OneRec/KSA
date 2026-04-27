"""
快速测试 muse → HF 转换后的权重能否正常对话。

用法:
    python inference.py --model_path <hf_ckpt_dir> [--prompt "你的问题"]

前置:
    已用 examples/pretrain/convert 脚本把 muse checkpoint 转成 HF 格式 (含
    config.json / modeling_qwen3sa.py / summary_context.py / tokenizer 等),
    目录形如 .../global_stepXXXX/hf
"""

import argparse
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser(description="测试 HF-converted muse 模型的前向生成")
    parser.add_argument(
        "--model_path",
        default="../global_step1000/hf",
        help="HF 权重目录",
    )
    parser.add_argument("--prompt", default="介绍一下你自己", help="测试用 prompt")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default="cuda:0", help="auto / cpu / cuda / cuda:N")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.7)
    parser.add_argument("--greedy", action="store_true", help="禁用采样，使用贪心解码")
    args = parser.parse_args()

    print(f"[1/4] 加载 tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    print(f"[2/4] 加载模型: {args.model_path}")
    device_map = args.device if args.device != "auto" else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    print(f"       模型设备: {next(model.parameters()).device}")
    print(f"       参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    print(f"[3/4] 构造输入 (prompt: {args.prompt!r})")
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": args.prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
    else:
        inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)

    input_len = inputs["input_ids"].shape[-1]
    print(f"       输入 token 数: {input_len}")

    print(f"[4/4] 开始生成 (max_new_tokens={args.max_new_tokens})")
    gen_kwargs = dict(max_new_tokens=args.max_new_tokens)
    if args.greedy:
        gen_kwargs.update(do_sample=False)
    else:
        gen_kwargs.update(do_sample=True, temperature=args.temperature, top_p=args.top_p)

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)
    elapsed = time.time() - t0

    new_tokens = output_ids[0][input_len:]
    num_new = len(new_tokens)
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    print("\n" + "=" * 60)
    print("模型回复:")
    print("=" * 60)
    print(response)
    print("=" * 60)
    print(f"生成 {num_new} tokens, 耗时 {elapsed:.2f}s, 速度 {num_new / max(elapsed, 1e-6):.1f} tokens/s")


if __name__ == "__main__":
    main()
