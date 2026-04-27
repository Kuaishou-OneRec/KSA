import os
import re
import time
import torch
import datetime
import contextlib
import argparse
import json
import itertools
import torch.distributed as dist
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.distributed.device_mesh import init_device_mesh, DeviceMesh

from collections import defaultdict

import gc
gc.disable()

process_group_timeout = datetime.timedelta(minutes=60*24)

# Muse imports
from muse.models import get_model_class, list_models
from muse.config import load_config
from muse.training.distributed import (
    shard_model, 
    load_from_full_model_state_dict,
    initialize_model_params
)
from muse.training.checkpoint import (
    AppState,
    DistributedCheckpointer,
    load_hf_checkpoint,
    get_checkpoint_path,
    save_checkpoint,
    save_dataloader_state,
    load_dataloader_state
)
from muse.training.common import (
    set_default_dtype, 
    clip_grad_by_value, 
    compute_fsdp_zero2_grad_norm
)

from muse.utils.common import Timer

from muse.training.lr_schedulers import get_scheduler
from muse.training.activations import set_activation_checkpointing
from muse.training.parallel import (
    get_context_parallel_group,
    get_context_parallel_rank,
    get_context_parallel_world_size,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    initialize_model_parallel,
    broadcast_batch_tensors,
    split_batch_for_context_parallel,
)
from muse.utils.common import (
    set_random_seed, 
    print_rank_0,
    print_rank_n,
    to_cuda,
    to_device,
    dist_reduce_dict
)
from muse.data.datasets.kai_mmap import KaiMMapDataset
from muse.losses.distill import attention_mse_loss, attention_cosine_loss
from muse.losses.chunked_distill_loss_computer import ChunkedDistillLossComputer
from muse.layers.linear import TiedLinear
from muse.layers.summary_utils import maybe_build_summary_batch
from muse.utils.mfu import estimate_flops, get_gpu_peak_tflops

from muse.utils.metrics import Logger, StdoutBackend, CSVBackend, TensorBoardBackend
from muse.training.common import initialize_metrics, StepScheduler

def get_argument_parser():
  parser = argparse.ArgumentParser()

  ############ Model args ############
  parser.add_argument("--model-config", type=str, default=None,
                      help="The config file path of the model to train (required for train from scratch), e.g. model_dir/config.json")

  ############ Dataset args ############
  parser.add_argument("--dataset-class", type=str, default=None,
                      help="The dataset class name registered in muse.datasets.")

  parser.add_argument("--dataset-config", type=str, default=None,
                      help="The config file path of the dataset to train.")

  parser.add_argument("--max-length", type=int, default=None,
                      help="Max tokens per sentence in corpus")
  
  parser.add_argument("--batch-size", type=int, default=None,
                      help="Batch size for training")

  parser.add_argument("--shuffle-buffer-size", type=int, default=0,
                      help="Size of shuffle buffer for local data shuffling (0 to disable)")

  parser.add_argument("--use-dataset-load-balance", action="store_true",
                      help="Use load balance for dataset")

  parser.add_argument("--packing", action="store_true", default=True,
                      help="Whether to use packing for dataset")

  ############ Checkpoint args ############
  parser.add_argument("--model-dir", type=str, default=None,
                      help="The directory of the pretrained model (required for continue pretrain).")

  parser.add_argument("--checkpoint-dir", type=str, default=None,
                      help="Specify the checkpoint directory to resume from.")

  parser.add_argument("--checkpoint-id", type=str, default=None,
                      help="Specify the checkpoint id to resume from, e.g. global_step1000")

  ############ Resume control ############
  parser.add_argument("--resume-weights", action="store_true", default=False,
                      help="Load model weights from checkpoint")
  parser.add_argument("--resume-optimizer", action="store_true", default=False,
                      help="Resume optimizer + lr/step schedulers from checkpoint")
  parser.add_argument("--resume-dataloader", action="store_true", default=False,
                      help="Resume dataloader position from checkpoint")
  
  parser.add_argument("--save-checkpoint-per-step", type=int, default=1000,
                      help="The number of steps to save a checkpoint")

  parser.add_argument("--save-checkpoint-every-epoch", action="store_true",
                      help="Save checkpoint at the end of every epoch")
  
  parser.add_argument("--output-dir", type=str, default=None,
                      help="The directory to write the trained model")
  
  parser.add_argument("--model-dtype", type=str, default="bfloat16",
                      choices=["bfloat16", "float16", "float32"],
                      help="The dtype of the model.")

  parser.add_argument("--enable-dataset-checkpointing", action="store_true",
                      help="Enable dataset checkpoint recovery")
  
  ############ FSDP Args ############
  parser.add_argument("--cpu-offload", action="store_true",
                      help="Whether to offload parameters, gradients, and optimizer states to CPU")

  parser.add_argument("--fp32-weight", action="store_true",
                      help="Whether use fp32 for model weight updating")

  parser.add_argument("--fp32-reduce", action="store_true",
                      help="Whether use fp32 for model gradient reduction")

  parser.add_argument("--reshard-after-forward", action="store_true",
                      help="Reshard params after forward pass, aka Zero3.")
  
  parser.add_argument("--prefetch-params-in-forward", action="store_true",
                      help="Prefetch parameters in forward pass.")

  ############ Optimizer & Learning Rate Args ############
  parser.add_argument("--lr-scheduler-type", type=str, default="cosine",
                      help="The type of learning rate scheduler.")

  parser.add_argument("--num-warmup-steps", type=int, default=0,
                      help="The number of warmup steps to do.")
  
  parser.add_argument("--num-decay-steps", type=int, default=1000,
                      help="The number of steps to decay.")

  parser.add_argument("--num-training-steps", type=int, default=1000,
                      help="The number of training steps to do.")

  parser.add_argument("--num-epochs", type=int, default=1,
                      help="Number of epochs to train, no effect for pretraining.")
  
  parser.add_argument("--min-lr", type=float, default=1e-6,
                      help="The minimum learning rate to reach after the cosine schedule.")

  parser.add_argument("--learning-rate", type=float, default=2e-4,
                      help="The peak learning rate for optimizer.")

  # For AdamW optimizer
  parser.add_argument("--weight-decay", type=float, default=0.1,
                      help="The weight decay for Adam Optimizer")
  
  parser.add_argument("--beta1", type=float, default=0.9,
                      help="beta1 for Adam Optimizer")

  parser.add_argument("--beta2", type=float, default=0.95,
                      help="beta2 for Adam Optimizer")
  
  parser.add_argument("--clip-range", type=float, default=1.0,
                      help="The gradient clip range.")

  ############ Training Args ############

  parser.add_argument("--use-flash-attention-2", action="store_true",
                      help="Whether to use flash attention 2")

  parser.add_argument("--enable-gradient-checkpointing", action="store_true",
                      help="Enable gradient checkpointing during training")

  parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                      help="Gradient accumulation steps. "
                           "global_batch = dp_size * grad_acc micro-batches/step, "
                           "dp_size = world_size / cp_size")

  parser.add_argument("--allow-random-init-params", type=str, default='',
                      help="Parameter names to allow random initialization")

  parser.add_argument("--context-parallel-size", type=int, default=1,
                      help="Context parallelism size. dp_size = world_size / cp_size")

  parser.add_argument("--chunked-loss-minibatch-size", type=int, default=2048,
                      help="Chunk size along seq dim for chunked CE/KL loss computation. "
                           "Should be <= local seq_length (seq_length / cp_size when CP > 1)")

  parser.add_argument("--disable-chunked-loss", action="store_true",
                      help="Disable chunked CE loss. Use standard forward + CE loss. "
                           "Faster for short sequences but uses more memory.")

  parser.add_argument("--logging-per-step", type=int, default=100,
                      help="The number of steps to log training info")
  
  parser.add_argument("--monitor-datasource-loss", action="store_true",
                      help="Monitor loss per data source at logging boundaries")

  parser.add_argument("--log-attn-stats", action="store_true",
                      help="Log attention flow statistics (text_to_old_summary etc.) to TB at logging steps. "
                           "Only works with summary_attention_mode='mask'. Adds per-head computation overhead at logging steps.")

  parser.add_argument("--comment", type=str, default=None,
                      help="Comment of this experiment.")

  parser.add_argument("--commit-id", type=str, default=None,
                      help="Git commit id for experiment.")

  parser.add_argument("--seed", type=int, default=123,
                      help="Manual seed for RNG")

  ############ Profile Args ############

  parser.add_argument("--enable-profile", action="store_true",
                      help="Enable torch profile")

  ############ Debug Args ############

  parser.add_argument("--overfit-batches", type=int, default=None,
                      help="Number of batches to cache for overfitting (debug mode)")

  ############ Distillation Args ############
  parser.add_argument("--enable-distill", action="store_true", default=False,
                      help="Enable self-distillation (dual forward: student w/ summary + teacher w/o)")
  parser.add_argument("--distill-lm-loss-factor", type=float, default=1.0,
                      help="Weight for student LM cross-entropy loss")
  parser.add_argument("--distill-kl-loss-factor", type=float, default=5.0,
                      help="Weight for KL divergence loss on logits")
  parser.add_argument("--distill-attn-mse-factor", type=float, default=5.0,
                      help="Weight for attention MSE distillation loss")
  parser.add_argument("--distill-attn-cos-factor", type=float, default=0.0,
                      help="Weight for attention cosine distillation loss")

  return parser

# TODO: move to muse.utils
def _init_profiler(output_dir) -> None:
    import torch.distributed as D
    import os
    if not os.path.exists(output_dir):
      if D.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)

    def trace_handler(prof):
      prof.export_chrome_trace(
        os.path.join(
          output_dir, str(prof.step_num) + f"_w{dist.get_rank()}" + ".json")
      )

    torch_profiler = torch.profiler.profile(
      activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
      ],
      schedule=torch.profiler.schedule(
        wait=50,
        warmup=1,
        active=10,
        repeat=1,
      ),
      on_trace_ready=trace_handler,
    )
    return torch_profiler


def _get_batch_source_name(batch):
    data_source = batch.get("data_source", None)
    if isinstance(data_source, str):
        return data_source
    if isinstance(data_source, (list, tuple)):
        if len(data_source) == 0:
            return "unknown"
        unique_sources = list(dict.fromkeys(data_source))
        if len(unique_sources) == 1:
            return unique_sources[0]
        return "mixed"
    return "unknown"


def _get_batch_domain_name(batch):
    data_domain = batch.get("data_domain", None)
    if isinstance(data_domain, str):
        return data_domain
    if isinstance(data_domain, (list, tuple)):
        if len(data_domain) == 0:
            return "unknown"
        unique = list(dict.fromkeys(data_domain))
        if len(unique) == 1:
            return unique[0]
        return "mixed"
    return _get_batch_source_name(batch)


def _get_source_alias(source_name: str) -> str:
    parts = str(source_name).rstrip("/").rsplit("/", 2)
    return "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1] or "unknown"


def train():
  arg_parser = get_argument_parser()
  args = arg_parser.parse_args()

  assert all([args.commit_id, args.seed, args.comment]), \
    "Git commit, seed, and comment is required for reproducibility"

  assert any([args.save_checkpoint_per_step, args.save_checkpoint_every_epoch]), \
      "The checkpoint saving frequency is not set, save_checkpoint_per_step or " \
      "save_checkpoint_every_epoch should be set."

  rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))
  world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", 0))
  local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))


  ##############
  with open(args.dataset_config, encoding="utf-8") as f:
    dataset_config = json.loads(f.read())

  dataset_name = dataset_config.pop("name", None)
  
  # Determine training mode and get model_class
  if args.model_config:
    # Explicit --model-config always takes priority
    model_config = load_config(args.model_config)
    if args.model_dir:
      print_rank_0(
        f"NOTE: --model-config ({args.model_config}) overrides config.json "
        f"in --model-dir ({args.model_dir}). Weights loaded from model-dir, "
        f"config from model-config."
      )
  elif args.model_dir:
    # Continue pretrain mode: get model_class from model_dir/config.json
    model_config_path = Path(args.model_dir) / "config.json"
    if not model_config_path.exists():
      raise FileNotFoundError(
        f"Config file not found: {model_config_path}. "
        f"Cannot continue pretrain without config.json in {args.model_dir}"
      )
    model_config = load_config(model_config_path)
  else:
    raise ValueError(
      "Either --model-config or --model-dir (with config.json) must be provided."
    )

  if args.use_flash_attention_2:
    model_config.attention_function = "flash_attention_2"
    print_rank_0("Use flash attention 2")
  else:
    print_rank_0("Warning: Use eager attention, performance may be degraded.")

  # Enable distillation attention output collection when distill is on
  if args.enable_distill:
    model_config.enable_summary_distill_attention = True
    print_rank_0(f"Distillation enabled: lm={args.distill_lm_loss_factor}, "
                 f"kl={args.distill_kl_loss_factor}, mse={args.distill_attn_mse_factor}, "
                 f"cos={args.distill_attn_cos_factor}")

  model_class_name = model_config.model_class

  # torch init
  print_rank_n(f"torch init rank={rank}, local_rank={local_rank}")
  torch.cuda.set_device(local_rank)
  torch.distributed.init_process_group(
    rank=rank, world_size=world_size,
    timeout=process_group_timeout
  )
  device_mesh = init_device_mesh("cuda", mesh_shape=(dist.get_world_size(),))

  ### initialize model parallel group
  initialize_model_parallel(context_parallel_size=args.context_parallel_size)
  cp_size = get_context_parallel_world_size()
  dp_size = get_data_parallel_world_size()

  print_rank_0(f"=" * 60)
  print_rank_0(f"Parallelism Configuration:")
  print_rank_0(f"  World size:           {dist.get_world_size()}")
  print_rank_0(f"  Context parallel (CP): {cp_size}")
  print_rank_0(f"  Data parallel (DP):    {dp_size}")
  print_rank_0(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
  print_rank_0(f"  Chunked loss minibatch: {args.chunked_loss_minibatch_size}")
  print_rank_0(f"  Global batch size:     {dp_size} x {args.gradient_accumulation_steps} = {dp_size * args.gradient_accumulation_steps} micro-batches/step")
  print_rank_0(f"=" * 60)

  set_random_seed(args.seed)
  if dist.get_rank() == 0:
    os.makedirs(args.output_dir, exist_ok=True)

  if dist.get_rank() == 0:
    args_str = json.dumps(vars(args), indent=2, ensure_ascii=False)
    print_rank_0(f"Training Arguments:\n{args_str}")
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    with open(os.path.join(args.output_dir,
          f"args-{args.commit_id}-{timestamp}.json"), 'w',
        encoding="utf-8") as f:
      f.write(args_str + "\n")

  # Get model class from registry
  print_rank_0(f"Available models: {list_models()}")
  print_rank_0(f"Loading model class: {model_class_name}")
  
  try:
    model_cls = get_model_class(model_class_name)
    print_rank_0(f"Get model class: {model_cls.__name__}")
  except KeyError:
    print_rank_0(
      f"Unavailable model: {model_class_name}, " \
      f"please choose from available models: {list_models()}")
    return

  # Load state dict and convert using model's converter (only for continue pretrain)
  state_dict = None
  
  # Load state_dict to CPU only on rank 0 to avoid CPU OOM
  if args.model_dir:
    # Continue pretrain: load weights from checkpoint
    if dist.get_rank() == 0:
      with set_default_dtype(args.model_dtype):
        print_rank_0(f"Loading checkpoint from: {args.model_dir}")
        state_dict = load_hf_checkpoint(args.model_dir)
    dist.barrier()
  else:
    # Train from scratch: no weights to load
    state_dict = None
    dist.barrier()

  # TODO: support wandb
  tb_writer = None
  if dist.get_rank() == 0:
    tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "log"))
    tb_writer.add_text("comment", args.comment, 0)
    tb_writer.add_text("comment_id", args.commit_id, 0)

  # Instantiate model on meta device, this is to avoid OOM
  with set_default_dtype(args.model_dtype), torch.device("meta"):
    # Train from scratch: create model with random initialization
    print_rank_0(f"Creating model from config: {args.model_config}")
    model = model_cls(model_config)
    print_rank_0(f"Model instantiated from config: {type(model).__name__}")
  
  if args.enable_gradient_checkpointing:
    print_rank_0("Enable gradient checkpointing")
    set_activation_checkpointing(
      model, auto_wrap_policy=model.get_checkpointable_module_classes()
    )

  # upcast fp32 to maintain master weight.
  # We need to save a fp32 model weight, otherwise the precision of the optimizer 
  # updating the weight will be reduced, affecting convergence
  if args.fp32_weight:
    model = model.float()

  # Shard model for distributed training
  shard_model(
    model=model,
    cpu_offload=args.cpu_offload,
    reshard_after_forward=args.reshard_after_forward,
    dp_mesh=device_mesh,
    fp32_weight=args.fp32_weight,
    prefetch_params_in_forward=args.prefetch_params_in_forward,
    fp32_reduce=args.fp32_reduce
  )
  dist.barrier()
  # 需要保证每个rank都执行了参数初始化或加载
  if args.model_dir:
    with Timer("Load state dict"):
      # Convert meta tensors to CUDA tensors
      # distribute the state_dict from rank 0 to all ranks
      load_from_full_model_state_dict(
        model=model, full_sd=state_dict,
        allow_random_init_params=args.allow_random_init_params
      )
  else:
    # Train from scratch: initialize model parameters randomly
    with Timer("Initialize model parameters"):
      initialize_model_params(model)

  with torch.device(torch.cuda.current_device()):
    # Initialize RoPE, if the buffer is not in the state_dict,
    # it still on meta device, so we need to initialize it here
    for m in model.modules():
      # RoPE is not covered in state dict
      if hasattr(m, "rope_init"):
        print_rank_0("Initialize RoPE")
        m.rope_init()

  # Check if all parameters & buffers are initialized
  for name, tensor in itertools.chain(model.named_parameters(), model.named_buffers()):
    assert tensor.device != torch.device("meta"), \
      f"{name} not initialized, device={tensor.device}"

  if state_dict is not None:
    # Free the state_dict to save memory
    del state_dict

  # Freeze non-summary parameters during distillation warmup
  # (aligned with Megatron: only summary_embedding and *_summary.* are trainable)
  if args.enable_distill:
    summary_keywords = ["summary_embedding", "_summary"]
    frozen_count = 0
    trainable_count = 0
    for name, param in model.named_parameters():
      if any(kw in name for kw in summary_keywords):
        param.requires_grad = True
        trainable_count += 1
      else:
        param.requires_grad = False
        frozen_count += 1
    print_rank_0(f"Distillation freeze: {frozen_count} params frozen, "
                 f"{trainable_count} summary params trainable")

  # Print trainable parameters
  print_rank_0("=" * 50)
  print_rank_0("Parameters:")
  for name, param in model.named_parameters():
    if param.requires_grad:
      print_rank_0(f"  {name}: {param.shape}")
    else:
      print_rank_0(f"  {name}: {param.shape} (not trainable)")
  print_rank_0("=" * 50)

  # ---- Loss setup ----
  if not args.disable_chunked_loss:
    model.model.skip_output_layer = True  # unembed() returns norm(h), skips lm_head
    output_proj = model.model.output
    if isinstance(output_proj, TiedLinear):
        # Wrap TiedLinear so ChunkedDistillLossComputer can access
        # .weight, .bias, .out_features, and .parameters().
        # Gradient ordering is already handled in the loss computer
        # (backward first, then lm_head grad accumulation).
        class _TiedLMHead(torch.nn.Module):
            def __init__(self, tied_module):
                super().__init__()
                self.tied_module = tied_module
                self.out_features = tied_module.weight.shape[0]
                self.bias = None

            @property
            def weight(self):
                return self.tied_module.weight

        lm_head = _TiedLMHead(output_proj.tied_module)
        print_rank_0("Chunked CE: using TiedLinear lm_head (weight shared with tok_embeddings)")
    else:
        lm_head = output_proj  # nn.Linear
    print_rank_0(f"Chunked CE: lm_head extracted (out_features={lm_head.out_features}), "
                 f"minibatch_size={args.chunked_loss_minibatch_size}")

    chunked_loss = ChunkedDistillLossComputer(
        lm_head=lm_head,
        minibatch_size=args.chunked_loss_minibatch_size,
        lm_factor=args.distill_lm_loss_factor if args.enable_distill else 1.0,
        kl_factor=args.distill_kl_loss_factor if args.enable_distill else 0.0,
        enable_kl=args.enable_distill,
    )
    ce_loss_fn = None
    print_rank_0(f"ChunkedDistillLossComputer ready (enable_kl={args.enable_distill})")
  else:
    chunked_loss = None
    lm_head = None
    ce_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    print_rank_0("Standard CrossEntropyLoss ready (chunked loss disabled)")

  # TODO: support other optimizers
  # Prepare optimizer
  optimizer = torch.optim.AdamW(
    model.get_optimizer_grouped_parameters(
      learning_rate=args.learning_rate,
      weight_decay=args.weight_decay
    ),
    lr=args.learning_rate,
    betas=(args.beta1, args.beta2),
    eps=1.0e-8
  )

  lr_scheduler = get_scheduler(
    name=args.lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=args.num_warmup_steps,
    num_training_steps=args.num_training_steps,
    num_decay_steps=args.num_decay_steps,
    min_lr=args.min_lr
  )

  scheduler = StepScheduler(args)

  if dist.get_rank() == 0:
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    with open(os.path.join(args.output_dir,
        f"dataset-{args.commit_id}-{timestamp}.json"), 'w',
        encoding="utf-8") as f:
      f.write(json.dumps(
        dataset_config, ensure_ascii=False, indent=2) + "\n")

  assert dataset_name == "KaiMMapDataset", \
    f"Only KaiMMapDataset is supported; got {dataset_name}"

  # max_length 优先级: CLI --max-length > dataset_config
  if args.max_length is not None:
    dataset_config["seq_length"] = args.max_length
    print_rank_0(f"Override max_length with --max-length: {args.max_length}")

  # Build dataloader
  dataloader = None
  with Timer("Build dataloader"):
    print_rank_0(f"Building dataloader with config: {dataset_config}")
    dataset_config["rank"] = get_data_parallel_rank()
    dataset_config["world_size"] = get_data_parallel_world_size()
    dataset = KaiMMapDataset(**dataset_config)
    dataloader = DataLoader(
      dataset,
      batch_size=1,
      shuffle=False,
      num_workers=dataset_config.get("num_workers", 1),
      collate_fn=lambda x: x[0]
    )

  # Build resume control from independent switches
  # NOTE: dataloader state is saved/loaded independently (not through DCP)
  # to support soft matching when dataset sources change between runs.
  skip_training_state_keys = []
  if not args.resume_optimizer:
    skip_training_state_keys.extend(["step_scheduler", "lr_scheduler"])

  training_state = {
    "step_scheduler": scheduler,
    "lr_scheduler": lr_scheduler,
  }
  app_state = AppState(
    model=model,
    optimizer=optimizer,
    training_state=training_state,
    skip_optimizer=not args.resume_optimizer,
    skip_training_state_keys=skip_training_state_keys,
  )

  dist_checkpointer = DistributedCheckpointer()
  if args.checkpoint_dir and args.resume_weights:
    print_rank_0(
      f"Resume from checkpoint: {args.checkpoint_dir}, tag={args.checkpoint_id}, "
      f"resume_optimizer={args.resume_optimizer}, "
      f"resume_dataloader={args.resume_dataloader}")

    state_dict = {"app": app_state}
    checkpoint_path = get_checkpoint_path(
      args.checkpoint_dir, args.checkpoint_id)

    if checkpoint_path is None:
      raise FileNotFoundError(
        f"No checkpoint found in {args.checkpoint_dir} "
        f"(checkpoint_id={args.checkpoint_id}). "
        "Check --checkpoint-dir and --checkpoint-id."
      )

    dist_checkpointer.load_checkpoint(
        state_dict=state_dict,
        checkpoint_path=checkpoint_path,
    )

    # step_offset = checkpoint_step - scheduler.global_step
    # Handles all resume combinations:
    #   --resume-weights only:    scheduler=0, offset = ckpt_step (numbering continues)
    #   --resume-optimizer:       scheduler restored to N, offset = ckpt_step - N
    #                             (preserves prior offset from cross-phase checkpoints)
    _m = re.search(r'global_step(\d+)', os.path.basename(checkpoint_path))
    ckpt_step = int(_m.group(1)) if _m else 0
    step_offset = ckpt_step - scheduler.global_step
    print_rank_0(f"Loaded checkpoint: ckpt_step={ckpt_step}, "
                 f"scheduler.global_step={scheduler.global_step}, "
                 f"step_offset={step_offset}")

    # Load dataloader state independently (soft matching: new datasets start
    # from 0, removed datasets are ignored — Megatron-style).
    if args.resume_dataloader:
      loaded = load_dataloader_state(dataset, args.checkpoint_dir, args.checkpoint_id)
      if loaded:
        print_rank_0("Dataloader state resumed from checkpoint (soft matching)")
      else:
        print_rank_0("Warning: --resume-dataloader set but no dataloader_state.pt found, starting from 0")

  elif args.checkpoint_dir and not args.resume_weights:
    print_rank_0(
      f"Checkpoint dir provided ({args.checkpoint_dir}) but --resume-weights not set. "
      "Skipping checkpoint loading."
    )
    step_offset = 0

    # Still allow standalone dataloader resume (e.g., with --model-dir for weights
    # and --checkpoint-dir just for dataloader state).
    if args.resume_dataloader:
      loaded = load_dataloader_state(dataset, args.checkpoint_dir, args.checkpoint_id)
      if loaded:
        print_rank_0("Dataloader state resumed from checkpoint (standalone, soft matching)")
      else:
        print_rank_0("Warning: --resume-dataloader set but no dataloader_state.pt found, starting from 0")

  else:
    step_offset = 0

  dist.barrier()

  ##############
  torch_profiler = _init_profiler(
    output_dir=os.path.join(args.output_dir, "torch_profile")) \
      if args.enable_profile else None

  if dist.get_rank() == 0:
    stdout_logger = Logger("stdout", [StdoutBackend()])
    csv_logger = Logger("csv", [CSVBackend(os.path.join(args.output_dir, "metrics.csv"))])
    tb_logger = Logger("tb", [TensorBoardBackend(args.output_dir)])
    loggers = [stdout_logger, csv_logger, tb_logger]
  else:
    loggers = []

  # Initialize metrics and step scheduler
  metrics = initialize_metrics(
    acc_steps=args.gradient_accumulation_steps,
    logging_per_step=args.logging_per_step,
    loggers=loggers,
    dp_size=dp_size
  )

  # Register distillation metrics
  if args.enable_distill:
    acc = args.gradient_accumulation_steps
    lps = args.logging_per_step
    metrics.new("distill_lm_loss", dtype="float", reduce="mean")
    metrics.new("distill_kl_loss", dtype="float", reduce="mean")
    metrics.new("distill_attn_mse", dtype="float", reduce="mean")
    metrics.new("distill_total_loss", dtype="float", reduce="mean")
    for name in ["distill_lm_loss", "distill_kl_loss", "distill_attn_mse", "distill_total_loss"]:
        series = getattr(metrics, name).avg(window=acc)[::acc][1:]
        metrics.logger.track(
            series.avg(window=lps)[::lps],
            name=name, group="distill")

  domain_window_stats = defaultdict(
    lambda: {"loss_sum": 0.0, "tokens": 0}
  )
  source_window_stats = defaultdict(
    lambda: {"loss_sum": 0.0, "tokens": 0}
  )

  # ---- MFU setup (config-only, no model code dependency) ----
  seq_length = dataset_config.get("seq_length", args.max_length or 32768)
  gpu_peak_tflops = get_gpu_peak_tflops()
  model_flops_per_micro, hw_flops_per_micro = estimate_flops(
    num_layers=model_config.num_layers,
    hidden_size=model_config.embed_dim,
    head_dim=model_config.head_dim,
    num_heads=model_config.num_heads,
    num_kv_heads=model_config.num_kv_heads,
    intermediate_size=model_config.intermediate_dim,
    vocab_size=model_config.vocab_size,
    seq_length=seq_length,
    batch_size=1,
    gradient_checkpointing=args.enable_gradient_checkpointing,
    distill=args.enable_distill,
    summary_chunk_size=getattr(model_config, 'summary_chunk_size', 0),
    summary_token_num=getattr(model_config, 'summary_token_num', 0),
  )
  print_rank_0(f"MFU setup: model={model_flops_per_micro/1e12:.1f} TFLOP, "
               f"hardware={hw_flops_per_micro/1e12:.1f} TFLOP per micro-batch, "
               f"GPU peak={gpu_peak_tflops} TFLOP/s ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'})")

  # Register MFU/HFU metrics (appended per micro-step, same pipeline as loss)
  if gpu_peak_tflops > 0:
    acc = args.gradient_accumulation_steps
    lps = args.logging_per_step
    metrics.new("mfu", dtype="float")
    metrics.new("hfu", dtype="float")
    metrics.new("tflops_per_gpu", dtype="float")
    for name in ["mfu", "hfu", "tflops_per_gpu"]:
        series = getattr(metrics, name).avg(window=acc)[::acc][1:]
        metrics.logger.track(
            series.avg(window=lps)[::lps],
            name=name, group="perf")

  last_micro_time = time.time()
  
  # Setup data iterator
  if dataloader is not None:
    if args.overfit_batches:
      # Overfit debug mode: cache n batches and cycle through them
      print_rank_0(f"=== OVERFIT DEBUG MODE: Caching {args.overfit_batches} batches ===")
      print_rank_0(f"Checkpoint saving will be disabled in overfit mode")
      cached_batches = []
      temp_iter = iter(dataloader)
      for i in range(args.overfit_batches):
        try:
          batch = next(temp_iter)
          cached_batches.append(batch)
        except StopIteration:
          print_rank_0(f"Warning: Only {i} batches available, less than requested {args.overfit_batches}")
          break
      print_rank_0(f"Successfully cached {len(cached_batches)} batches for overfitting")
      print_rank_0(f"Model will cycle through these batches indefinitely")
      # Create infinite iterator that cycles through cached batches
      data_iter = iter(itertools.cycle(cached_batches))
    else:
      # Normal mode: use dataloader as-is
      data_iter = iter(dataloader)
  else:
    print_rank_0("Warning: No dataloader available. Training loop will not run.")
    data_iter = iter([])
  
  step_t0 = time.time()
  while True:
    with contextlib.ExitStack() as ctx:
      if torch_profiler:
        ctx.enter_context(torch_profiler)

      try:
        batch = next(data_iter)
      except StopIteration:
        break

      # Clone tensors to avoid mutating cached batches (needed for overfit mode)
      if args.overfit_batches:
        batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

      to_cuda(batch)

      # Advance scheduler (manages micro_step and global_step)
      scheduler.step()

      # ---- Summary token insertion (BEFORE CP split) ----
      # Labels must be created from ORIGINAL input_ids (before summary expansion).

      # Save original batch for teacher forward (distillation)
      origin_batch = None
      if args.enable_distill:
          origin_batch = {
              "input_ids": batch["input_ids"].clone(),
              "position_ids": batch.get("position_ids", torch.arange(
                  batch["input_ids"].shape[1], device=batch["input_ids"].device
              ).unsqueeze(0).expand(batch["input_ids"].shape[0], -1)).clone(),
              "cu_seqlens": batch.get("cu_seqlens", None),
          }

      # Labels from ORIGINAL input_ids + loss_mask (before any summary expansion)
      original_ids = batch["input_ids"]
      original_mask = batch["loss_mask"]
      shifted_ids = F.pad(original_ids[:, 1:], (0, 1), value=0)
      shifted_mask = F.pad(original_mask[:, 1:], (0, 1), value=0)
      batch["labels"] = shifted_ids * shifted_mask + (-100) * (1 - shifted_mask)
      batch["loss_mask"] = shifted_mask.contiguous()

      summary_ctx = None
      if hasattr(model_config, 'summary_enabled') and model_config.summary_enabled:
          # Expand input_ids + position_ids + cu_seqlens with summary tokens
          batch, summary_ctx = maybe_build_summary_batch(batch, model_config)
          summary_ctx.curr_iteration = scheduler.global_step + step_offset

      # Context parallel: broadcast batch from cp_rank=0, then split along seq dim
      if cp_size > 1:
          broadcast_batch_tensors(batch, get_context_parallel_group())

      if summary_ctx is not None:
          # Labels already created above; just do CP split on input_ids/position_ids
          if cp_size > 1:
              cp_rank = get_context_parallel_rank()
              expanded_len = batch["input_ids"].shape[1]
              chunk = expanded_len // cp_size
              start = cp_rank * chunk
              end = start + chunk
              batch["input_ids"] = batch["input_ids"][:, start:end].contiguous()
              if batch.get("position_ids") is not None:
                  batch["position_ids"] = batch["position_ids"][:, start:end].contiguous()
              # Labels: split at original length
              orig_len = batch["labels"].shape[1]
              orig_chunk = orig_len // cp_size
              orig_start = cp_rank * orig_chunk
              orig_end = orig_start + orig_chunk
              batch["labels"] = batch["labels"][:, orig_start:orig_end].contiguous()
              batch["loss_mask"] = batch["loss_mask"][:, orig_start:orig_end].contiguous()

          # CP slice summary_ctx
          if cp_size > 1:
              summary_ctx.full_summary_mask = summary_ctx.summary_mask
              new_seq_len = summary_ctx.summary_mask.shape[1]
              chunk = new_seq_len // cp_size
              start = cp_rank * chunk
              end = start + chunk
              summary_ctx.summary_mask = summary_ctx.summary_mask[:, start:end].contiguous()
              summary_ctx.position_ids = summary_ctx.position_ids[:, start:end].contiguous()
          else:
              summary_ctx.full_summary_mask = summary_ctx.summary_mask

          # CP split origin_batch for teacher forward (distillation)
          if origin_batch is not None and cp_size > 1:
              orig_len = origin_batch["input_ids"].shape[1]
              orig_chunk = orig_len // cp_size
              orig_start = cp_rank * orig_chunk
              orig_end = orig_start + orig_chunk
              origin_batch["input_ids"] = origin_batch["input_ids"][:, orig_start:orig_end].contiguous()
              origin_batch["position_ids"] = origin_batch["position_ids"][:, orig_start:orig_end].contiguous()
      else:
          # No summary: standard split (creates labels internally)
          split_batch_for_context_parallel(batch, get_context_parallel_rank(), cp_size)

      # Extract batch data
      input_ids = batch["input_ids"]
      labels = batch["labels"]
      position_ids = batch.get("position_ids", None)
      cu_seqlens = batch.get("cu_seqlens", None)

      num_tokens = input_ids.shape[1] * cp_size  # global token count
      metrics.tokens.append(num_tokens)

      num_samples = cu_seqlens.shape[0] - 1
      metrics.samples.append(num_samples)
      # ================================================ Forward pass ================================================
      # Enable attn stats collection at logging steps
      if args.log_attn_stats and scheduler.should_logging():
          model.set_log_attn_stats(True)

      student_output = model(tokens=input_ids, cu_seqlens=cu_seqlens, input_pos=position_ids, summary_ctx=summary_ctx)

      # Unpack student output: with skip_output_layer=True, output is hidden states [b, s, d]
      # May be tensor, (hidden, attn_outputs), or (hidden, attn_outputs, attn_stats_list)
      student_attn_outputs = None
      student_attn_stats = None
      if isinstance(student_output, tuple):
          if len(student_output) == 3:
              student_hidden, student_attn_outputs, student_attn_stats = student_output
          else:
              student_hidden, student_attn_outputs = student_output
      else:
          student_hidden = student_output

      # Disable attn stats after forward
      if args.log_attn_stats and scheduler.should_logging():
          model.set_log_attn_stats(False)

      grad_acc = args.gradient_accumulation_steps

      teacher_hidden = None
      attn_loss_val = 0.0

      if args.enable_distill and origin_batch is not None:
          # ---- Teacher forward (no summary, no gradient) ----
          with torch.no_grad():
              teacher_output = model(
                  tokens=origin_batch["input_ids"],
                  cu_seqlens=origin_batch.get("cu_seqlens", cu_seqlens),
                  input_pos=origin_batch["position_ids"],
                  summary_ctx=None,
              )
          teacher_attn_outputs = None
          if isinstance(teacher_output, tuple):
              if len(teacher_output) == 3:
                  teacher_hidden, teacher_attn_outputs, _ = teacher_output
              else:
                  teacher_hidden, teacher_attn_outputs = teacher_output
          else:
              teacher_hidden = teacher_output

          # ---- Attention distillation (MSE / cos, independent of logits) ----
          attn_mse = torch.stack([
              attention_mse_loss(s, t) for s, t in zip(student_attn_outputs, teacher_attn_outputs)
          ]).mean()
          attn_loss = args.distill_attn_mse_factor * attn_mse
          if args.distill_attn_cos_factor > 0:
              attn_cos = torch.stack([
                  attention_cosine_loss(s, t) for s, t in zip(student_attn_outputs, teacher_attn_outputs)
              ]).mean()
              attn_loss = attn_loss + args.distill_attn_cos_factor * attn_cos

          (attn_loss / grad_acc).backward(retain_graph=True)
          attn_loss_val = attn_loss.detach().item()

      # ---- CE + optional KL loss ----
      if chunked_loss is not None:
        loss_dict = chunked_loss.forward_and_backward(
            student_hidden, teacher_hidden, labels,
            gradient_accumulation_steps=grad_acc,
        )
        local_loss = loss_dict["total_loss"].item() + attn_loss_val
      else:
        # Standard CE path (skip_output_layer=False, student_hidden is logits)
        # Labels are already pre-shifted (done in batch preparation), no shift needed here.
        logits = student_hidden
        ce_loss = ce_loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        (ce_loss / grad_acc).backward()
        loss_dict = {"total_loss": ce_loss, "ce_loss": ce_loss, "kl_loss": torch.tensor(0.0)}
        local_loss = ce_loss.item() + attn_loss_val
      detached_loss = local_loss
      # CP all-reduce for consistent logging
      if cp_size > 1:
          t = torch.tensor(local_loss, device="cuda")
          dist.all_reduce(t, group=get_context_parallel_group())
          detached_loss = t.item() / cp_size
      metrics.loss.append(detached_loss)

      # Distillation metrics (only when distill is enabled)
      if args.enable_distill:
          metrics.distill_lm_loss.append(loss_dict["ce_loss"].item())
          metrics.distill_kl_loss.append(loss_dict["kl_loss"].item())
          metrics.distill_attn_mse.append(attn_mse.detach().item())
          metrics.distill_total_loss.append(local_loss)

      if args.monitor_datasource_loss:
        domain_name = _get_batch_domain_name(batch)
        source_name = _get_batch_source_name(batch)
        valid_tokens = int((labels != -100).sum().item())
        if valid_tokens > 0:
          ce_loss_val = loss_dict["ce_loss"].item()
          d = domain_window_stats[domain_name]
          d["loss_sum"] += ce_loss_val * valid_tokens
          d["tokens"] += valid_tokens
          s = source_window_stats[source_name]
          s["loss_sum"] += ce_loss_val * valid_tokens
          s["tokens"] += valid_tokens
      # ================================================ End of Forward + Backward ================================================

      # ================================================ Optimizer step ================================================
      # clip_grad moved to accumulation boundary (backward done inside chunked class)
      if scheduler.is_gradient_accumulation_boundary():
        clip_grad_by_value(model, args.clip_range)
        grad_norm = compute_fsdp_zero2_grad_norm(model)
        metrics.grad_norm.append(grad_norm)
        learning_rate = lr_scheduler.get_last_lr()[0]
        metrics.learning_rate.append(learning_rate)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
      # ================================================ End of Optimizer step ================================================

      # MFU / HFU per micro-step (Megatron-aligned: divide by cp_size)
      if gpu_peak_tflops > 0:
        now = time.time()
        micro_elapsed = now - last_micro_time
        last_micro_time = now
        if micro_elapsed > 0:
          tflops = model_flops_per_micro / (micro_elapsed * 1e12 * cp_size)
          metrics.tflops_per_gpu.append(tflops)
          metrics.mfu.append(tflops / gpu_peak_tflops)
          metrics.hfu.append(hw_flops_per_micro / (micro_elapsed * 1e12 * cp_size) / gpu_peak_tflops)

      # Advance metrics index for this step
      metrics.step_time.tick()
      metrics.step()

      # Logging at specified intervals
      if scheduler.should_logging():
        step_elapsed = time.time() - step_t0
        step_t0 = time.time()
        metrics.write_logs(scheduler.global_step + step_offset)
        if dist.get_rank() == 0 and tb_writer is not None:
          tb_writer.add_scalar(
            "perf/step_elapsed", step_elapsed, scheduler.global_step + step_offset)
          # Log summary mixoff coefficient
          if getattr(model_config, 'summary_mix_qkv', False):
            cur = scheduler.global_step + step_offset
            start = model_config.summary_mix_start_iter
            end = max(start + 1, model_config.summary_mix_end_iter)
            if cur <= start:
              mix = 1.0
            elif cur >= end:
              mix = 0.0
            else:
              mix = 1.0 - (cur - start) / (end - start)
            tb_writer.add_scalar("summary/mix_coeff", mix, cur)
        if args.monitor_datasource_loss and (domain_window_stats or source_window_stats):
          step = scheduler.global_step + step_offset
          # Domain-level: reduce and log (~15 domains)
          local_domain_stats = {k: dict(v) for k, v in domain_window_stats.items()}
          reduced_domain = dist_reduce_dict(local_domain_stats)
          if dist.get_rank() == 0 and tb_writer is not None:
            for domain_name in sorted(reduced_domain):
              stats = reduced_domain[domain_name]
              token_count = stats.get("tokens", 0)
              if token_count <= 0:
                continue
              avg_loss = stats["loss_sum"] / token_count
              tb_writer.add_scalar(f"domain_loss/{domain_name}", avg_loss, step)
              tb_writer.add_scalar(f"domain_tokens/{domain_name}", token_count, step)
          domain_window_stats.clear()

          # Source-level: reduce and log (per data source)
          local_source_stats = {k: dict(v) for k, v in source_window_stats.items()}
          reduced_source = dist_reduce_dict(local_source_stats)
          if dist.get_rank() == 0 and tb_writer is not None:
            for source_name in sorted(reduced_source):
              stats = reduced_source[source_name]
              token_count = stats.get("tokens", 0)
              if token_count <= 0:
                continue
              avg_loss = stats["loss_sum"] / token_count
              source_alias = _get_source_alias(source_name)
              tb_writer.add_scalar(f"datasource_loss/{source_alias}", avg_loss, step)
              tb_writer.add_scalar(f"datasource_tokens/{source_alias}", token_count, step)
          source_window_stats.clear()

        # Log attention flow stats
        if args.log_attn_stats and student_attn_stats and tb_writer is not None:
            for layer_idx, stats in enumerate(student_attn_stats):
                for key, val in stats.items():
                    tb_writer.add_scalar(
                        f"attn_flow/layer_{layer_idx}/{key}", val, scheduler.global_step + step_offset)

      # Save checkpoint at specified intervals
      if scheduler.should_save_checkpoint():
        if args.overfit_batches:
          print_rank_0(f"Skipping checkpoint save at step {scheduler.global_step + step_offset} (overfit debug mode)")
        else:
          torch.cuda.empty_cache()
          gc.collect()

          with Timer("save checkpoint"):
            save_checkpoint(
              app_state=app_state,
              dist_checkpointer=dist_checkpointer,
              checkpoint_dir=args.output_dir,
              global_step=scheduler.global_step + step_offset
            )
            save_dataloader_state(
              dataset=dataset,
              checkpoint_dir=args.output_dir,
              global_step=scheduler.global_step + step_offset
            )

      if torch_profiler:
        torch_profiler.step()

      # Stop at num_training_steps
      if scheduler.global_step >= args.num_training_steps:
        print_rank_0(f"Reached num_training_steps={args.num_training_steps}, stopping.")
        break

  # Save final checkpoint (skip if already saved at this step)
  if not args.overfit_batches and not scheduler.should_save_checkpoint():
    save_checkpoint(
      app_state=app_state,
      dist_checkpointer=dist_checkpointer,
      checkpoint_dir=args.output_dir,
      global_step=scheduler.global_step + step_offset)
    save_dataloader_state(
      dataset=dataset,
      checkpoint_dir=args.output_dir,
      global_step=scheduler.global_step + step_offset)
  elif args.overfit_batches:
    print_rank_0(f"Skipping final checkpoint save (overfit debug mode)")

if __name__ == "__main__":
  train()
