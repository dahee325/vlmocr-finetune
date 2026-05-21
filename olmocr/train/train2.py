"""
Simple script to test OlmOCR dataset loading with YAML configuration.
"""

import argparse
import json
import logging
import math
import os
import shutil
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import wandb
import torch.distributed as dist
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    get_scheduler,
)

from olmocr.train.config import Config
from olmocr.train.dataloader import BaseMarkdownPDFDataset
from olmocr.train.muon import SingleDeviceMuonWithAuxAdam

from olmocr.train.env_overrides import apply_env_overrides # 추가


# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def is_distributed_mode() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def is_main_process() -> bool:
    if not is_distributed_mode():
        return True
    return int(os.environ.get("RANK", "0")) == 0


def setup_distributed() -> tuple[torch.device, int, int]:
    if not is_distributed_mode():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device, 0, 1

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return device, rank, world_size


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def compute_best_selection_score(
    eval_metric: float,
    train_loss: float,
    greater_is_better: bool,
    eval_weight: float,
    train_weight: float,
) -> float:
    """Compute a lower-is-better score for best-checkpoint selection."""
    total_weight = eval_weight + train_weight
    if total_weight <= 0:
        eval_weight, train_weight = 1.0, 0.0
        total_weight = 1.0
    eval_weight /= total_weight
    train_weight /= total_weight

    # Keep a consistent "lower is better" convention.
    eval_cost = -float(eval_metric) if greater_is_better else float(eval_metric)
    return eval_weight * eval_cost + train_weight * float(train_loss)


def prepare_lora_model(model: torch.nn.Module, model_cfg) -> torch.nn.Module:
    """Wrap the model with a LoRA adapter according to the configuration."""
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise ImportError("LoRA training requires the `peft` package. Install it with `pip install peft`.") from exc

    lora_kwargs = dict(
        r=model_cfg.lora_rank,
        lora_alpha=model_cfg.lora_alpha,
        lora_dropout=model_cfg.lora_dropout,
        target_modules=model_cfg.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if model_cfg.lora_modules_to_save:
        lora_kwargs["modules_to_save"] = model_cfg.lora_modules_to_save

    lora_config = LoraConfig(**lora_kwargs)
    model = get_peft_model(model, lora_config)

    if hasattr(model, "config"):
        model.config.base_model_name_or_path = model_cfg.name
    base_model = getattr(model, "base_model", None)
    if base_model is not None:
        inner_model = getattr(base_model, "model", None)
        if inner_model is not None and hasattr(inner_model, "config"):
            inner_model.config._name_or_path = model_cfg.name

    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    return model


def is_lora_checkpoint(checkpoint_dir: str) -> bool:
    """Detect whether a checkpoint directory contains LoRA adapter weights."""
    return os.path.exists(os.path.join(checkpoint_dir, "adapter_config.json"))


class QwenDataCollator:
    """Data collator for vision-language models that handles numpy arrays."""

    def __init__(self, max_token_len: Optional[int] = None):
        self.max_token_len = max_token_len

    def __call__(self, examples):
        # Filter out None values and extract the fields we need
        batch = {"input_ids": [], "attention_mask": [], "labels": [], "pixel_values": [], "image_grid_thw": []}

        for example in examples:
            if example is not None:
                # Convert numpy arrays to tensors
                input_ids = torch.from_numpy(example["input_ids"]) if isinstance(example["input_ids"], np.ndarray) else example["input_ids"]
                attention_mask = torch.from_numpy(example["attention_mask"]) if isinstance(example["attention_mask"], np.ndarray) else example["attention_mask"]
                labels = torch.from_numpy(example["labels"]) if isinstance(example["labels"], np.ndarray) else example["labels"]

                # Trim to max_token_len if specified
                if self.max_token_len is not None:
                    input_ids = input_ids[: self.max_token_len]
                    attention_mask = attention_mask[: self.max_token_len]
                    labels = labels[: self.max_token_len]

                batch["input_ids"].append(input_ids)
                batch["attention_mask"].append(attention_mask)
                batch["labels"].append(labels)

                # Handle pixel_values which might be numpy array or already a tensor
                pixel_values = example["pixel_values"]
                if isinstance(pixel_values, np.ndarray):
                    pixel_values = torch.from_numpy(pixel_values)
                batch["pixel_values"].append(pixel_values)

                # Handle image_grid_thw
                image_grid_thw = example["image_grid_thw"]
                if isinstance(image_grid_thw, np.ndarray):
                    image_grid_thw = torch.from_numpy(image_grid_thw)
                batch["image_grid_thw"].append(image_grid_thw)

        # Check if we have any valid samples
        if not batch["input_ids"]:
            return None

        # Convert lists to tensors with proper padding
        # Note: For Qwen2-VL, we typically handle variable length sequences
        # The model's processor should handle the padding internally
        return {
            "input_ids": torch.stack(batch["input_ids"]),
            "attention_mask": torch.stack(batch["attention_mask"]),
            "labels": torch.stack(batch["labels"]),
            "pixel_values": torch.stack(batch["pixel_values"]),  # Stack into tensor
            "image_grid_thw": torch.stack(batch["image_grid_thw"]),
        }


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    epoch: float,
    global_step: int,
    samples_seen: int,
    best_metric: float,
    best_selection_score: Optional[float],
    output_dir: str,
    save_total_limit: Optional[int] = None,
    best_checkpoint: Optional[str] = None,
) -> str:
    """Save model, optimizer, scheduler, and training state."""
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save model
    unwrap_model(model).save_pretrained(checkpoint_dir)

    # Save optimizer and scheduler
    torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
    torch.save(lr_scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))

    # Save training state
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "samples_seen": samples_seen,
        "best_metric": best_metric,
        "best_selection_score": best_selection_score,
        "best_checkpoint": best_checkpoint,
    }
    torch.save(state, os.path.join(checkpoint_dir, "training_state.pt"))

    logger.info(f"Saved checkpoint to {checkpoint_dir}")

    return checkpoint_dir


def load_checkpoint(
    model_class: type,
    init_kwargs: Dict[str, Any],
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    checkpoint_dir: str,
    device: torch.device,
    *,
    base_model_path: Optional[str] = None,
    use_lora: bool = False,
) -> tuple[torch.nn.Module, Dict[str, Any]]:
    """Load model, optimizer, scheduler, and training state from checkpoint."""
    checkpoint_has_lora = is_lora_checkpoint(checkpoint_dir)

    if checkpoint_has_lora or use_lora:
        if base_model_path is None:
            raise ValueError("base_model_path must be provided when loading LoRA checkpoints.")

        try:
            from peft import PeftModel
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise ImportError("Loading a LoRA checkpoint requires the `peft` package. Install it with `pip install peft`.") from exc

        base_model = model_class.from_pretrained(base_model_path, **init_kwargs)
        model = PeftModel.from_pretrained(base_model, checkpoint_dir, is_trainable=True)
        if hasattr(model, "config"):
            model.config.base_model_name_or_path = base_model_path
    else:
        model = model_class.from_pretrained(checkpoint_dir, **init_kwargs)

    model.to(device)

    optimizer.load_state_dict(torch.load(os.path.join(checkpoint_dir, "optimizer.pt"), map_location=device))
    lr_scheduler.load_state_dict(torch.load(os.path.join(checkpoint_dir, "scheduler.pt"), map_location=device))

    state = torch.load(os.path.join(checkpoint_dir, "training_state.pt"), map_location=device)
    logger.info(f"Resumed from checkpoint: {checkpoint_dir} at epoch {state['epoch']:.2f}, step {state['global_step']}, samples seen {state['samples_seen']}")
    return model, state


def load_model_from_checkpoint(
    model_class: type,
    init_kwargs: Dict[str, Any],
    checkpoint_dir: str,
    device: torch.device,
    *,
    base_model_path: Optional[str] = None,
    use_lora: bool = False,
) -> torch.nn.Module:
    """Load model weights only (no optimizer/scheduler state)."""
    checkpoint_has_lora = is_lora_checkpoint(checkpoint_dir)

    if checkpoint_has_lora or use_lora:
        if base_model_path is None:
            raise ValueError("base_model_path must be provided when loading LoRA checkpoints.")

        try:
            from peft import PeftModel
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise ImportError("Loading a LoRA checkpoint requires the `peft` package. Install it with `pip install peft`.") from exc

        base_model = model_class.from_pretrained(base_model_path, **init_kwargs)
        model = PeftModel.from_pretrained(base_model, checkpoint_dir, is_trainable=True)
        if hasattr(model, "config"):
            model.config.base_model_name_or_path = base_model_path
    else:
        model = model_class.from_pretrained(checkpoint_dir, **init_kwargs)

    model.to(device)
    return model


def _history_file_path(output_dir: str) -> str:
    return os.path.join(output_dir, "best_checkpoint_history.jsonl")


def _ranking_file_path(output_dir: str) -> str:
    return os.path.join(output_dir, "best_checkpoint_ranking.txt")


def load_best_checkpoint_history(output_dir: str) -> List[Dict[str, Any]]:
    """Load best checkpoint history if it exists."""
    history_path = _history_file_path(output_dir)
    if not os.path.exists(history_path):
        return []

    history = []
    with open(history_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            history.append(json.loads(line))
    return history


def save_best_checkpoint_history(output_dir: str, history: List[Dict[str, Any]]) -> None:
    """Persist best checkpoint history in JSONL format."""
    history_path = _history_file_path(output_dir)
    with open(history_path, "w", encoding="utf-8") as f:
        for record in history:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")


def recompute_combined_scores(history: List[Dict[str, Any]]) -> None:
    """Recompute combined scores using eval + train losses.

    Score definition (smaller is better):
    - eval_weight = 0.8
    - train_weight = 0.2
    - each term uses min-max normalization across saved best checkpoints
    """
    if not history:
        return

    eval_vals = [float(r.get("eval_metric", 0.0)) for r in history]
    train_vals = [float(r.get("train_loss", 0.0)) for r in history]

    eval_min, eval_max = min(eval_vals), max(eval_vals)
    train_min, train_max = min(train_vals), max(train_vals)

    def minmax(value: float, low: float, high: float) -> float:
        if high <= low:
            return 0.0
        return (value - low) / (high - low)

    for r in history:
        eval_norm = minmax(float(r.get("eval_metric", 0.0)), eval_min, eval_max)
        train_norm = minmax(float(r.get("train_loss", 0.0)), train_min, train_max)
        r["combined_score"] = 0.8 * eval_norm + 0.2 * train_norm


def prune_best_checkpoints(
    output_dir: str,
    history: List[Dict[str, Any]],
    save_total_limit: Optional[int],
    protected_checkpoint: Optional[str],
) -> List[Dict[str, Any]]:
    """Prune checkpoints when count exceeds save_total_limit using combined score."""
    if save_total_limit is None or save_total_limit <= 0:
        return history

    # Keep history aligned with existing checkpoint directories.
    existing = {
        os.path.join(output_dir, d)
        for d in os.listdir(output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
    }
    history = [r for r in history if r.get("checkpoint") in existing]

    while len(history) > save_total_limit:
        recompute_combined_scores(history)

        candidates = [r for r in history if r.get("checkpoint") != protected_checkpoint]
        if not candidates:
            logger.warning("All remaining checkpoints are protected; skipping pruning.")
            break

        worst = max(candidates, key=lambda r: (float(r.get("combined_score", 0.0)), float(r.get("eval_metric", 0.0)), int(r.get("step", 0))))
        worst_path = worst["checkpoint"]
        worst_name = os.path.basename(worst_path)
        if os.path.exists(worst_path):
            shutil.rmtree(worst_path)
            logger.info(
                f"Deleted checkpoint by combined score: {worst_name} "
                f"(eval={float(worst.get('eval_metric', 0.0)):.6f}, "
                f"train={float(worst.get('train_loss', 0.0)):.6f}, "
                f"combined={float(worst.get('combined_score', 0.0)):.6f})"
            )
        history = [r for r in history if r.get("checkpoint") != worst_path]

    recompute_combined_scores(history)
    return history


def write_best_checkpoint_ranking(output_dir: str, history: List[Dict[str, Any]], metric_name: str) -> None:
    """Write ranked checkpoint summary at training end."""
    ranking_path = _ranking_file_path(output_dir)
    if not history:
        with open(ranking_path, "w", encoding="utf-8") as f:
            f.write("No best-checkpoint records were collected.\n")
        return

    recompute_combined_scores(history)
    ranked = sorted(history, key=lambda r: (float(r.get("combined_score", 0.0)), float(r.get("eval_metric", 0.0)), int(r.get("step", 0))))

    with open(ranking_path, "w", encoding="utf-8") as f:
        f.write("Best checkpoint ranking (lower is better)\n")
        f.write(f"Metric key: {metric_name}\n")
        f.write("Combined score = 0.8 * normalized_eval + 0.2 * normalized_train\n")
        f.write("=" * 88 + "\n")
        f.write("rank | step | epoch    | eval_metric | train_loss | combined_score | checkpoint\n")
        f.write("-" * 88 + "\n")
        for idx, r in enumerate(ranked, start=1):
            f.write(
                f"{idx:>4} | "
                f"{int(r.get('step', 0)):>4} | "
                f"{float(r.get('epoch', 0.0)):>8.3f} | "
                f"{float(r.get('eval_metric', 0.0)):>11.6f} | "
                f"{float(r.get('train_loss', 0.0)):>10.6f} | "
                f"{float(r.get('combined_score', 0.0)):>14.6f} | "
                f"{r.get('checkpoint', '')}\n"
            )


def resolve_best_checkpoint_from_history(history: List[Dict[str, Any]], greater_is_better: bool) -> Optional[str]:
    """Resolve global best checkpoint path from history records."""
    if not history:
        return None
    if greater_is_better:
        best_record = max(history, key=lambda r: float(r.get("eval_metric", -float("inf"))))
    else:
        best_record = min(history, key=lambda r: float(r.get("eval_metric", float("inf"))))
    return best_record.get("checkpoint")


def evaluate_model(
    model: torch.nn.Module,
    eval_dataloaders: Dict[str, DataLoader],
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate on all eval datasets and return average loss per dataset."""
    model.eval()
    eval_metrics = {}

    for dataset_name, dataloader in eval_dataloaders.items():
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                # Skip if batch is None (all samples were filtered out)
                if batch is None:
                    continue
                batch = {k: v.to(device) for k, v in batch.items()}
                with autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                    outputs = model(**batch)
                total_loss += outputs.loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        eval_metrics[f"eval_{dataset_name}_loss"] = avg_loss
        logger.info(f"Eval {dataset_name} loss: {avg_loss:.4f}")

    # Compute overall eval loss as average across datasets (or customize as needed)
    if eval_metrics:
        overall_loss = sum(eval_metrics.values()) / len(eval_metrics)
        eval_metrics["eval_loss"] = overall_loss

    return eval_metrics


def create_train_dataloader(
    train_dataset,
    config,
    data_collator,
    seed_worker,
    epoch_num: int = 0,
    distributed_sampler: Optional[DistributedSampler] = None,
) -> DataLoader:
    """Create a training dataloader with epoch-specific shuffling.

    Args:
        train_dataset: The training dataset
        config: Training configuration
        data_collator: Data collator for batching
        seed_worker: Worker initialization function
        epoch_num: Current epoch number for seed generation

    Returns:
        DataLoader with epoch-specific shuffling
    """
    # Create generator with epoch-specific seed for different shuffling each epoch
    epoch_generator = torch.Generator()
    if config.training.data_seed is not None:
        # Use epoch number to ensure different shuffling each epoch while maintaining reproducibility
        epoch_generator.manual_seed(config.training.data_seed + epoch_num)
    else:
        # Use a random seed if no data_seed specified
        epoch_generator.manual_seed(int(torch.randint(0, 2**32 - 1, (1,)).item()))

    return DataLoader(
        train_dataset,
        batch_size=config.training.per_device_train_batch_size,
        shuffle=distributed_sampler is None,
        sampler=distributed_sampler,
        collate_fn=data_collator,
        num_workers=config.training.dataloader_num_workers,
        drop_last=config.training.dataloader_drop_last,
        worker_init_fn=seed_worker,
        generator=epoch_generator,
    )


def main():
    parser = argparse.ArgumentParser(description="Train OlmOCR model")
    parser.add_argument("--config", type=str, default="olmocr/train/configs/example_config.yaml", help="Path to YAML configuration file")

    # 추가
    parser.add_argument(
        "--env-file",
        type=str,
        default=".env.train.qwen35_9b",
        help="Path to env override file",
    )

    args = parser.parse_args()

    # Distributed setup must run before device/model init.
    device, rank, world_size = setup_distributed()

    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    config = Config.from_yaml(args.config)

    # 추가
    # Apply .env overrides
    logger.info("Applying env overrides")
    config = apply_env_overrides(config)

    # 임시 로그 추가
    logger.info(f"[ENV OVERRIDE] model.name = {config.model.name}")
    logger.info(f"[ENV OVERRIDE] torch_dtype = {config.model.torch_dtype}")
    logger.info(f"[ENV OVERRIDE] use_lora = {config.model.use_lora}")
    logger.info(f"[ENV OVERRIDE] lora_rank = {config.model.lora_rank}")
    logger.info(f"[ENV OVERRIDE] learning_rate = {config.training.learning_rate}")
    logger.info(f"[ENV OVERRIDE] gradient_accumulation_steps = {config.training.gradient_accumulation_steps}")

    # 성공 출력 결과
    # [ENV OVERRIDE] model.name = Qwen/Qwen3.5-VL-9B
    # [ENV OVERRIDE] lora_rank = 8
    # [ENV OVERRIDE] learning_rate = 5e-06

    # Validate configuration
    try:
        config.validate()
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        return

    # Set wandb project from config
    if config.project_name and is_main_process():
        os.environ["WANDB_PROJECT"] = config.project_name
        logger.info(f"Setting WANDB_PROJECT to: {config.project_name}")

    # Initialize wandb if reporting to it
    if "wandb" in config.training.report_to and is_main_process():
        wandb.init(project=config.project_name, name=config.run_name, config=config.to_dict())

    # Load processor for tokenization
    logger.info(f"Loading processor: {config.model.name}")
    processor = AutoProcessor.from_pretrained(
        config.model.name,
        trust_remote_code=config.model.trust_remote_code, # 추가
    )

    # Model init kwargs to reuse for loading checkpoints
    model_init_kwargs = {
        "torch_dtype": getattr(torch, config.model.torch_dtype) if config.model.torch_dtype != "auto" else "auto",
        "device_map": None if is_distributed_mode() else config.model.device_map,
        "trust_remote_code": config.model.trust_remote_code,
        "attn_implementation": config.model.attn_implementation if config.model.use_flash_attention else None,
    }

    # Load model
    logger.info(f"Loading model: {config.model.name}")
    if (
        "qwen2.5-vl" in config.model.name.lower()
        or "olmocr-2-7b-1025" in config.model.name.lower()
        or "qwen3.5-9b" in config.model.name.lower()
        or "qwen3" in config.model.name.lower()
        or "qwen3.5-vl-9b" in config.model.name.lower() # 추가
    ):
        model_class = Qwen2_5_VLForConditionalGeneration
        model = model_class.from_pretrained(config.model.name, **model_init_kwargs)
    elif "qwen2-vl" in config.model.name.lower():
        model_class = Qwen2VLForConditionalGeneration
        model = model_class.from_pretrained(config.model.name, **model_init_kwargs)
    else:
        raise NotImplementedError()

    # 추가
    if getattr(config.training, "gradient_checkpointing", False):
        logger.info("Enabling gradient checkpointing")
        model.config.use_cache = False

        gradient_checkpointing_kwargs = getattr(
            config.training,
            "gradient_checkpointing_kwargs",
            None,
        )
        
        if gradient_checkpointing_kwargs:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )
        else:
            model.gradient_checkpointing_enable()


    if config.model.use_lora:
        logger.info("Applying LoRA adapters as specified in the config.")
        model = prepare_lora_model(model, config.model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_ratio = (trainable_params / total_params * 100) if total_params else 0.0
    if is_main_process():
        logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_ratio:.2f}%)")

    # Enable gradient checkpointing if configured 삭제처리
    # if config.training.gradient_checkpointing:
    #     model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=config.training.gradient_checkpointing_kwargs)

    # Create training datasets
    logger.info("Creating training datasets...")
    train_datasets = []
    for i, dataset_cfg in enumerate(config.dataset.train):
        root_dir = dataset_cfg["root_dir"]
        pipeline_steps = config.get_pipeline_steps(dataset_cfg["pipeline"], processor)

        logger.info(f"Creating training dataset {i+1} from: {root_dir}")
        dataset = BaseMarkdownPDFDataset(root_dir, pipeline_steps)
        logger.info(f"Found {len(dataset)} samples")

        if len(dataset) > 0:
            train_datasets.append(dataset)

    # Combine all training datasets
    train_dataset = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    logger.info(f"Total training samples: {len(train_dataset)}")

    # Create evaluation datasets
    logger.info("Creating evaluation datasets...")
    eval_datasets = {}
    for i, dataset_cfg in enumerate(config.dataset.eval):
        root_dir = dataset_cfg["root_dir"]
        pipeline_steps = config.get_pipeline_steps(dataset_cfg["pipeline"], processor)

        # Use dataset name if provided, otherwise use root_dir as name
        dataset_name = dataset_cfg.get("name", f"eval_dataset_{i+1}")

        logger.info(f"Creating evaluation dataset '{dataset_name}' from: {root_dir}")
        dataset = BaseMarkdownPDFDataset(root_dir, pipeline_steps)
        logger.info(f"Found {len(dataset)} samples")

        if len(dataset) > 0:
            eval_datasets[dataset_name] = dataset

    # Log total evaluation samples across all datasets
    total_eval_samples = sum(len(dataset) for dataset in eval_datasets.values())
    logger.info(f"Total evaluation samples across {len(eval_datasets)} datasets: {total_eval_samples}")

    # Construct full output directory by appending run_name to base output_dir
    full_output_dir = os.path.join(config.training.output_dir, config.run_name)
    logger.info(f"Setting output directory to: {full_output_dir}")
    os.makedirs(full_output_dir, exist_ok=True)

    # Check for existing checkpoints if any
    found_resumable_checkpoint = None
    if os.path.exists(full_output_dir):
        # Look for checkpoint directories
        checkpoint_dirs = [d for d in os.listdir(full_output_dir) if d.startswith("checkpoint-") and os.path.isdir(os.path.join(full_output_dir, d))]
        if checkpoint_dirs:
            # Sort by checkpoint number and get the latest
            checkpoint_dirs.sort(key=lambda x: int(x.split("-")[1]))
            latest_checkpoint = os.path.join(full_output_dir, checkpoint_dirs[-1])
            logger.info(f"Found existing checkpoint: {latest_checkpoint}")
            found_resumable_checkpoint = latest_checkpoint
        else:
            logger.info("No existing checkpoints found in output directory")

    # Set seeds
    torch.manual_seed(config.training.seed)

    # Set up data loader seed worker function
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        import random

        random.seed(worker_seed)

    model.to(device)

    if is_distributed_mode():
        model = DDP(model, device_ids=[device.index], output_device=device.index, find_unused_parameters=False)

    # Apply torch compile if enabled
    if config.training.torch_compile:
        logger.info(f"Compiling model with torch.compile (backend={config.training.torch_compile_backend}, mode={config.training.torch_compile_mode})")
        model = torch.compile(
            model,
            backend=config.training.torch_compile_backend,
            mode=config.training.torch_compile_mode,
            fullgraph=config.training.torch_compile_fullgraph,
            dynamic=config.training.torch_compile_dynamic,
        )
        logger.info("Model compilation complete")

    # Set up optimizer
    trainable_named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not trainable_named_params:
        raise ValueError("No trainable parameters found. Check model fine-tuning configuration.")

    if config.training.optim == "adamw_torch":
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in trainable_named_params if not any(nd in n for nd in no_decay)],
                "weight_decay": config.training.weight_decay,
            },
            {
                "params": [p for n, p in trainable_named_params if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=float(config.training.learning_rate),
            betas=(config.training.adam_beta1, config.training.adam_beta2),
            eps=float(config.training.adam_epsilon),
        )
    elif config.training.optim == "muon":
        if config.model.use_lora:
            raise NotImplementedError("LoRA training is not currently supported with the Muon optimizer in this loop.")

        # Separate parameters for Muon (hidden matrices) and Adam (embeddings, scalars, head)
        hidden_matrix_params = [p for n, p in trainable_named_params if p.ndim >= 2 and "embed" not in n and "lm_head" not in n]
        embed_params = [p for n, p in trainable_named_params if "embed" in n]
        scalar_params = [p for n, p in trainable_named_params if p.ndim < 2]
        head_params = [p for n, p in trainable_named_params if "lm_head" in n]

        # Create Adam groups with different learning rates
        adam_groups = [
            dict(params=head_params, lr=float(config.training.learning_rate) * config.training.muon_lr_multiplier_head, use_muon=False),
            dict(params=embed_params, lr=float(config.training.learning_rate) * config.training.muon_lr_multiplier_embed, use_muon=False),
            dict(params=scalar_params, lr=float(config.training.learning_rate) * config.training.muon_lr_multiplier_scalar, use_muon=False),
        ]

        # Add Adam hyperparameters to groups
        for g in adam_groups:
            g["betas"] = (config.training.adam_beta1, config.training.adam_beta2)
            g["eps"] = float(config.training.adam_epsilon)
            g["weight_decay"] = config.training.weight_decay

        # Create Muon group
        muon_group = dict(
            params=hidden_matrix_params,
            lr=float(config.training.learning_rate),
            momentum=config.training.muon_momentum,
            weight_decay=config.training.weight_decay,
            use_muon=True,
        )

        # Combine all groups
        param_groups = [*adam_groups, muon_group]
        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
    else:
        raise NotImplementedError(f"Optimizer {config.training.optim} not supported in custom loop")

    # Total training steps calculation
    samples_per_microbatch = config.training.per_device_train_batch_size * world_size
    samples_per_step = samples_per_microbatch * config.training.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_dataset) / samples_per_step)
    max_train_steps = int(math.ceil(config.training.num_train_epochs * num_update_steps_per_epoch))
    max_train_samples = int(math.ceil(config.training.num_train_epochs * len(train_dataset)))

    # Set up scheduler
    lr_scheduler = get_scheduler(
        name=config.training.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(max_train_steps * config.training.warmup_ratio),
        num_training_steps=max_train_steps,
        scheduler_specific_kwargs=config.training.lr_scheduler_kwargs,
    )

    # Data collator
    data_collator = QwenDataCollator(max_token_len=config.training.collator_max_token_len)

    # Resume from checkpoint if available
    global_step = 0
    samples_seen = 0
    best_metric = float("inf") if not config.training.greater_is_better else -float("inf")
    best_selection_score = float("inf")
    best_checkpoint = None
    best_checkpoint_history = load_best_checkpoint_history(full_output_dir)
    eval_metric_weight = float(config.training.best_model_eval_metric_weight)
    train_loss_weight = float(config.training.best_model_train_loss_weight)
    if is_main_process():
        logger.info(
            "Best-checkpoint score weights: eval=%.3f, train=%.3f",
            eval_metric_weight,
            train_loss_weight,
        )

    if found_resumable_checkpoint:
        model, state = load_checkpoint(
            model_class,
            model_init_kwargs,
            optimizer,
            lr_scheduler,
            found_resumable_checkpoint,
            device,
            base_model_path=config.model.name,
            use_lora=config.model.use_lora,
        )
        global_step = state["global_step"]
        best_metric = state["best_metric"]
        if "best_selection_score" in state:
            best_selection_score = float(state["best_selection_score"])
        else:
            # Backward compatibility for older checkpoints that only stored best_metric.
            best_selection_score = compute_best_selection_score(
                eval_metric=float(best_metric),
                train_loss=0.0,
                greater_is_better=config.training.greater_is_better,
                eval_weight=eval_metric_weight,
                train_weight=train_loss_weight,
            )
        samples_seen = state["samples_seen"]
        best_checkpoint = state.get("best_checkpoint")
        resolved_best = resolve_best_checkpoint_from_history(best_checkpoint_history, config.training.greater_is_better)
        if resolved_best is not None:
            best_checkpoint = resolved_best
            matched = next((r for r in best_checkpoint_history if r.get("checkpoint") == resolved_best), None)
            if matched is not None:
                best_selection_score = compute_best_selection_score(
                    eval_metric=float(matched.get("eval_metric", best_metric)),
                    train_loss=float(matched.get("train_loss", 0.0)),
                    greater_is_better=config.training.greater_is_better,
                    eval_weight=eval_metric_weight,
                    train_weight=train_loss_weight,
                )

    train_sampler: Optional[DistributedSampler] = None
    if is_distributed_mode():
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=config.training.data_seed or config.training.seed,
            drop_last=config.training.dataloader_drop_last,
        )

    # Create dataloaders - use epoch 0 initially (will be recreated with proper epoch if resuming)
    current_epoch_num = int(samples_seen / len(train_dataset)) if samples_seen > 0 else 0
    if train_sampler is not None:
        train_sampler.set_epoch(current_epoch_num)
    train_dataloader = create_train_dataloader(
        train_dataset,
        config,
        data_collator,
        seed_worker,
        epoch_num=current_epoch_num,
        distributed_sampler=train_sampler,
    )

    eval_dataloaders = {
        name: DataLoader(
            dataset,
            batch_size=config.training.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=data_collator,
            num_workers=config.training.dataloader_num_workers,
            drop_last=False,
        )
        for name, dataset in eval_datasets.items()
    }

    # Always evaluate on start (rank 0 only in DDP)
    if is_main_process():
        metrics = evaluate_model(model, eval_dataloaders, device)
        logger.info(f"Initial evaluation: {metrics}")
        if "wandb" in config.training.report_to:
            wandb.log(metrics, step=global_step)
    if is_distributed_mode():
        dist.barrier()

    # Main training loop
    current_epoch = samples_seen / len(train_dataset)
    if is_main_process():
        logger.info(f"Starting training from epoch {current_epoch:.2f} (step {global_step}, samples {samples_seen}) to {config.training.num_train_epochs} epochs")
        logger.info(f"Total training steps: {max_train_steps}, Total samples to process: {max_train_samples}")

    if samples_seen >= max_train_samples:
        logger.info("Training already completed based on samples seen!")
        logger.info("Skipping to final model save.")
    else:
        model.train()
        accumulated_loss = 0.0
        num_losses_accumulated = 0
        latest_train_loss_for_checkpoint = 0.0

        # Create epoch iterator and skip samples if resuming
        epoch_iterator = iter(train_dataloader)
        if samples_seen > 0:
            samples_to_skip = samples_seen % len(train_dataset)
            batches_to_skip = samples_to_skip // samples_per_microbatch
            logger.info(f"Resuming training: skipping {batches_to_skip} batches ({samples_to_skip} samples) to reach position {samples_seen}")

            # Skip batches to resume from the correct position within the epoch
            for _ in range(batches_to_skip):
                try:
                    next(epoch_iterator)
                except StopIteration:
                    # We've reached the end of the epoch while skipping
                    # This shouldn't normally happen, but handle it gracefully
                    logger.warning(f"Reached end of epoch while skipping batches. Creating new epoch.")
                    current_epoch_num += 1
                    train_dataloader = create_train_dataloader(
                        train_dataset,
                        config,
                        data_collator,
                        seed_worker,
                        epoch_num=current_epoch_num,
                    )
                    epoch_iterator = iter(train_dataloader)
                    break

        # Create progress bar
        pbar = tqdm(total=max_train_samples - samples_seen, desc=f"Training from step {global_step}", unit="samples") if is_main_process() else None

        while samples_seen < max_train_samples and global_step < max_train_steps:
            try:
                batch = next(epoch_iterator)
            except StopIteration:
                # End of epoch, create new dataloader with fresh shuffle
                current_epoch = samples_seen / len(train_dataset)
                if is_main_process():
                    logger.info(f"Completed epoch {current_epoch:.2f}")

                # Increment epoch number for new shuffle seed
                current_epoch_num += 1

                if train_sampler is not None:
                    train_sampler.set_epoch(current_epoch_num)
                # Recreate dataloader with new generator for fresh shuffle
                train_dataloader = create_train_dataloader(
                    train_dataset,
                    config,
                    data_collator,
                    seed_worker,
                    epoch_num=current_epoch_num,
                    distributed_sampler=train_sampler,
                )
                epoch_iterator = iter(train_dataloader)
                batch = next(epoch_iterator)

            # Skip if batch is None (all samples were filtered out)
            if batch is None:
                continue

            batch = {k: v.to(device) for k, v in batch.items()}

            with autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                outputs = model(**batch)
            loss = outputs.loss / config.training.gradient_accumulation_steps
            loss.backward()

            accumulated_loss += outputs.loss.item()  # Use undivided loss for logging
            num_losses_accumulated += 1
            samples_seen += samples_per_microbatch

            # Update progress bar
            if pbar is not None:
                pbar.update(samples_per_microbatch)

            # Check if we should do a gradient update
            if samples_seen % samples_per_step == 0 or samples_seen >= max_train_samples:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

                # Step optimizer and scheduler
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                current_epoch = samples_seen / len(train_dataset)

                # Update progress bar with current stats
                current_lr = lr_scheduler.get_last_lr()[0]
                avg_loss = accumulated_loss / num_losses_accumulated if num_losses_accumulated > 0 else 0
                latest_train_loss_for_checkpoint = avg_loss
                if pbar is not None:
                    pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{current_lr:.2e}", "epoch": f"{current_epoch:.2f}", "step": global_step})

                # Logging
                if config.training.logging_steps > 0 and global_step % config.training.logging_steps == 0:
                    avg_train_loss = accumulated_loss / num_losses_accumulated if num_losses_accumulated > 0 else 0
                    logs = {
                        "train_loss": avg_train_loss,
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "epoch": current_epoch,
                        "samples_seen": samples_seen,
                    }
                    if is_main_process():
                        logger.info(f"Step {global_step}: epoch={current_epoch:.3f}, loss={avg_train_loss:.4f}, lr={lr_scheduler.get_last_lr()[0]:.2e}")
                    if "wandb" in config.training.report_to and is_main_process():
                        wandb.log(logs, step=global_step)

                    accumulated_loss = 0.0
                    num_losses_accumulated = 0

                # Evaluation
                metric_improved = False
                if config.training.eval_steps > 0 and global_step % config.training.eval_steps == 0 and global_step > 0 and is_main_process():
                    metrics = evaluate_model(model, eval_dataloaders, device)
                    logger.info(f"Evaluation at step {global_step}: {metrics}")
                    if "wandb" in config.training.report_to:
                        wandb.log(metrics, step=global_step)

                    # Update best metric
                    current_metric = metrics.get(config.training.metric_for_best_model, None)
                    if current_metric is not None:
                        current_selection_score = compute_best_selection_score(
                            eval_metric=float(current_metric),
                            train_loss=float(latest_train_loss_for_checkpoint),
                            greater_is_better=config.training.greater_is_better,
                            eval_weight=eval_metric_weight,
                            train_weight=train_loss_weight,
                        )
                        if current_selection_score < best_selection_score:
                            best_metric = current_metric
                            best_selection_score = current_selection_score
                            metric_improved = True
                            logger.info(
                                "New best checkpoint at step %s: %s=%.6f, train_loss=%.6f, score=%.6f",
                                global_step,
                                config.training.metric_for_best_model,
                                float(best_metric),
                                float(latest_train_loss_for_checkpoint),
                                float(best_selection_score),
                            )
                    else:
                        logger.warning(
                            f"Metric '{config.training.metric_for_best_model}' not found in evaluation metrics: {list(metrics.keys())}"
                        )

                    # Return to training mode
                    model.train()

                # Saving
                if metric_improved and is_main_process():
                    just_saved_checkpoint = save_checkpoint(
                        model,
                        optimizer,
                        lr_scheduler,
                        current_epoch,
                        global_step,
                        samples_seen,
                        best_metric,
                        best_selection_score,
                        full_output_dir,
                        None,
                        best_checkpoint,
                    )
                    best_checkpoint = just_saved_checkpoint
                    logger.info(f"Updated best checkpoint to {best_checkpoint}")

                    # Ensure resumability: the just-saved checkpoint should also point to itself as best.
                    training_state_path = os.path.join(just_saved_checkpoint, "training_state.pt")
                    if os.path.exists(training_state_path):
                        checkpoint_state = torch.load(training_state_path, map_location="cpu")
                        checkpoint_state["best_checkpoint"] = just_saved_checkpoint
                        torch.save(checkpoint_state, training_state_path)

                    best_checkpoint_history.append(
                        {
                            "step": global_step,
                            "epoch": current_epoch,
                            "checkpoint": just_saved_checkpoint,
                            "eval_metric": float(best_metric),
                            "train_loss": float(latest_train_loss_for_checkpoint),
                            "combined_score": 0.0,
                        }
                    )
                    best_checkpoint_history = prune_best_checkpoints(
                        full_output_dir,
                        best_checkpoint_history,
                        config.training.save_total_limit,
                        best_checkpoint,
                    )
                    save_best_checkpoint_history(full_output_dir, best_checkpoint_history)
                if is_distributed_mode():
                    dist.barrier()

            # Check if we've reached our training limit
            if samples_seen >= max_train_samples or global_step >= max_train_steps:
                break

        # Close progress bar
        if pbar is not None:
            pbar.close()

    # Fallback: if no best checkpoint was ever saved, save current model once.
    if best_checkpoint is None and is_main_process():
        logger.warning("No best checkpoint was recorded during training. Saving current model as fallback checkpoint.")
        fallback_checkpoint = save_checkpoint(
            model,
            optimizer,
            lr_scheduler,
            current_epoch,
            global_step,
            samples_seen,
            best_metric,
            best_selection_score,
            full_output_dir,
            None,
            None,
        )
        best_checkpoint = fallback_checkpoint
        best_checkpoint_history.append(
            {
                "step": global_step,
                "epoch": current_epoch,
                "checkpoint": fallback_checkpoint,
                "eval_metric": float(best_metric) if np.isfinite(best_metric) else float("inf"),
                "train_loss": 0.0,
                "combined_score": 0.0,
            }
        )
        best_checkpoint_history = prune_best_checkpoints(
            full_output_dir,
            best_checkpoint_history,
            config.training.save_total_limit,
            best_checkpoint,
        )
        save_best_checkpoint_history(full_output_dir, best_checkpoint_history)

    if config.training.load_best_model_at_end and is_main_process():
        if best_checkpoint and os.path.exists(best_checkpoint):
            logger.info(f"Loading best model from checkpoint: {best_checkpoint}")
            model = load_model_from_checkpoint(
                model_class,
                model_init_kwargs,
                best_checkpoint,
                device,
                base_model_path=config.model.name,
                use_lora=config.model.use_lora,
            )
        else:
            logger.warning("load_best_model_at_end is enabled, but no best checkpoint was found. Keeping current model.")

    # Log final training state
    if is_main_process():
        final_epoch = samples_seen / len(train_dataset)
        logger.info(f"Training completed at epoch {final_epoch:.3f}, step {global_step}, samples {samples_seen}")

        # Final evaluation
        final_metrics = evaluate_model(model, eval_dataloaders, device)
        logger.info(f"Final evaluation metrics: {final_metrics}")
        write_best_checkpoint_ranking(full_output_dir, best_checkpoint_history, config.training.metric_for_best_model)
        logger.info(f"Saved best-checkpoint ranking to {_ranking_file_path(full_output_dir)}")
        if "wandb" in config.training.report_to:
            wandb.log(final_metrics, step=global_step)
            wandb.finish()

    if is_distributed_mode():
        dist.barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()