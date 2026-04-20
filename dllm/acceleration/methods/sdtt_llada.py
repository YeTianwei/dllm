"""
Run:
    CUDA_VISIBLE_DEVICES=3 python examples/benchmarks/train_sdtt_llada.py \
        --teacher_model_name_or_path .models/smoke_test_llada_sft/checkpoint-final \
        --student_model_name_or_path .models/smoke_test_llada_sft/checkpoint-final \
        --output_dir .models/sdtt-llada-smoke
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import transformers
from huggingface_hub import try_to_load_from_cache
from peft import AutoPeftModel

import dllm
from dllm.acceleration.base import AccelerationMethod
from dllm.acceleration.benchmark import BenchmarkConfig
from dllm.core.samplers import MDLMSampler
from dllm.core.samplers.utils import get_num_transfer_tokens
from dllm.core.schedulers import BaseAlphaScheduler, LinearAlphaScheduler


SDTT_CONFIG_NAME = "sdtt_config.json"
SDTT_BENCHMARK_STEPS = 24
SDTT_BENCHMARK_BLOCK_SIZE = 24


@dataclass
class SDTTMethodMetadata:
    teacher_model_name_or_path: str
    student_model_name_or_path: str
    teacher_steps: int
    student_steps: int
    block_size: int
    temperature: float = 1.0
    method_name: str = "sdtt_llada"


@dataclass
class SDTTDistillationArguments(dllm.utils.TrainingArguments):
    output_dir: str = ".models/sdtt-llada"
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    num_train_epochs: float = 1.0
    learning_rate: float = 2e-5
    logging_steps: float = 1
    save_steps: float = 1000
    eval_strategy: str = "no"
    report_to: str = "none"
    bf16: bool = True
    remove_unused_columns: bool = False
    teacher_steps: int = 64
    student_steps: int = 8
    distill_temperature: float = 1.0
    time_epsilon: float = 1e-3
    loss_weight_type: str = "scheduler"
    block_size: int = 64


def _load_sdtt_metadata(path: str) -> SDTTMethodMetadata | None:
    config_path = Path(path) / SDTT_CONFIG_NAME
    if not config_path.exists():
        return None
    return SDTTMethodMetadata(**json.loads(config_path.read_text(encoding="utf-8")))


def save_sdtt_metadata(path: str, metadata: SDTTMethodMetadata) -> None:
    output_path = Path(path) / SDTT_CONFIG_NAME
    output_path.write_text(
        json.dumps(metadata.__dict__, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _make_quant_config(load_in_4bit: bool, dtype: torch.dtype):
    if not load_in_4bit:
        return None
    return transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def _resolve_local_files_only(model_name_or_path: str) -> bool:
    if os.path.isdir(model_name_or_path):
        return True

    cached_config = try_to_load_from_cache(
        repo_id=model_name_or_path,
        filename="config.json",
        repo_type="model",
    )
    return isinstance(cached_config, str) and os.path.exists(cached_config)


def load_llada_checkpoint(
    model_name_or_path: str,
    *,
    is_trainable: bool,
    load_in_4bit: bool,
    device_map: dict[str, int] | None,
    dtype: torch.dtype = torch.bfloat16,
):
    adapter_config_path = Path(model_name_or_path) / "adapter_config.json"
    quant_config = _make_quant_config(load_in_4bit, dtype)
    local_files_only = _resolve_local_files_only(model_name_or_path)
    common_kwargs = {
        "device_map": device_map,
        "local_files_only": local_files_only,
    }
    if quant_config is not None:
        common_kwargs["quantization_config"] = quant_config
    else:
        common_kwargs["torch_dtype"] = dtype

    if adapter_config_path.exists():
        return AutoPeftModel.from_pretrained(
            model_name_or_path,
            is_trainable=is_trainable,
            **common_kwargs,
        )

    model_args = dllm.utils.ModelArguments(
        model_name_or_path=model_name_or_path,
        load_in_4bit=load_in_4bit,
        lora=is_trainable,
    )
    model = dllm.utils.get_model(
        model_args=model_args,
        dtype=dtype,
        device_map=device_map,
    )
    for param in model.parameters():
        param.requires_grad_(is_trainable and param.requires_grad)
    return model


def _record_teacher_targets(
    teacher_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    teacher_rollout_steps: int,
    scheduler: BaseAlphaScheduler,
    mask_token_id: int,
) -> torch.Tensor:
    x = input_ids.clone()
    initial_mask = x.eq(mask_token_id)
    targets = torch.full(
        (*x.shape, teacher_model.config.vocab_size),
        fill_value=-torch.inf,
        device=x.device,
        dtype=torch.float32,
    )

    num_transfer_tokens = get_num_transfer_tokens(
        mask_index=initial_mask,
        steps=teacher_rollout_steps,
        scheduler=scheduler,
        stochastic=False,
    )

    for i in range(num_transfer_tokens.shape[1]):
        current_mask = x.eq(mask_token_id)
        if not current_mask.any():
            break

        logits = teacher_model(input_ids=x, attention_mask=attention_mask).logits
        log_probs = F.log_softmax(logits.float(), dim=-1)
        x0 = torch.argmax(log_probs, dim=-1)
        confidence = torch.gather(log_probs.exp(), -1, x0.unsqueeze(-1)).squeeze(-1)
        confidence = torch.where(current_mask, confidence, torch.full_like(confidence, -torch.inf))

        transfer_index = torch.zeros_like(current_mask, dtype=torch.bool)
        for row in range(x.shape[0]):
            k = int(num_transfer_tokens[row, i].item())
            if k <= 0:
                continue
            _, select_index = torch.topk(confidence[row], k=k)
            transfer_index[row, select_index] = True

        if transfer_index.any():
            targets[transfer_index] = log_probs[transfer_index]
            x[transfer_index] = x0[transfer_index]

    remaining_mask = x.eq(mask_token_id)
    if remaining_mask.any():
        logits = teacher_model(input_ids=x, attention_mask=attention_mask).logits
        log_probs = F.log_softmax(logits.float(), dim=-1)
        targets[remaining_mask] = log_probs[remaining_mask]
    return targets


class LLaDATrajectoryDistillationTrainer(transformers.Trainer):
    def __init__(
        self,
        *,
        teacher_model,
        args: SDTTDistillationArguments,
        scheduler: BaseAlphaScheduler | None = None,
        **kwargs,
    ):
        super().__init__(args=args, **kwargs)
        self.teacher_model = teacher_model.eval()
        self.teacher_model.requires_grad_(False)
        self.scheduler = scheduler if scheduler is not None else LinearAlphaScheduler()
        self.teacher_steps = args.teacher_steps
        self.student_steps = args.student_steps
        self.temperature = args.distill_temperature
        self.time_epsilon = args.time_epsilon
        self.loss_weight_type = args.loss_weight_type
        self.teacher_rollout_steps = max(1, args.teacher_steps // max(1, args.student_steps))

    def _compute_loss_weights(
        self,
        t: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self.loss_weight_type == "scheduler":
            return self.scheduler.weight(t).unsqueeze(1).repeat(1, input_ids.shape[1])
        if self.loss_weight_type == "uniform":
            return torch.ones_like(input_ids, dtype=torch.float32)
        raise NotImplementedError(self.loss_weight_type)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        attention_mask = inputs.get("attention_mask", None)
        maskable_mask = labels != -100
        batch_size, seq_len = input_ids.shape

        t = self.time_epsilon + (1 - self.time_epsilon) * torch.rand(
            batch_size,
            device=input_ids.device,
        )
        p_mask = 1.0 - self.scheduler(t).unsqueeze(1).expand(batch_size, seq_len)
        masked_mask = (
            torch.rand((batch_size, seq_len), device=input_ids.device) < p_mask
        ) & maskable_mask
        noised_input_ids = torch.where(
            masked_mask,
            self.processing_class.mask_token_id,
            input_ids,
        )

        with torch.no_grad():
            teacher_targets = _record_teacher_targets(
                self.teacher_model,
                noised_input_ids,
                attention_mask,
                teacher_rollout_steps=self.teacher_rollout_steps,
                scheduler=self.scheduler,
                mask_token_id=self.processing_class.mask_token_id,
            )

        outputs = model(input_ids=noised_input_ids, attention_mask=attention_mask)
        student_log_probs = F.log_softmax(outputs.logits.float() / self.temperature, dim=-1)

        loss_weights = self._compute_loss_weights(t=t, input_ids=input_ids).to(student_log_probs.dtype)
        masked_indices = masked_mask.unsqueeze(-1).expand_as(student_log_probs)

        teacher_masked = teacher_targets[masked_mask]
        student_masked = student_log_probs[masked_mask]
        per_token_kl = F.kl_div(
            teacher_masked,
            student_masked,
            log_target=True,
            reduction="none",
        ).sum(dim=-1)
        weight_masked = loss_weights[masked_mask]
        loss = (per_token_kl * weight_masked).sum() / masked_mask.sum().clamp_min(1)
        return (loss, outputs) if return_outputs else loss


class SDTTLLaDAMethod(AccelerationMethod):
    method_name = "sdtt_llada"

    def prepare_benchmark_config(self, config: BenchmarkConfig) -> BenchmarkConfig:
        metadata = _load_sdtt_metadata(config.model_name_or_path)
        if metadata is None:
            return config
        if config.steps == 64:
            config.steps = max(metadata.student_steps, SDTT_BENCHMARK_STEPS)
        if config.block_size == 64:
            config.block_size = max(metadata.block_size, SDTT_BENCHMARK_BLOCK_SIZE)
        return config

    def build_model(self, config: BenchmarkConfig):
        model_args = dllm.utils.ModelArguments(model_name_or_path=config.model_name_or_path)
        return dllm.utils.get_model(model_args=model_args).eval()

    def build_tokenizer(self, config: BenchmarkConfig):
        model_args = dllm.utils.ModelArguments(model_name_or_path=config.model_name_or_path)
        return dllm.utils.get_tokenizer(model_args=model_args)

    def build_sampler(self, *, model, tokenizer, config: BenchmarkConfig):
        return MDLMSampler(model=model, tokenizer=tokenizer)
