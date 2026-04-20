"""
Run:
    CUDA_VISIBLE_DEVICES=3 python examples/benchmarks/train_sdtt_llada.py \
        --teacher_model_name_or_path .models/smoke_test_llada_sft/checkpoint-final \
        --student_model_name_or_path .models/smoke_test_llada_sft/checkpoint-final \
        --dataset_args "tatsu-lab/alpaca[train:8,test:4]" \
        --max_steps 1 \
        --teacher_steps 16 \
        --student_steps 4 \
        --output_dir .models/sdtt-llada-smoke

    CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
        --config_file scripts/accelerate_configs/zero3.yaml \
        --num_processes 2 \
        examples/benchmarks/train_sdtt_llada.py \
        --teacher_model_name_or_path .models/smoke_test_llada_sft/checkpoint-final \
        --student_model_name_or_path .models/smoke_test_llada_sft/checkpoint-final \
        --dataset_args "tatsu-lab/alpaca[train:8,test:4]" \
        --max_steps 1 \
        --teacher_steps 16 \
        --student_steps 4 \
        --output_dir .models/sdtt-llada-smoke
"""

import os
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import accelerate
import deepspeed
import torch
import transformers
from transformers.integrations import deepspeed as hf_deepspeed

import dllm
from dllm.acceleration.methods.sdtt_llada import (
    LLaDATrajectoryDistillationTrainer,
    SDTTDistillationArguments,
    SDTTMethodMetadata,
    load_llada_checkpoint,
    save_sdtt_metadata,
)

logger = dllm.utils.get_default_logger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_repo_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return str((REPO_ROOT / path).resolve())


@dataclass
class SDTTModelArguments:
    teacher_model_name_or_path: str = (
        ".models/smoke_test_llada_sft/checkpoint-final"
    )
    student_model_name_or_path: str = (
        ".models/smoke_test_llada_sft/checkpoint-final"
    )
    load_in_4bit: bool = False
    dtype: str = "bfloat16"


@dataclass
class SDTTDataArguments(dllm.utils.DataArguments):
    dataset_args: str = "tatsu-lab/alpaca[train:32,test:8]"
    load_preprocessed_data: bool = False
    mask_prompt_loss: bool = field(
        default=True,
        metadata={"help": "Whether to mask the loss on the prompt tokens"},
    )


def main():
    parser = transformers.HfArgumentParser(
        (SDTTModelArguments, SDTTDataArguments, SDTTDistillationArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args.teacher_model_name_or_path = _resolve_repo_path(
        model_args.teacher_model_name_or_path
    )
    model_args.student_model_name_or_path = _resolve_repo_path(
        model_args.student_model_name_or_path
    )
    training_args.output_dir = _resolve_repo_path(training_args.output_dir)

    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)
    state = accelerate.PartialState()
    is_zero3 = transformers.modeling_utils.is_deepspeed_zero3_enabled()

    tokenizer = dllm.utils.get_tokenizer(
        model_name_or_path=model_args.teacher_model_name_or_path
    )

    with accelerate.PartialState().local_main_process_first():
        dataset = dllm.data.load_sft_dataset(
            data_args.dataset_args,
            load_preprocessed_data=data_args.load_preprocessed_data,
        )
        if not data_args.load_preprocessed_data:
            map_fn = partial(
                dllm.utils.default_sft_map_fn,
                tokenizer=tokenizer,
                mask_prompt_loss=data_args.mask_prompt_loss,
            )
            dataset = dataset.map(
                map_fn,
                num_proc=data_args.num_proc,
                desc="Mapping dataset to SDTT format",
            )
        dataset = dllm.utils.post_process_dataset(dataset, data_args)
        keep_columns = {"input_ids", "labels", "attention_mask"}
        for split in list(dataset.keys()):
            drop_columns = [
                col for col in dataset[split].column_names if col not in keep_columns
            ]
            if drop_columns:
                dataset[split] = dataset[split].remove_columns(drop_columns)

    dtype = getattr(torch, model_args.dtype)
    process_device = state.device
    teacher_device_map = None if is_zero3 else (
        {"": state.local_process_index} if torch.cuda.is_available() else None
    )
    student_device_map = None if is_zero3 else teacher_device_map
    logger.info(
        "Distributed setup: rank=%s local_rank=%s world_size=%s zero3=%s teacher_device_map=%s student_device_map=%s",
        state.process_index,
        state.local_process_index,
        state.num_processes,
        is_zero3,
        teacher_device_map,
        student_device_map,
    )
    teacher_load_context = deepspeed.zero.Init(enabled=False) if is_zero3 else nullcontext()
    hf_ds_config = None
    if is_zero3:
        weak_ref = getattr(hf_deepspeed, "_hf_deepspeed_config_weak_ref", None)
        hf_ds_config = weak_ref() if weak_ref is not None else None
        hf_deepspeed.unset_hf_deepspeed_config()
    with teacher_load_context:
        teacher_model = load_llada_checkpoint(
            model_args.teacher_model_name_or_path,
            is_trainable=False,
            load_in_4bit=model_args.load_in_4bit,
            device_map=teacher_device_map,
            dtype=dtype,
        )
    if is_zero3 and hf_ds_config is not None:
        hf_deepspeed.set_hf_deepspeed_config(hf_ds_config)
    if is_zero3 and torch.cuda.is_available():
        teacher_model.to(process_device)
    student_model = load_llada_checkpoint(
        model_args.student_model_name_or_path,
        is_trainable=True,
        load_in_4bit=model_args.load_in_4bit,
        device_map=student_device_map,
        dtype=dtype,
    )

    # Enable gradient checkpointing to save memory.
    if training_args.gradient_checkpointing:
        teacher_model.gradient_checkpointing_enable()
        student_model.gradient_checkpointing_enable()

    trainer = LLaDATrajectoryDistillationTrainer(
        teacher_model=teacher_model,
        model=student_model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test", None),
        args=training_args,
        data_collator=dllm.utils.NoAttentionMaskWrapper(
            transformers.DataCollatorForSeq2Seq(
                tokenizer,
                return_tensors="pt",
                padding=True,
                label_pad_token_id=-100,
            ),
        ),
    )

    logger.info("Start SDTT-style trajectory distillation...")
    trainer.train()

    final_dir = os.path.join(training_args.output_dir, "checkpoint-final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    save_sdtt_metadata(
        final_dir,
        SDTTMethodMetadata(
            teacher_model_name_or_path=model_args.teacher_model_name_or_path,
            student_model_name_or_path=model_args.student_model_name_or_path,
            teacher_steps=training_args.teacher_steps,
            student_steps=training_args.student_steps,
            block_size=training_args.block_size,
            temperature=training_args.distill_temperature,
        ),
    )


if __name__ == "__main__":
    main()
