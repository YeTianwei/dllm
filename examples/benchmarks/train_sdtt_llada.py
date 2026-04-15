"""
Run:
    CUDA_VISIBLE_DEVICES=3 python /data/ytw/VLA_baseline/dllm/examples/benchmarks/train_sdtt_llada.py \
        --teacher_model_name_or_path /data/ytw/VLA_baseline/dllm/.models/smoke_test_llada_sft/checkpoint-final \
        --student_model_name_or_path /data/ytw/VLA_baseline/dllm/.models/smoke_test_llada_sft/checkpoint-final \
        --dataset_args "tatsu-lab/alpaca[train:8,test:4]" \
        --max_steps 1 \
        --teacher_steps 16 \
        --student_steps 4 \
        --output_dir /data/ytw/VLA_baseline/dllm/.models/sdtt-llada-smoke
"""

import os
from dataclasses import dataclass, field
from functools import partial

import accelerate
import torch
import transformers

import dllm
from dllm.acceleration.methods.sdtt_llada import (
    LLaDATrajectoryDistillationTrainer,
    SDTTDistillationArguments,
    SDTTMethodMetadata,
    load_llada_checkpoint,
    save_sdtt_metadata,
)

logger = dllm.utils.get_default_logger(__name__)


@dataclass
class SDTTModelArguments:
    teacher_model_name_or_path: str = (
        "/data/ytw/VLA_baseline/dllm/.models/smoke_test_llada_sft/checkpoint-final"
    )
    student_model_name_or_path: str = (
        "/data/ytw/VLA_baseline/dllm/.models/smoke_test_llada_sft/checkpoint-final"
    )
    load_in_4bit: bool = True
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

    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

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

    device_map = {"": accelerate.PartialState().local_process_index}
    dtype = getattr(torch, model_args.dtype)
    teacher_model = load_llada_checkpoint(
        model_args.teacher_model_name_or_path,
        is_trainable=False,
        load_in_4bit=model_args.load_in_4bit,
        device_map=device_map,
        dtype=dtype,
    )
    student_model = load_llada_checkpoint(
        model_args.student_model_name_or_path,
        is_trainable=True,
        load_in_4bit=model_args.load_in_4bit,
        device_map=device_map,
        dtype=dtype,
    )

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
