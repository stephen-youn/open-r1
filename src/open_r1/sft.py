# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Supervised fine-tuning script for decoder language models.

Usage:

# One 1 node of 8 x H100s
accelerate launch --config_file=recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --model_name_or_path open-r1/Qwen2.5-Math-7B-RoPE-300k \
    --dataset_name open-r1/Mixture-of-Thoughts \
    --dataset_config all \
    --eos_token '<|im_end|>' \
    --learning_rate 4.0e-5 \
    --num_train_epochs 5 \
    --max_seq_length 32768 \
    --per_device_train_batch_size 2 \
    --gradient_checkpointing \
    --bf16 \
    --use_liger_kernel \
    --output_dir data/OpenR1-Distill-7B

accelerate launch --config_file=recipes/accelerate_configs/fsdp.yaml src/open_r1/sft.py \
    --model_name_or_path /home/stepyoun/projects/sparse-cot/duo_attn/hybrid_model/DeepSeek-R1-Distill-Llama-8B-Untrained \
    --dataset_name open-r1/Mixture-of-Thoughts \
    --dataset_config all     --learning_rate 4.0e-5     --num_train_epochs 5     --max_seq_length 16384     --per_device_train_batch_size 1 \
    --bf16  True   --use_liger_kernel  --output_dir data/r1-8B-10000 2>&1 | tee aug20.10000.log
"""

import logging
import os
import sys
import torch

import datasets
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import ScriptArguments, SFTConfig
from open_r1.utils import get_dataset, get_model, get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from trl import ModelConfig, SFTTrainer, TrlParser, get_peft_config, setup_chat_format

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)

from duo_attn.patch import enable_duo_attention_eval, enable_duo_attention_sft
from duo_attn.utils import (
    to_device,
    load_attn_pattern,
    sparsify_attention_heads,
)
from duo_attn.patch.tuple_kv_cache import enable_tuple_kv_cache
import json

logger = logging.getLogger(__name__)

def load_model_and_tokenizer(path, model_name, max_seq_len, args):
    tokenizer = AutoTokenizer.from_pretrained(
        path, trust_remote_code=True, use_fast=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )

    generation_config = GenerationConfig.from_pretrained(path)
    eos_token_ids = generation_config.eos_token_id
    if not isinstance(eos_token_ids, list):
        eos_token_ids = [eos_token_ids]

    model = model.eval()

    assert args.attn_load_dir is not None, "attn_load_dir must be provided"
    print(
        f"Loading attention pattern from {args.attn_load_dir} with sparsity {args.sparsity}"
    )
    full_attention_heads, sink_size, recent_size = load_attn_pattern(
        args.attn_load_dir
    )

    if args.sink_size is not None:
        sink_size = args.sink_size
    if args.recent_size is not None:
        recent_size = args.recent_size

    full_attention_heads, sparsity = sparsify_attention_heads(
        full_attention_heads, None, sparsity=args.sparsity
    )
    print(f"True sparsity: {sparsity}")

    enable_duo_attention_sft(
        model,
        full_attention_heads,
        sink_size,
        recent_size,
        # 0, # query_size
        # lm_eval=True,
        # sparse_prefill=args.spattern if args.sparsity else "",
        sparse_prefill=args.spattern if args.sparsity and (args.spattern.lower() != "none") else "",
        # sparse_attn_implementation="flex",
        sparse_attn_implementation="sdpa",
        max_seq_len=max_seq_len,
    )

    return model, tokenizer, eos_token_ids

# (c) Meta Platforms, Inc. and affiliates. 
import logging
import socket
from datetime import datetime

logging.basicConfig(
   format="%(levelname)s:%(asctime)s %(message)s",
   level=logging.INFO,
   datefmt="%Y-%m-%d %H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"

# Keep a max of 100,000 alloc/free events in the recorded history
# leading up to the snapshot.
MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000

def start_record_memory_history() -> None:
   if not torch.cuda.is_available():
       logger.info("CUDA unavailable. Not recording memory history")
       return

   logger.info("Starting snapshot record_memory_history")
   torch.cuda.memory._record_memory_history(
       max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
   )

def stop_record_memory_history() -> None:
   if not torch.cuda.is_available():
       logger.info("CUDA unavailable. Not recording memory history")
       return

   logger.info("Stopping snapshot record_memory_history")
   torch.cuda.memory._record_memory_history(enabled=None)

def export_memory_snapshot() -> None:
   if not torch.cuda.is_available():
       logger.info("CUDA unavailable. Not exporting memory snapshot")
       return

   # Prefix for file names.
   host_name = socket.gethostname()
   timestamp = datetime.now().strftime(TIME_FORMAT_STR)
   file_prefix = f"{host_name}_{timestamp}"

   try:
       logger.info(f"Saving snapshot to local file: {file_prefix}.pickle")
       torch.cuda.memory._dump_snapshot(f"{file_prefix}.pickle")
   except Exception as e:
       logger.error(f"Failed to capture memory snapshot {e}")
       return


def main(script_args, training_args, model_args):
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    log_level = logging.DEBUG
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    ######################################
    # Load dataset, tokenizer, and model #
    ######################################
    dataset = get_dataset(script_args)
    tokenizer = get_tokenizer(model_args, training_args)
    if True:
        # model_name = model_args.model_name_or_path.split('/')[-1]
        # model_args.attn_load_dir  = f"/home/stepyoun/projects/sparse-cot/attn_patterns/{model_name}/lr=0.02-reg=0.05-ctx=1000_16384-multi_passkey10"
        # model_args.sparsity = 0.5
        # model_args.sink_size = 256
        # model_args.recent_size = 128
        # model_args.spattern = "streaming"
        # model, tokenizer, eos_token_ids = load_model_and_tokenizer(model_args.model_name_or_path, model_name, training_args.max_seq_length, model_args)
        # # tokenizer.eos_token_id = eos_token_ids
        from duo_attn.hybrid_model.hybrid_wrapper import SparseTransformerHybridModelWrapper, load_sparse_attn_config
        from trl import ModelConfig, get_kbit_device_map, get_quantization_config
        torch._dynamo.config.recompile_limit = 8192
        torch_dtype = (
            model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
        )
        # quantization_config = get_quantization_config(model_args)
        model_kwargs = dict(
            # revision=model_args.model_revision,
            # trust_remote_code=model_args.trust_remote_code,
            # attn_implementation=model_args.attn_implementation,
            torch_dtype=torch_dtype,
            # use_cache=False if training_args.gradient_checkpointing else True,
            use_cache=False,
            # device_map=get_kbit_device_map() if quantization_config is not None else None,
            # quantization_config=quantization_config,
        )
        # model = AutoModelForCausalLM.from_pretrained(
        #     model_args.model_name_or_path,
        #     **model_kwargs,
        # )
        assert not training_args.gradient_checkpointing
        model = SparseTransformerHybridModelWrapper.from_pretrained(model_args.model_name_or_path, **model_kwargs)
        # model.model.use_cache = False
        # if 'qwen' in model_args.model_name_or_path.lower():
        tokenizer.padding_side  = 'left'
    else:
        model = get_model(model_args, training_args)

    if torch.distributed.get_rank() == 0:
        logger.info(model_args)
        logger.info(training_args)
        logger.info(script_args)
        logger.info(f"model=\n{model}")

    # training_args.find_unused_parameters=True
    training_args.gradient_checkpointing=False
    training_args.use_cache=False
    if tokenizer.chat_template is None:
        logger.info("No chat template provided, defaulting to ChatML.")
        model, tokenizer = setup_chat_format(model, tokenizer, format="chatml")

    ############################
    # Initialize the SFT Trainer
    ############################
    # training_args.packing=True
    # training_args.max_length=16384
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split].select(range(16384)),
        eval_dataset=(dataset[script_args.dataset_test_split].select(range(16384)) if training_args.eval_strategy != "no" else None),
        # train_dataset=dataset[script_args.dataset_train_split],
        # eval_dataset=(dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None),
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
    )

    # from torch.profiler import profile, record_function, ProfilerActivity
    # if torch.distributed.get_rank() == 0:
    #     start_record_memory_history()
    #     with profile(activities=[ProfilerActivity.CUDA],
    #         profile_memory=True, record_shapes=True) as prof:
    #         ###############
    #         # Training loop
    #         ###############
    #         logger.info("*** Train ***")
    #         torch.autograd.set_detect_anomaly(True)
    #         checkpoint = None
    #         if training_args.resume_from_checkpoint is not None:
    #             checkpoint = training_args.resume_from_checkpoint
    #         elif last_checkpoint is not None:
    #             checkpoint = last_checkpoint
    #         train_result = trainer.train(resume_from_checkpoint=checkpoint)
    #     sort_metric = "self_cuda_memory_usage" if torch.cuda.is_available() else "self_cpu_memory_usage"
    #     print(prof.key_averages().table(sort_by=sort_metric, row_limit=32))
    #     # Create the memory snapshot file
    #     export_memory_snapshot()
    #     # Stop recording memory snapshot history
    #     stop_record_memory_history()
    
    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    torch.autograd.set_detect_anomaly(True)
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    # Align the model's generation config with the tokenizer's eos token
    # to avoid unbounded generation in the transformers `pipeline()` function
    # trainer.model.generation_config.eos_token_id = tokenizer.eos_token_id
    trainer.model.model.generation_config.eos_token_id = tokenizer.eos_token_id
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
