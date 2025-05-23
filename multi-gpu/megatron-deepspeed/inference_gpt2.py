from megatron import get_args
from megatron import print_rank_0
from megatron import get_tokenizer
from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.model import GPTModel
from megatron.training import (print_datetime,
                               _create_ds_config_dict,
                               setup_model_and_optimizer)
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.arguments import core_transformer_config_from_args

import deepspeed
from deepspeed.runtime.utils import see_memory_usage
from deepspeed.accelerator.real_accelerator import get_accelerator

import sys
import torch
from megatron.core import mpu, tensor_parallel
from megatron.model import GPTModel
from megatron.core.enums import ModelType
from megatron.initialize import initialize_megatron
from megatron.initialize import set_jit_fusion_options
from megatron.utils import found_kill_switch
from megatron.arguments import core_transformer_config_from_args

import deepspeed
from deepspeed.accelerator import get_accelerator

from utils import (add_model_hooks, remove_model_hooks,
                   cache_bytes,get_filename,
                   hidden_bytes,model_bytes, write_benchmark_log,
                   get_prompts, get_inference_args)

from BatchGenerator import BatchGenerator
from timer import timers


def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building GPT model ...')
    see_memory_usage(f"Before Building Model", force=True)

    args = get_args()
    config = core_transformer_config_from_args(args)
    with deepspeed.zero.Init(data_parallel_group=mpu.get_sequence_data_parallel_group(),
                             config_dict_or_path=args.deepspeed_config_dict,
                             enabled=args.use_zero,
                             mpu=mpu):
            model = GPTModel(
                config=config,
                num_tokentypes=0,
                parallel_output=True,
                pre_process=pre_process,
                post_process=post_process,
                return_moe_loss=True
            )
    see_memory_usage(f"After Building Model", force=True)
    return model


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()
    data = next(data_iterator)

    keys = ['text']
    datatype = torch.int64
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)
    tokens = data_b['text'].long()

    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    return tokens, attention_mask, position_ids

def generate(data_iterator, model):
    """Generate multiple tokens per sequence in one forward step with updated attention_mask."""
    args = get_args()
    tokens, attention_mask, position_ids = get_batch(data_iterator)

    batch_size, seq_len = tokens.size()
    generated_tokens = []

    for step in range(args.gen_len):
        # Forward pass
        print_rank_0(f"tokens: {tokens.shape}")
        logits = model(tokens, position_ids, attention_mask)[0]
        full_logits = tensor_parallel.gather_from_tensor_model_parallel_region(logits)
        next_token_logits = full_logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)

        # Append the predicted token
        generated_tokens.append(next_token_id)

        # Update input tensors
        tokens = torch.roll(tokens, shifts=-1, dims=1)
        tokens[:, -1] = next_token_id.squeeze(1)

    # Stack all generated tokens into a tensor of shape [batch_size, num_tokens_to_generate]
    generated_tokens = torch.cat(generated_tokens, dim=1)

    return generated_tokens


def inference(model_provider,
             model_type,
             generate_func,
             extra_args_provider=None,
             args_defaults={},
             external_args={}):

    initialize_megatron(extra_args_provider=extra_args_provider,
                        args_defaults=args_defaults,
                        external_args=external_args)

    args = get_args()

    if found_kill_switch():
        print_datetime(f"Detected kill switch at {args.kill_switch_file}. Exiting")
        sys.exit()

    # Set pytorch JIT layer fusion options and warmup JIT functions.
    if get_accelerator().device_name() == 'cuda':
        set_jit_fusion_options()

    if args.deepspeed:
        args.deepspeed_config_dict = _create_ds_config_dict()

    model = setup_model_and_optimizer(model_provider, model_type)[0][0]
    promts = get_prompts(filename="/home/ubuntu/NIR/benchmark_mean.txt")
    batch_size = args.micro_batch_size
    tokenizer = get_tokenizer()
    
    add_model_hooks(model)
    def set_model_stage(model, stage):
        model.stage = stage
        
    prefill_timings = []
    timer = timers("generate-forward")
    
    length = (len(promts) + batch_size - 1) // batch_size
    for _ in range(args.loops):
        data_iterator = BatchGenerator.batch_generator(promts, batch_size, args.seq_length, tokenizer)
        timer.start(sync_func=get_accelerator().synchronize)
        output =[]
        with torch.no_grad():
            set_model_stage(model, "prefill")
            
            for _ in range(length):
                output.append(generate_func(data_iterator, model))
                
            prefill_timings.append(model.__duration__)
        timer.stop(sync_func=get_accelerator().synchronize)
            
    remove_model_hooks(model) 
            
    costs = timers("generate-forward").costs
    
    if args.local_rank != 0:
        return

    # Log output
    print(f"Summary:")
    print(f"costs = {costs} prefill_timings = {prefill_timings}")
    total_latency = costs[-1]
    prefill_latency = prefill_timings[-1]
    
    prefill_throughput = args.global_batch_size * args.seq_length / prefill_latency
    decode_latency = total_latency - prefill_latency
    decode_throughput = args.global_batch_size * (args.gen_len - 1) / max(decode_latency, 1e-10)
    num_generated_tokens = args.global_batch_size * args.gen_len
    total_throughput = num_generated_tokens / total_latency
    gpu_peak_mem = get_accelerator().max_memory_allocated(torch.device("cuda"))

    filename = args.output_file
    
    args.vocab_size = 50257
    cache_size = cache_bytes(args, args.global_batch_size, args.seq_length + args.gen_len)
    hidden_size = hidden_bytes(args, args.global_batch_size, args.seq_length + args.gen_len)
    log_str = write_benchmark_log(
        args, 
        filename,
        model_bytes(args),
        cache_size,
        hidden_size,
        gpu_peak_mem,
        prefill_latency,
        prefill_throughput,
        decode_latency,
        decode_throughput,
        total_latency,
        total_throughput,
    )
    if args.verbose >= 1:
        print(log_str) 
    
    torch.distributed.destroy_process_group()
    
    return model, output

if __name__ == "__main__":
    inference(model_provider=model_provider,
             model_type=ModelType.encoder_or_decoder,
             generate_func=generate,
             extra_args_provider=get_inference_args,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
