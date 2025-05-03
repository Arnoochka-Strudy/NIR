from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.data.gpt_dataset import build_train_valid_test_datasets
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
import time
_TRAIN_START_TIME = time.time()
import torch
from megatron import get_args
from megatron import get_timers
from megatron.core import mpu, tensor_parallel
from megatron import print_rank_0
from megatron.model import GPTModel
from megatron.core.enums import ModelType
from megatron.initialize import initialize_megatron
from megatron.initialize import set_jit_fusion_options
from megatron.utils import found_kill_switch
from megatron.arguments import core_transformer_config_from_args

import deepspeed
from deepspeed.accelerator import get_accelerator


def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building GPT model ...')
    see_memory_usage(f"Before Building Model", force=True)

    args = get_args()
    config = core_transformer_config_from_args(args)
    with deepspeed.zero.Init(data_parallel_group=mpu.get_sequence_data_parallel_group(),
                             config_dict_or_path=args.deepspeed_config_dict,
                             enabled=True,
                             mpu=mpu):
            model = GPTModel(
                config=config,
                num_tokentypes=0,
                parallel_output=True,
                pre_process=pre_process,
                post_process=post_process,
                return_moe_loss=False
            )
    see_memory_usage(f"After Building Model", force=True)
    return model

def encode_batch_from_sentences(data_iterator, tokenizer, seq_length, device='cuda'):
    
    sentences = next(data_iterator)

    encoded = [tokenizer.tokenize(s) for s in sentences]

    tokens = []
    for ids in encoded:
        if len(ids) >= seq_length:
            tokens.append(ids[:seq_length])
        else:
            pad_len = seq_length - len(ids)
            tokens.append(ids + [tokenizer.eod] * pad_len)

    tokens_tensor = torch.tensor(tokens, dtype=torch.long, device=device)
    return {'text': tokens_tensor}


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()
    
    data = encode_batch_from_sentences(data_iterator, tokenizer, args.seq_length)

    # Items and their type.
    # keys = ['text']
    # datatype = torch.int64
    
    # if data_iterator is not None:
    #     data = next(data_iterator)
    # else:
    #     data = None
    # data_b = tensor_parallel.broadcast_data(keys, data, datatype)
    tokens_ = data['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    return tokens, labels, loss_mask, attention_mask, position_ids

def batch_generator(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def forward_step(data_iterator, model):
    """Forward step."""
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator)
    timers('batch-generator').stop()
    logits, loss = model(tokens, position_ids, attention_mask,
                                            labels=None)
    full_logits = tensor_parallel.gather_from_tensor_model_parallel_region(logits)
    next_token_logits = full_logits[:, -1, :]
    next_token_id = torch.argmax(next_token_logits, dim=-1)  
    print(f"forward: logits:{logits},\nid:{next_token_id},\nfull:{full_logits}")
    return next_token_id


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for GPT ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
        train_data_prefix=args.train_data_path,
        valid_data_prefix=args.valid_data_path,
        test_data_prefix=args.test_data_path,
        data_cache_path=args.data_cache_path)
    print_rank_0("> finished creating GPT datasets ...")
    print(f"train_ds: {train_ds}")
    return train_ds, valid_ds, test_ds

def get_prompts(filename: str | None = None) -> list[str]:
    
    if filename is None:
        return ["Paris is the capital city of"] * 4
    
    with open(filename, 'r') as file:
        promts = [line for line in file]
        return promts


def inference(model_provider,
             model_type,
             forward_step_func,
             extra_args_provider=None,
             args_defaults={},
             external_args={}):

    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(extra_args_provider=extra_args_provider,
                        args_defaults=args_defaults, external_args=external_args)

    args = get_args()

    if found_kill_switch():
        print_datetime(f"Detected kill switch at {args.kill_switch_file}. Exiting")
        sys.exit()

    # Set pytorch JIT layer fusion options and warmup JIT functions.
    if get_accelerator().device_name() == 'cuda':
        set_jit_fusion_options()

    # Adjust the startup time so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    global _TRAIN_START_TIME
    start_time_tensor = get_accelerator().DoubleTensor([_TRAIN_START_TIME])
    torch.distributed.all_reduce(start_time_tensor,
                                 op=torch.distributed.ReduceOp.MIN)
    _TRAIN_START_TIME = start_time_tensor.item()
    print_rank_0('time to initialize megatron (seconds): {:.3f}'.format(
        time.time() - _TRAIN_START_TIME))
    print_datetime('after megatron is initialized')

    timers = get_timers()

    if args.deepspeed:
        args.deepspeed_config_dict = _create_ds_config_dict()

    # Model, optimizer, and learning rate.
    timers('model-and-optimizer-setup', log_level=0).start(barrier=True)
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
        model_provider, model_type)
    timers('model-and-optimizer-setup').stop()
    print_datetime('model is builded')
    
    promts = get_prompts("/home/victor/Study/NIR/benchmark_mean.txt")
    batch_size = args.micro_batch_size
    data_iterator = batch_generator(promts, batch_size)
    length = (len(promts) + batch_size - 1) // batch_size
    output = []
    for _ in range(length):
        output.append(forward_step_func(data_iterator, model))
    
    return model, output

if __name__ == "__main__":
    inference(model_provider=model_provider,
             model_type=ModelType.encoder_or_decoder,
             forward_step_func=forward_step,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
