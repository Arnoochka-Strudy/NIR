# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain GPT"""
from functools import partial
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.model import GPTModel
from megatron.training import (evaluate_and_print_results,
                               print_datetime,
                               _create_ds_config_dict,
                               build_train_valid_test_data_iterators,
                               setup_model_and_optimizer,
                               train)
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import average_losses_across_data_parallel_group
from megatron.arguments import core_transformer_config_from_args

import deepspeed
from deepspeed.runtime.utils import see_memory_usage
from deepspeed.accelerator.real_accelerator import get_accelerator

import sys
import time
# The earliest we can measure the start time.
_TRAIN_START_TIME = time.time()
import torch
from megatron import get_args
from megatron import get_timers
from megatron.core import mpu, tensor_parallel
from megatron import print_rank_0
from megatron.checkpointing import save_checkpoint
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

def encode_batch_from_sentences(sentences, tokenizer, seq_length, device='cuda'):
    """Токенизирует и паддит список предложений до заданной длины."""
    import torch

    # Токенизация с добавлением специальных токенов, если нужно
    encoded = [tokenizer.tokenize(s) for s in sentences]

    # Обрезка или паддинг
    tokens = []
    for ids in encoded:
        if len(ids) >= seq_length:
            tokens.append(ids[:seq_length])
        else:
            # Паддинг до нужной длины токеном EOD
            pad_len = seq_length - len(ids)
            tokens.append(ids + [tokenizer.eod] * pad_len)

    # Преобразуем в тензор
    tokens_tensor = torch.tensor(tokens, dtype=torch.long, device=device)
    return {'text': tokens_tensor}


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()
    data = encode_batch_from_sentences(["hello world", "hi", "hr", "fuck"], tokenizer, args.seq_length)

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    # if data_iterator is not None:
    #     data = next(data_iterator)
    # else:
    #     data = None
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)
    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    skip_mask = args.use_flash_attn or args.use_flash_attn_triton
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss,
        skip_mask)

    return tokens, labels, loss_mask, attention_mask, position_ids


def loss_func(loss_mask, output_tensor):
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])
    return loss, {'lm loss': averaged_loss[0]}

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

def init_megatron(extra_args_provider=None,
                  args_defaults={},
                  external_args={}):
    
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
        
    return args, timers

def inference(train_valid_test_dataset_provider,
             model_provider,
             model_type,
             forward_step_func,
             process_non_loss_data_func=None,
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
        model_provider, model_type,
        build_train_valid_test_datasets_provider=train_valid_test_dataset_provider)
    timers('model-and-optimizer-setup').stop()
    print_datetime('after model, optimizer, and learning rate '
                   'scheduler are built')

    # Data stuff.
    timers('train/valid/test-data-iterators-setup', log_level=0).start(
        barrier=True)

    train_data_iterator, valid_data_iterator, test_data_iterator \
        = build_train_valid_test_data_iterators(
            train_valid_test_dataset_provider)
        
    timers('train/valid/test-data-iterators-setup').stop()
    print_datetime('after dataloaders are built')

    # args.teacher_model is used as global variable to pass the teacher model
    # for knowledge distillation. Users do not need to set it in the command
    # line to use kd, but users do need to provide teacher model configurations
    # like args.num_layers_teacher as described in setup_teacher_model()
    args.teacher_model = None

    # Print setup timing.
    print_rank_0('done with setup ...')
    timers.log(['model-and-optimizer-setup',
                'train/valid/test-data-iterators-setup'], barrier=True)

    if not args.skip_train:
        print_rank_0('training ...')

        iteration = 0
        if args.do_train and args.train_iters > 0:
            iteration = train(forward_step_func,
                            model, optimizer, opt_param_scheduler,
                            train_data_iterator, valid_data_iterator,
                            process_non_loss_data_func)

        print_datetime('after training is done')

        if args.save and iteration != 0:
            save_checkpoint(iteration, model, optimizer, opt_param_scheduler)
    else:
        print_rank_0('skipping training (--skip-train is on) ...')

        iteration = args.iteration
    return model

if __name__ == "__main__":
    inference(train_valid_test_datasets_provider,
             model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
    args, timers = init_megatron(args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
    print(setup_model_and_optimizer(
    model_provider, ModelType.encoder_or_decoder)[0])
    
    
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    sentence = "The quick brown fox jumps over the lazy dog."
    tokens = tokenizer(sentence, return_tensors="pt")
    print()
