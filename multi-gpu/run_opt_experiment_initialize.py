from arguments import get_args
import gc
import multiprocessing as mp
import os
import torch
import deepspeed
from deepspeed.accelerator import get_accelerator
import deepspeed.comm as dist
from timer import timers
from transformers import (AutoConfig, AutoTokenizer, OPTForCausalLM,
                        )
from transformers.deepspeed import HfDeepSpeedConfig
from utils import (GB, add_model_hooks, cache_bytes,
                   get_filename, get_quant_config, hidden_bytes,
                   model_bytes, write_benchmark_log)
from packaging import version

assert version.parse(deepspeed.__version__) >= version.parse("0.10.3"), "ZeRO-Inference with weight quantization and kv cache offloading is available only in DeepSpeed 0.10.3+, please upgrade DeepSpeed"

def get_tokenizer(model_name): # only opt
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

def get_model_config(model_name): # only opt
    print("get model config")
    return AutoConfig.from_pretrained(model_name, trust_remote_code=True)

def get_model(model_name, dtype): # only opt
    return OPTForCausalLM.from_pretrained(model_name, torch_dtype=dtype)

def get_ds_model(
    model_name,
    world_size,
    cpu_offload,
    disk_offload,
    offload_dir,
    bits,
    group_size,
    batch_size,
    pin_memory
):

    config = get_model_config(model_name)
    hidden_size = config.hidden_size

    if getattr(config, 'torch_dtype', None) is None:
        dtype = torch.float16
    else:
        dtype = config.torch_dtype

    ds_config = {
        "fp16": {
            "enabled": dtype == torch.float16,
        },
        "bf16": {
            "enabled": dtype == torch.bfloat16,
        },
        "zero_optimization": {
            "stage": 3,
            "stage3_prefetch_bucket_size": 2 * hidden_size * hidden_size, 
            "stage3_param_persistence_threshold": hidden_size,
            "stage3_max_live_parameters": 2 * hidden_size * hidden_size,
        },
        "steps_per_print": 2000,
        "train_batch_size": batch_size,
        "wall_clock_breakdown": False,
    }

    if bits == 4:
        quant_config = get_quant_config(config, bits=bits, group_size=group_size)
        ds_config.update(quant_config)
    if cpu_offload:
        ds_config["zero_optimization"]["offload_param"] = dict(
            device="cpu", pin_memory=pin_memory
        )

    if disk_offload:
        buffer_count = 2
        buffer_size = int(0.65*GB)

        ds_config["zero_optimization"]["offload_param"] = dict(
            device="nvme",
            pin_memory=pin_memory,
            nvme_path=offload_dir,
            buffer_count=buffer_count,
            buffer_size=buffer_size
        )
        
        ds_config["aio"] = {
            "block_size": 1048576*16,
            "queue_depth": 64,
            "thread_count": 8,
            "use_gds": True,
            "single_submit": False,
            "overlap_events": True,
        }
        
    ds_config["tensor_parallel"] = {
        "tensor_parallel": {
            "tp_size": world_size
        }
    }
    print(ds_config)
    dschf = HfDeepSpeedConfig(
        ds_config
    )  # this tells from_pretrained to instantiate directly on gpus

    # clear cache / free memory
    get_accelerator().empty_cache()
    gc.collect()

    model = get_model(model_name, dtype)
    model = model.eval()
    ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
    
    ds_engine.module.eval()
    model = ds_engine.module
    print(f"model.config = {model.config}")

    return model


def run_generation(
    model_name,
    world_size,
    local_rank,
    batch_size,
    prompt_len,
    gen_len,
    cpu_offload,
    disk_offload,
    offload_dir,
    output_file,
    verbose,
    kv_offload,
    quant_bits,
    quant_group_size,
    pin_kv_cache,
    async_kv_offload,
    loops,
    pin_memory
):
    # Load tokenizer
    config = get_model_config(model_name)    

    tokenizer = get_tokenizer(model_name)
        
    print("load model")
    with torch.no_grad():
        model = get_ds_model(
            model_name,
            world_size,
            cpu_offload,
            disk_offload,
            offload_dir,
            quant_bits,
            quant_group_size,
            batch_size,
            bool(pin_memory)
        )

    # Run generation
    print("run generation")
    execute_gen_len = gen_len
    prompts = ["Paris is the capital city of"] * batch_size

    def _batch_encode(prompts):
        input_tokens = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding="max_length", max_length=prompt_len)
        for t in input_tokens:
            if torch.is_tensor(input_tokens[t]):
                input_tokens[t] = input_tokens[t].to(torch.cuda.current_device())
        return input_tokens

    input_tokens = _batch_encode(prompts)

    if kv_offload:
        model.set_kv_cache_offload(True, gen_len, pin_kv_cache, async_kv_offload)

    print(model, model.config)


    add_model_hooks(model)

    def set_model_stage(model, stage):
        model.stage = stage

    print(f"benchmark, prompt_len = {prompt_len}, execute_gen_len = {execute_gen_len}, input_ids.shape = {input_tokens.input_ids.shape}")

    generate_kwargs = dict(max_new_tokens=execute_gen_len, do_sample=False)
    prefill_timings = []
    timer = timers("generate-forward")
    for _ in range(loops):
        timer.start(sync_func=get_accelerator().synchronize)
        with torch.no_grad():
            set_model_stage(model, "prefill")
            output_ids = model.generate(**input_tokens, **generate_kwargs)
            prefill_timings.append(model.__duration__)
        timer.stop(sync_func=get_accelerator().synchronize)
    costs = timers("generate-forward").costs

    if args.local_rank != 0:
        return

    def remove_model_hooks(module):
        if hasattr(module, "__start_time_hook_handle__"):
            module.__start_time_hook_handle__.remove()
            del module.__start_time_hook_handle__
        if hasattr(module, "__end_time_hook_handle__"):
            module.__end_time_hook_handle__.remove()
            del module.__end_time_hook_handle__
        if hasattr(module, "stage"):
            del module.stage
        if hasattr(module, "__duration__"):
            del module.__duration__

    # Log output
    print(f"Summary:")
    print(f"costs = {costs}, prefill_timings = {prefill_timings}")
    total_latency = costs[-1]
    prefill_latency = prefill_timings[-1]
    remove_model_hooks(model)

    prefill_throughput = batch_size * prompt_len / prefill_latency
    decode_latency = total_latency - prefill_latency
    decode_throughput = batch_size * (gen_len - 1) / max(decode_latency, 1e-10)
    num_generated_tokens = batch_size * gen_len
    total_throughput = num_generated_tokens / total_latency
    gpu_peak_mem = get_accelerator().max_memory_allocated(torch.device("cuda"))
    out_str = ""

    if verbose >= 2:
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        show_str = "Outputs:\n" + 70 * "-" + "\n"
        for i in [0, (len(outputs) - 1) // 2, len(outputs) - 1]:
            show_str += f"{i}: {outputs[i]}\n"
            show_str += 70 * "-" + "\n"
        print(show_str)

        # Check lengths
        input_lens = [len(x) for x in input_tokens.input_ids]
        output_lens = [len(x) for x in output_ids]
        assert all(x == prompt_len for x in input_lens)
        assert all(x == prompt_len + execute_gen_len for x in output_lens)

    if output_file == "auto":
        filename = (
            get_filename(
                model_name,
                batch_size,
                prompt_len,
                gen_len,
                cpu_offload,
                disk_offload,
                kv_offload,
                quant_bits != 16,
            )
            + ".log"
        )
    else:
        filename = output_file

    cache_size = cache_bytes(config, batch_size, prompt_len + gen_len)
    hidden_size = hidden_bytes(config, batch_size, prompt_len + gen_len)
    log_str = write_benchmark_log(
        filename,
        model_bytes(config),
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
    if verbose >= 1:
        print(log_str)
        
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    args = get_args()

    deepspeed.init_distributed("nccl") 
    run_generation(
        model_name=args.model,
        world_size=dist.get_world_size(),
        local_rank=args.local_rank,
        batch_size=args.batch_size,
        prompt_len=args.prompt_len,
        gen_len=args.gen_len,
        cpu_offload=args.cpu_offload,
        disk_offload=args.disk_offload,
        offload_dir=os.path.abspath(os.path.expanduser(args.offload_dir)),
        output_file=args.log_file,
        verbose=args.verbose,
        kv_offload=args.kv_offload,
        quant_bits=args.quant_bits,
        quant_group_size=args.quant_group_size,
        pin_kv_cache=args.pin_kv_cache,
        async_kv_offload=args.async_kv_offload,
        loops=args.loops,
        pin_memory=args.pin_memory
    )