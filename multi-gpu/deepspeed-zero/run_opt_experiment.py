import torch
import deepspeed
from deepspeed.accelerator import get_accelerator
from timer import timers
from utils import (GB, add_model_hooks, remove_model_hooks, cache_bytes,
                   get_filename, get_quant_config, hidden_bytes,
                   model_bytes, write_benchmark_log, get_prompts)
from packaging import version
from model_helper import Configurator, ModelGetter

assert version.parse(deepspeed.__version__) >= version.parse("0.10.3"), "ZeRO-Inference with weight quantization and kv cache offloading is available only in DeepSpeed 0.10.3+, please upgrade DeepSpeed"


def run_generation(
    configurator: Configurator
):
    args = configurator.args
    config = Configurator.get_model_config(args.model)
    
    with torch.no_grad():
        model = ModelGetter.get_model(configurator)
        
    tokenizer = ModelGetter.get_tokenizer(args.model)
    
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape} (device: {param.device})")

    execute_gen_len = args.gen_len
    prompts = get_prompts("/home/victor/NIR/benchmark_mean.txt")

    def _batch_encode(prompts):
        input_tokens = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding="max_length", max_length=args.prompt_len)
        for t in input_tokens:
            if torch.is_tensor(input_tokens[t]):
                input_tokens[t] = input_tokens[t].to(torch.cuda.current_device())
        return input_tokens

    input_tokens = _batch_encode(prompts)


    add_model_hooks(model)

    def set_model_stage(model, stage):
        model.stage = stage

    print(f"benchmark, prompt_len = {args.prompt_len}, execute_gen_len = {execute_gen_len}, input_ids.shape = {input_tokens.input_ids.shape}")

    generate_kwargs = dict(max_new_tokens=execute_gen_len, do_sample=False)
    prefill_timings = []
    timer = timers("generate-forward")
    for _ in range(args.loops):
        timer.start(sync_func=get_accelerator().synchronize)
        with torch.no_grad():
            set_model_stage(model, "prefill")
            output_ids = model.generate(**input_tokens, **generate_kwargs)
            prefill_timings.append(model.__duration__)
        timer.stop(sync_func=get_accelerator().synchronize)
    costs = timers("generate-forward").costs

    if args.local_rank != 0:
        return

    # Log output
    print(f"Summary:")
    print(f"costs = {costs}, prefill_timings = {prefill_timings}")
    total_latency = costs[-1]
    prefill_latency = prefill_timings[-1]
    remove_model_hooks(model)

    prefill_throughput = args.batch_size * args.prompt_len / prefill_latency
    decode_latency = total_latency - prefill_latency
    decode_throughput = args.batch_size * (args.gen_len - 1) / max(decode_latency, 1e-10)
    num_generated_tokens = args.batch_size * args.gen_len
    total_throughput = num_generated_tokens / total_latency
    gpu_peak_mem = get_accelerator().max_memory_allocated(torch.device("cuda"))
    out_str = ""

    if args.verbose >= 2:
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        show_str = "Outputs:\n" + 70 * "-" + "\n"
        for i in [0, (len(outputs) - 1) // 2, len(outputs) - 1]:
            show_str += f"{i}: {outputs[i]}\n"
            show_str += 70 * "-" + "\n"
        print(show_str)

        # Check lengths
        input_lens = [len(x) for x in input_tokens.input_ids]
        output_lens = [len(x) for x in output_ids]
        assert all(x == args.prompt_len for x in input_lens)
        assert all(x == args.prompt_len + execute_gen_len for x in output_lens)

    if args.output_file == "auto":
        filename = (
            get_filename(
                args.model,
                args.batch_size,
                args.prompt_len,
                args.gen_len,
                args.cpu_offload,
                args.disk_offload,
                args.kv_offload,
                args.quant_bits != 16,
            )
            + ".log"
        )
    else:
        filename = args.output_file

    cache_size = cache_bytes(config, args.batch_size, args.prompt_len + args.gen_len)
    hidden_size = hidden_bytes(config, args.batch_size, args.prompt_len + args.gen_len)
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
    if args.verbose >= 1:
        print(log_str)
        
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    deepspeed.init_distributed("nccl") 
    configurator = Configurator()

    run_generation(
        configurator
    )