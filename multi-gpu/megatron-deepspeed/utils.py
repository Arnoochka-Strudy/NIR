import torch
import time
from argparse import ArgumentParser

KB = 1 << 10
MB = 1 << 20
GB = 1 << 30
T = 1e12

def model_bytes(config):
    h = config.hidden_size
    v = config.vocab_size
    n = config.num_layers
    return 	2 * (n * (
    # config-attention
    h * (3 * h + 1) + h * (h + 1) +
    # mlp
    h * (4 * h + 1) + h * 4 * (h + 1) +
    # layer norm
    h * 4) +
    # embedding
    v * (h + 1))
        

def cache_bytes(config, batch_size, seq_len):
    return 2 * batch_size * seq_len * config.num_layers * config.hidden_size * 2

def hidden_bytes(config, batch_size, seq_len):
    return batch_size * seq_len * config.hidden_size * 4


def write_benchmark_log(args, filename, model_size, cache_size, hidden_size,
        gpu_peak_mem, prefill_latency, prefill_throughput,
        decode_latency, decode_throughput, total_latency, total_throughput):
    log_str = (f"GPT model:\n"
               f" num layers: {args.num_layers}\n"
               f" hidden size: {args.hidden_size}\n"
               f" seq length: {args.seq_length}\n"
               f" ffn hidden size: {args.ffn_hidden_size}\n"
               f" max position embeddings: {args.max_position_embeddings}\n"
               f" use zero: {args.use_zero}\n"
               f" num experts: {args.num_experts[0]}\n"
               f"Used Sizes:\n"
               f"model size: {model_size/GB:.3f} GB\t"
               f"cache size: {cache_size/GB:.3f} GB\t"
               f"hidden size (p): {hidden_size/GB:.3f} GB\n"
               f"peak gpu mem: {gpu_peak_mem / GB:.3f} GB\t"
               f"prefill latency: {prefill_latency:.3f} s\t"
               f"prefill throughput: {prefill_throughput:.3f} token/s\n"
               f"decode latency: {decode_latency:.3f} s\t"
               f"decode throughput: {decode_throughput:.3f} token/s\n"
               f"total latency: {total_latency:.3f} s\t"
               f"total throughput: {total_throughput:.3f} token/s\n")
    with open(filename, "a") as fout:
        fout.write(log_str + "\n")

    return log_str

def get_filename(model_name, batch_size, prompt_len, gen_len,
                 cpu_offload, disk_offload,
                 kv_offload, weight_quantize):
    simple_name = model_name.split('/')[-1]
    filename = "ds-"
    filename += f"{simple_name}-bs{batch_size}-prompt{prompt_len}-gen{gen_len}-"
    if cpu_offload:
        filename += "cpu-offload"
    elif disk_offload:
        filename += "disk-offload"
    else:
        filename += "no-offload"
    if kv_offload:
        filename += "-kv_offload"
    if weight_quantize:
        filename += "-w_quant"
        
    return filename

# add timing hooks
def add_model_hooks(model: torch.nn.Module):

    def start_time_hook(module, input):
        if hasattr(module, 'stage') and module.stage == "decode":
            return
        elif hasattr(module, 'stage') and module.stage == 'prefill':
            torch.cuda.synchronize()
            module.__start_time__ = time.time()

    def end_time_hook(module, input, output):
        if hasattr(module, 'stage') and module.stage == "decode":
            return
        elif hasattr(module, 'stage') and module.stage == 'prefill':
            torch.cuda.synchronize()
            module.__duration__ = time.time() - module.__start_time__
            module.stage = "decode"

    if not hasattr(model, '__start_time_hook_handle'):
        model.__start_time_hook_handle__ = model.register_forward_pre_hook(
            start_time_hook, )

    if not hasattr(model, '__end_time_hook_handle__'):
        model.__end_time_hook_handle__ = model.register_forward_hook(
            end_time_hook, )
        
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
        
        

def get_prompts(filename: str | None = None) -> list[str]:
    
    if filename is None:
        return ["Paris is the capital city of"] * 4
    
    with open(filename, 'r') as file:
        promts = [line for line in file]
        return promts
    
def get_inference_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("--data-dir", type=str, default="/home/victor/NIR/benchmark_mean.txt", help="directory of dataset")
    parser.add_argument("--loops", type=int, default=3, help="num repeat batch inference")
    parser.add_argument("--gen-len", type=int, default=1, help="number generate tokens")
    parser.add_argument("--verbose", type=int, default=1, help="verbose level")
    parser.add_argument("--output-file", type=str, default="results.log", help="write results inference")
    parser.add_argument("--use-zero", action="store_true", help="use ZeRO-inference")
    return parser