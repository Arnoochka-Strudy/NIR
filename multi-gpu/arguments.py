import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/opt-1.3b", help="model name or path; currently only supports OPT and BLOOM models")
    parser.add_argument("--local_rank", type=int, help="local rank for distributed inference")
    parser.add_argument("--loops", type=int, default=3,  help="Number of token generation iterations")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--prompt-len", type=int, default=512,  help="prompt length")
    parser.add_argument("--gen-len", type=int, default=32,  help="number of tokens to generate")
    parser.add_argument("--pin-memory", type=int, default=0, help="whether to pinned CPU memory for ZeRO offloading")
    parser.add_argument("--cpu-offload", action="store_true", help="Use cpu offload.")
    parser.add_argument("--disk-offload", action="store_true", help="Use disk offload.")
    parser.add_argument("--offload-dir", type=str, default="~/offload_dir", help="Directory to store offloaded cache.")
    parser.add_argument("--kv-offload", action="store_true", help="Use kv cache cpu offloading.")
    parser.add_argument("--log-file", type=str, default="auto", help="log file name")
    parser.add_argument("--verbose", type=int, default=2, help="verbose level")
    parser.add_argument("--quant_bits", type=int, default=16, help="model weight quantization bits; either 4 or 8")
    parser.add_argument("--quant_group_size", type=int, default=64, help="model weight quantization group size")
    parser.add_argument("--pin_kv_cache", action="store_true", help="Allocate kv cache in pinned memory for offloading.")
    parser.add_argument("--async_kv_offload", action="store_true", help="Using non_blocking copy for kv cache offloading.")
    parser.add_argument("--use_gds", action="store_true", help="Use NVIDIA GPU DirectStorage to transfer between NVMe and GPU.")
    args = parser.parse_args()
    
    return args
