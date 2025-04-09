import argparse
import json
from transformers import (AutoConfig, AutoTokenizer, OPTForCausalLM)
from torch.nn import Module

class ModelHelper:
    def __init__(self, args = None, config = None):
        if config is None:
            if args is None:
                args = ModelHelper.get_args()
                
            config = ModelHelper.get_config(args=args)
            
        self.config = config
    
    @staticmethod
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
        parser.add_argument("--kv-offload", action="store_true", help="Use kv cache cpu offloading.")

        parser.add_argument("--offload-dir", type=str, default="~/offload_dir", help="Directory to store offloaded cache.")
        parser.add_argument("--pin_kv_cache", action="store_true", help="Allocate kv cache in pinned memory for offloading.")
        parser.add_argument("--async_kv_offload", action="store_true", help="Using non_blocking copy for kv cache offloading.")
        parser.add_argument("--use_gds", action="store_true", help="Use NVIDIA GPU DirectStorage to transfer between NVMe and GPU.")

        parser.add_argument("--log-file", type=str, default="auto", help="log file name")
        parser.add_argument("--verbose", type=int, default=2, help="verbose level")

        parser.add_argument("--quant_bits", type=int, default=16, help="model weight quantization bits; either 4 or 8")
        parser.add_argument("--quant_group_size", type=int, default=64, help="model weight quantization group size")
        
        args = parser.parse_args()

        return args
    
    @staticmethod
    def get_config(self, args = None, filedir = None):
        if args is None:
            if filedir is None: 
                return None
            
            with open(filedir, 'r', encoding='utf-8') as file:
                return json.load(file)
            
        pass
            
    @staticmethod     
    def write_config(filedir, config):
        with open(filedir, 'w', encoding='utf-8') as file:
            json.dump(config, file, ensure_ascii=False, indent=4)
            
    def get_model_config(model_name): # only opt
        return AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    
    def get_model() -> Module:
        pass
    
    def __get_tokenizer(self, model_name): # only opt
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def __get_pretrained_model(self, model_name, dtype) -> Module: # only opt
        return OPTForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
