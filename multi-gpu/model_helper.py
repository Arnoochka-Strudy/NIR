import argparse
import json
import deepspeed.comm
from transformers import (AutoConfig, AutoTokenizer, OPTForCausalLM)
from transformers.deepspeed import HfDeepSpeedConfig
from torch import nn, float16
from utils import GB
import deepspeed
from deepspeed.accelerator import get_accelerator
import gc

class Configurator:
    def __init__(self, args = None, config = None):
        if config is None:
            if args is None:
                args = Configurator.get_args()
            
            config = Configurator.get_config(args=args)
         
        self.args = args
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

        parser.add_argument("--use_zero", action="store_true", help="use ZeRO-inference")
        parser.add_argument("--use_aio", action="store_true", help="use AIO")
        parser.add_argument("--use_tp_parallel", action="store_true", help="use tensor parallelism")
        parser.add_argument("--zero_config", type=str, default="not", help="zero configuration")
        parser.add_argument("--aio_config", type=str, default="not", help="aio configuration")
        parser.add_argument("--tp_config", type=str, default="not", help="tensor parallel configuration")
        
        parser.add_argument("--pin-memory", action="store_true", help="whether to pinned CPU memory for ZeRO offloading")
        parser.add_argument("--cpu-offload", action="store_true", help="Use cpu offload.")
        parser.add_argument("--disk-offload", action="store_true", help="Use disk offload.")
        parser.add_argument("--kv-offload", action="store_true", help="Use kv cache cpu offloading.")
        parser.add_argument("--buffer_count", type=int, default=2, help="count buffer for disk offload")
        parser.add_argument("--buffer_size", type=float, default=0.5, help="buffer size in GB for disk offload")

        parser.add_argument("--offload-dir", type=str, default="~/offload_dir", help="Directory to store offloaded cache.")
        parser.add_argument("--pin_kv_cache", action="store_true", help="Allocate kv cache in pinned memory for offloading.")
        parser.add_argument("--async_kv_offload", action="store_true", help="Using non_blocking copy for kv cache offloading.")
        parser.add_argument("--use_gds", action="store_true", help="Use NVIDIA GPU DirectStorage to transfer between NVMe and GPU.")

        parser.add_argument("--log-file", type=str, default="auto", help="log file name")
        parser.add_argument("--verbose", type=int, default=2, help="verbose level")
        
        parser.add_argument("--quant_bits", type=int, default=16, help="model weight quantization bits; either 4 or 8")
        parser.add_argument("--quant_group_size", type=int, default=64, help="model weight quantization group size")
        
        parser.add_argument("--half-precision", type=str, default="not", help="use fp16 of bf16 to acclelerate calculations")
        
        parser.add_argument("--output-file", type=str, default="auto", help="write results inference")
        
        args = parser.parse_args()

        return args
    
    @staticmethod
    def get_config(args = None, filedir = None):
        
        
        if args is None:
            if filedir is None: 
                return {}
            
            with open(filedir, 'r', encoding='utf-8') as file:
                return json.load(file)
         
        config = Configurator.get_model_config(args.model) 
        config = {
            "fp16": {
                "enabled": args.half_precision == "fp16"
            },
            "bp16": {
                "enabled": args.half_precision == "bp16"
            },
            "steps_per_print": 2000,
            "train_batch_size": args.batch_size,
            "wall_clock_breakdown": False,
        }
        
        if args.use_zero:
            config = Configurator.__zero_config(args, config)
        if args.use_aio:
            config = Configurator.__aio_config(args, config)
        if args.use_tp_parallel and deepspeed.comm.get_world_size() > 1:
            config = Configurator.__tp_config(args, config)
            
        
        return config
            
    
    @staticmethod
    def __zero_config(args, config):
        if args.zero_config != "not":
            config["zero_optimization"] = json.load(args.zero_zonfig)
            return config
        
        hidden_size = Configurator.get_model_config(args.model).hidden_size
        config["zero_optimization"] = {
            "stage": 3,
            "stage3_prefetch_bucket_size": 2 * hidden_size * hidden_size, 
            "stage3_param_persistence_threshold": hidden_size,
            "stage3_max_live_parameters": 2 * hidden_size * hidden_size,
            }
    
        if args.cpu_offload:
            config["zero_optimization"]["offload_param"] = dict(
                device="cpu", pin_memory=args.pin_memory
            )
        if args.disk_offload:
            config["zero_optimization"]["offload_param"] = dict(
                device="nvme",
                pin_memory=args.pin_memory,
                nvme_path=args.offload_dir,
                buffer_count=args.buffer_count,
                buffer_size=int(args.buffer_size * GB)
            )
            
        return config
    
    @staticmethod
    def __aio_config(args, config):
        if args.aio_config != "not":
            config["aio"] = json.load(args.aio_config)
            return config
        
        config["aio"] = {
            "enabled": True,
            "block_size": 1048576*16,
            "queue_depth": 64,
            "thread_count": 8,
            "use_gds": False,
            "single_submit": False,
            "overlap_events": True,
        }  
        
        return config  
    
    @staticmethod
    def __tp_config(args, config):
        if args.tp_config != "not":
            config["tensor_parallel"] = json.load(args.tp_config)
            return config
        world_size = deepspeed.comm.get_world_size()
        config["tensor_parallel"] = {
            "tp_size": world_size
        }
        
        return config
                
        
            
    @staticmethod      
    def write_config(filedir, config):
        with open(filedir, 'w', encoding='utf-8') as file:
            json.dump(config, file, ensure_ascii=False, indent=4)
    
    @staticmethod        
    def get_model_config(model_name): # only opt
        return AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    
    
    

class ModelGetter:
    
    @staticmethod
    def get_model(configurator: Configurator) -> nn.Module:
        args = configurator.args
        ds_config = configurator.config
        Configurator.write_config("config.json", ds_config)
        dschf = HfDeepSpeedConfig(
            ds_config
        )
        get_accelerator().empty_cache()
        gc.collect()

        model = ModelGetter.get_pretrained_model(args.model, float16)
        model = model.eval()
        ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
    
        ds_engine.module.eval()
        model = ds_engine.module
        
        if configurator.args.kv_offload:
            model.set_kv_cache_offload(True, args.gen_len, args.pin_kv_cache, args.async_kv_offload)
        
        return model
    
    @staticmethod
    def get_tokenizer(model_name): # only opt
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token

        return tokenizer
    
    @staticmethod
    def get_pretrained_model(model_name, dtype) -> nn.Module: # only opt
        return OPTForCausalLM.from_pretrained(model_name, torch_dtype=dtype)