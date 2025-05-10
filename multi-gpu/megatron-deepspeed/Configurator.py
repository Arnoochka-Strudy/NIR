import argparse
import json
from utils import GB

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
        
        parser.add_argument("--hidden-size", type=int, help="hidden size model")
        
        parser.add_argument("--local_rank", type=int, help="local rank for distributed inference")

        parser.add_argument("--batch-size", type=int, default=1, help="batch size")
        parser.add_argument("--prompt-len", type=int, default=512,  help="prompt length")

        parser.add_argument("--use-zero", action="store_true", help="use ZeRO-inference")
        parser.add_argument("--use-aio", action="store_true", help="use AIO")
        parser.add_argument("--use-quant", action="store_true", help="Using model weight quantization bits")
        
        parser.add_argument("--pin-memory", action="store_true", help="whether to pinned CPU memory for ZeRO offloading")
        parser.add_argument("--cpu-offload", action="store_true", help="Use cpu offload.")
        parser.add_argument("--disk-offload", action="store_true", help="Use disk offload.")
        parser.add_argument("--buffer-count", type=int, default=2, help="count buffer for disk offload")
        parser.add_argument("--buffer-size", type=float, default=0.5, help="buffer size in GB for disk offload")

        parser.add_argument("--offload-dir", type=str, default="~/offload_dir", help="Directory to store offloaded cache.")
        parser.add_argument("--use-gds", action="store_true", help="Use NVIDIA GPU DirectStorage to transfer between NVMe and GPU.")

        parser.add_argument("--quant-bits", type=int, default=16, help="model weight quantization bits; either 4 or 8")
        parser.add_argument("--quant-group-size", type=int, default=64, help="model weight quantization group size")
        
        parser.add_argument("--half-precision", type=str, default="not", help="use fp16 of bf16 to acclelerate calculations")
        
        args = parser.parse_args()

        return args
    
    @staticmethod
    def get_config(args = None, filedir = None):
        
        
        if args is None:
            if filedir is None: 
                return {}
            
            with open(filedir, 'r', encoding='utf-8') as file:
                return json.load(file)
        config = {
            "fp16": {
                "enabled": args.half_precision == "fp16"
            },
            "bp16": {
                "enabled": args.half_precision == "bp16"
            },
            "steps_per_print": 2000,
            "train_batch_size": args.batch_size,
            "train_micro_batch_size_per_gpu": args.batch_size,
            "wall_clock_breakdown": False,
        }
        
        if args.use_zero:
            config = Configurator.__zero_config(args, config)
        if args.use_aio:
            config = Configurator.__aio_config(args, config)
        if args.use_quant:
            config = Configurator.__quant_config(args, config)
            
        
        return config
            
    
    @staticmethod
    def __zero_config(args, config):
        hidden_size = args.hidden_size
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
        config["aio"] = {
            "enabled": True,
            "block_size": 1048576*16,
            "queue_depth": 64,
            "thread_count": 8,
            "use_gds": args.use_gds,
            "single_submit": False,
            "overlap_events": True,
        }  
        
        return config  
    
    @staticmethod
    def __quant_config(args, config):
        quant_config = {
            'weight_quantization': {
                'quantized_initialization' : {
                    'num_bits': args.quant_bits,
                    'group_size': args.quant_group_size,
                    "group_dim": 1,
                    "symmetric": False
                }
            }
        }
        config['weight_quantization'] = quant_config
        return config
                
        
            
    @staticmethod      
    def write_config(filedir, config):
        with open(filedir, 'w', encoding='utf-8') as file:
            json.dump(config, file, ensure_ascii=False, indent=4)  
            
def main():
    args = Configurator.get_args()
    config = Configurator.get_config(args=args)
    Configurator.write_config("ds_config.json", config=config)
    
if __name__ == "__main__":
    main()
    