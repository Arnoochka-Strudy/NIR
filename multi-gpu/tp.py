import os
import deepspeed
import torch
from transformers import pipeline
from arguments import get_args

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))
args = get_args()
print(local_rank, world_size)
generator = pipeline('text-generation', model='gpt2',
                     device=local_rank)

generator.model = deepspeed.init_inference(generator.model,
                                           tensor_parallel={"tp_size": world_size},
                                           dtype=torch.float,
                                           replace_with_kernel_inject=True)

string = generator("DeepSpeed is", do_sample=True, min_length=50)
if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 1:
    print(string)