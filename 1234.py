import os
import torch
from transformers import AutoModelForCausalLM

def split_tensor(tensor, num_shards, dim=0):
    return torch.chunk(tensor, num_shards, dim=dim)

def convert(model_name: str, num_shards: int, output_dir: str):
    print(f"Loading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    state_dict = model.state_dict()

    print("Sharding weights...")
    shard_dicts = [{} for _ in range(num_shards)]

    for name, tensor in state_dict.items():
        if tensor.ndim >= 2 and tensor.shape[0] % num_shards == 0:
            shards = split_tensor(tensor, num_shards, dim=0)
            for i in range(num_shards):
                shard_dicts[i][name] = shards[i].clone()
        else:
            for i in range(num_shards):
                shard_dicts[i][name] = tensor.clone()

    print("Saving shards...")
    for rank in range(num_shards):
        rank_dir = os.path.join(output_dir, f"mp_rank_{rank:02d}")
        os.makedirs(rank_dir, exist_ok=True)
        torch.save({'module': shard_dicts[rank]}, os.path.join(rank_dir, "model_optim_rng.pt"))
    
    print(f"✅ Conversion complete. Files saved to: {output_dir}")

if __name__ == "__main__":
    # Пример запуска
    convert("facebook/opt-125m", num_shards=2, output_dir="checkpoints/global_step1")
