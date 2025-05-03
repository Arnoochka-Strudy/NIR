from transformers import (AutoTokenizer, OPTForCausalLM)
import deepspeed
from deepspeed.pipe import PipelineModule
import torch.nn as nn
import torch

deepspeed.init_distributed("nccl") 

def get_prompts(filename: str | None = None) -> list[str]:
    
    if filename is None:
        return ["Paris is the capital city of"] * 4
    
    with open(filename, 'r') as file:
        promts = [line for line in file]
        return promts

# Загрузка модели
model_name = "facebook/opt-1.3b"
model = OPTForCausalLM.from_pretrained(model_name)

# 1. Извлекаем слои OPT
layers = [
    model.model.decoder.layers[i] 
    for i in range(len(model.model.decoder.layers))
]

# 2. Добавляем входной и выходной слои
layers = [
    model.model.decoder.embed_tokens,  # Входной слой (эмбеддинги)
    *layers,
    model.model.decoder.final_layer_norm,  # Нормализация
    model.lm_head  # Выходной слой (классификатор)
]

# 3. Создаём PipelineModule
model = PipelineModule(
    layers=layers,
    num_stages=2,  # Должно совпадать с "pipe_parallel_size" в конфиге
)

# 4. Инициализация DeepSpeed
model_engine = deepspeed.initialize(
    model=model,
    config="ds_config.json"
)[0]

prompts = get_prompts("/home/victor/NIR/benchmark_mean.txt")

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token

def _batch_encode(prompts):
    input_tokens = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding="max_length", max_length=args.prompt_len)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to(torch.cuda.current_device())
    return input_tokens

input_tokens = _batch_encode(prompts)
generate_kwargs = dict(max_new_tokens=32, do_sample=False)

output_ids = model.generate(**input_tokens, **generate_kwargs)

print(output_ids)


