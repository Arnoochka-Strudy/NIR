import torch
from copy import deepcopy
from torch.types import Tensor

class BatchGenerator:
    @staticmethod  
    def encode_batch_from_sentences(sentences, tokenizer, seq_length) -> Tensor:
        encoded = [tokenizer.tokenize(s) for s in sentences]

        tokens = []
        for ids in encoded:
            if len(ids) >= seq_length:
                tokens.append(ids[:seq_length])
            else:
                pad_len = seq_length - len(ids)
                tokens.append([tokenizer.eod] * pad_len + ids)

        tokens_tensor = torch.tensor(tokens, dtype=torch.long, device='cpu')
        return tokens_tensor
    
    @staticmethod
    def batch_generator(data, batch_size, seq_length, tokenizer, device = 'cuda'):
        data_with_batches = [
            BatchGenerator.encode_batch_from_sentences(
                [promt for _ in range(batch_size)], tokenizer, seq_length
            )
            for promt in data
        ]
        for i in range(0, len(data), 1): 
            batch = {"text": data_with_batches[i].to(device)}
            yield batch