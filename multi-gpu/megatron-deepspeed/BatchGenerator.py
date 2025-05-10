import torch

class BatchGenerator:
    @staticmethod  
    def encode_batch_from_sentences(sentences, tokenizer, seq_length, device='cuda'):
        encoded = [tokenizer.tokenize(s) for s in sentences]

        tokens = []
        for ids in encoded:
            if len(ids) >= seq_length:
                tokens.append(ids[:seq_length])
            else:
                pad_len = seq_length - len(ids)
                tokens.append([tokenizer.eod] * pad_len + ids)

        tokens_tensor = torch.tensor(tokens, dtype=torch.long, device=device)
        return {'text': tokens_tensor}
    
    @staticmethod
    def batch_generator(data, batch_size, seq_length, tokenizer, device = 'cuda'):
        for i in range(0, len(data), batch_size):
            yield BatchGenerator.encode_batch_from_sentences(data[i:min(i + batch_size, len(data))],
                                                             tokenizer,
                                                             seq_length,
                                                             device=device) 