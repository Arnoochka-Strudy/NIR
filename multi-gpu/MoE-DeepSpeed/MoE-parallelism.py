import torch
import torch.nn as nn
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertAttention, BertConfig
from deepspeed.moe.layer import MoE
import deepspeed
import argparse

# === ПАРАМЕТРЫ МОДЕЛИ ===
HIDDEN_SIZE = 1024
NUM_HEADS = 16
NUM_LAYERS = 24
VOCAB_SIZE = 50272
SEQ_LEN = 512
NUM_EXPERTS = 16
USE_TUTEL = False
EP_SIZE=2

# === ТРАНСФОРМЕРНЫЙ БЛОК С MOE ===
class MoETransformerBlock(nn.Module):
    def __init__(self, hidden_size=HIDDEN_SIZE, num_heads=NUM_HEADS,
                 ffn_size=HIDDEN_SIZE, num_experts=NUM_EXPERTS, ep_size=EP_SIZE):
        super().__init__()
        self.attn = BertAttention(BertConfig(hidden_size=hidden_size, num_attention_heads=num_heads))
        self.moe = MoE(
            hidden_size=hidden_size,
            expert=nn.Linear(hidden_size, ffn_size),
            num_experts=num_experts,
            ep_size=ep_size,
            k=1,
            use_tutel=False
        )

        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        attn_out = self.attn(x)[0]
        x = self.ln1(x + attn_out)
        moe_out, _, _ = self.moe(x)
        x = self.ln2(x + moe_out)

        return x


# === ЯЗЫКОВАЯ МОДЕЛЬ С MOE ===
class MoETransformerForLM(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, hidden_size=HIDDEN_SIZE,
                 num_layers=NUM_LAYERS, seq_len=SEQ_LEN):
        super().__init__()
        self.seq_len = seq_len
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            MoETransformerBlock(hidden_size=hidden_size, num_experts=NUM_EXPERTS, ep_size=EP_SIZE)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.unembed = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.unembed(x)
        return logits

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank passed by DeepSpeed")
    args = parser.parse_args()
    deepspeed.init_distributed(dist_backend="nccl")
    device = torch.device("cuda", args.local_rank)
    torch.cuda.set_device(device)

    model = MoETransformerForLM()
    model.to(device)

    ds_config = {
        "moe": {
            "enabled": True,
            "num_experts": [NUM_EXPERTS],
            "ep_size": EP_SIZE,
        },
        "dtype": torch.float
    }

    engine = deepspeed.init_inference(
        model=model,
        config=ds_config
    )

    engine.eval()

    print(f"\n[Rank {args.local_rank}] Model device: {next(model.parameters()).device}")
    for name, param in model.named_parameters():
        print(f"{name} is on {param.device}")

    input_ids = torch.randint(
        low=0,
        high=VOCAB_SIZE,
        size=(1, SEQ_LEN),
        dtype=torch.long 
    ).to(device)

    with torch.no_grad():
        output = engine(input_ids)
        print(f"[Rank {args.local_rank}] Output shape: {output.shape}")

if __name__ == "__main__":
    main()