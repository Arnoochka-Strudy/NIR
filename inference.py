import argparse
import torch
from transformers import AutoTokenizer
from megatron import initialize_megatron
from megatron.model import GPTModel
from megatron.checkpointing import load_checkpoint
from megatron.utils import print_rank_0

def parse_args():
    parser = argparse.ArgumentParser(description="Megatron Inference")
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--tensor-model-parallel-size', type=int, default=2, help="Number of tensor model parallel GPUs")
    parser.add_argument('--load', type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument('--inference-engine', type=str, default="normal", help="Inference engine to use (normal/deepspeed)")
    parser.add_argument('--deepspeed_config', type=str, required=False, help="Path to DeepSpeed configuration file")
    parser.add_argument('--model', type=str, default="opt", help="Model type (e.g. opt, gpt2, etc.)")
    return parser.parse_args()

def setup_model_and_inference(args):
    # Initialize Megatron
    initialize_megatron(args_defaults=args)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")

    # Load the model
    model = GPTModel.from_pretrained(args.load)

    # Load checkpoint
    checkpoint_path = args.load
    load_checkpoint(model, checkpoint_path)

    # Move model to GPU
    model = model.to(torch.device("cuda"))

    return model, tokenizer

def generate_inference(model, tokenizer, input_text, max_length=50):
    # Tokenize input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(torch.device("cuda"))

    # Generate text
    with torch.no_grad():
        output = model.generate(input_ids, max_length=max_length)

    # Decode the output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

def main():
    args = parse_args()

    # Setup model and tokenizer
    model, tokenizer = setup_model_and_inference(args)

    # Perform inference (you can modify the input text)
    input_text = "Once upon a time"
    output_text = generate_inference(model, tokenizer, input_text)

    print(f"Generated text: {output_text}")

if __name__ == "__main__":
    main()
