import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import argparse

from lora import do_fine_tune


# Initialize model
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
nn_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
print("Model loaded.")

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
    nn_model = nn_model.to("cuda")

# Fine-tune model
parser = argparse.ArgumentParser(description="Smart Medical Bot")
parser.add_argument("-n", "--no-tune", action= "store_true", help="Do not tune the model.")
args = parser.parse_args()
if not args.no_tune:
    print("Fine-tuning model...")
    do_fine_tune(nn_model, tokenizer)
    print("Fine-tuning complete.")


def prepare_input_ids(text: str, tokenizer: AutoTokenizer) -> torch.Tensor:
    input_ids = tokenizer.encode(text, return_tensors="pt")
    input_ids = input_ids.to(device)
    return input_ids


def infer(
    input_ids: torch.Tensor, max_length: int = 1000, num_return_sequences: int = 1
) -> str:
    input_ids = input_ids.to(device)
    output = nn_model.generate(
        input_ids, max_length=max_length, num_return_sequences=num_return_sequences
    )
    if device.type == "cuda":
        output = output.cpu()
    # Decode
    return tokenizer.decode(output[0], skip_special_tokens=True)


def inpput_to_output(input: str) -> str:
    input_ids = prepare_input_ids(input, tokenizer)
    output: str = infer(input_ids)
    output: str = output[len(input) :]
    output = output.split("$")[0]
    return output
