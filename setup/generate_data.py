import torch
from transformers import AutoTokenizer

model_id = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# A long string of text to chop up (you can replace this with a real Wikipedia article)
dummy_text = "The quick brown fox jumps over the lazy dog. " * 500

# Tokenize the text
full_tokens = tokenizer(dummy_text, return_tensors="pt").input_ids

# Chop into exact lengths and save
sequence_lengths = [128, 256, 512, 1024]

for length in sequence_lengths:
    # Slice the tensor to the exact length
    exact_tensor = full_tokens[:, :length]
    torch.save(exact_tensor, f"prompt_{length}.pt")
    print(f"Saved prompt_{length}.pt with shape {exact_tensor.shape}")
