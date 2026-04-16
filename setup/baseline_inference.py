import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Llama-3.1-8B-Instruct"

print("Loading model and tokenizer (Baseline SDPA)...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load the model with standard SDPA attention (Baseline)
# We explicitly set attn_implementation="sdpa" to avoid flash_attention_2 for the baseline
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    device_map="auto",
    attn_implementation="sdpa" 
)

# Load the 1024-token synthetic prompt you just generated
print("Loading synthetic prompt...")
input_ids = torch.load("prompt_1024.pt").to("cuda")

print(f"Input shape: {input_ids.shape}")

# Warm-up run (PyTorch CUDA operations need a warm-up pass for accurate timing)
print("Performing CUDA warm-up...")
with torch.no_grad():
    _ = model(input_ids)
torch.cuda.synchronize()

# --- PREFILL PHASE (Time-To-First-Token) ---
print("\n--- Measuring Prefill (TTFT) ---")
start_time = time.perf_counter()

with torch.no_grad():
    outputs = model(input_ids, use_cache=True)
    
torch.cuda.synchronize()
prefill_time = time.perf_counter() - start_time
print(f"Prefill Time (TTFT) for 1024 tokens: {prefill_time:.4f} seconds")

# --- DECODE PHASE (Tokens / Sec) ---
print("\n--- Measuring Decode Throughput ---")
# Get the past_key_values (KV Cache) from the prefill phase
past_key_values = outputs.past_key_values
# Start with the last token generated from the prefill
next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)

decode_tokens = 20 # Number of tokens to generate for the test
start_time = time.perf_counter()

with torch.no_grad():
    for _ in range(decode_tokens):
        decode_outputs = model(
            next_token,
            past_key_values=past_key_values,
            use_cache=True
        )
        past_key_values = decode_outputs.past_key_values
        next_token = torch.argmax(decode_outputs.logits[:, -1, :], dim=-1).unsqueeze(0)

torch.cuda.synchronize()
decode_time = time.perf_counter() - start_time
throughput = decode_tokens / decode_time

print(f"Generated {decode_tokens} tokens in {decode_time:.4f} seconds")
print(f"Decode Throughput: {throughput:.2f} tokens/second")
print("\nBaseline test complete! The environment is ready.")
