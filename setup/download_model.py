from transformers import AutoModelForCausalLM, AutoTokenizer

# Replace this with Mistral if you prefer!
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct" 

print(f"Starting download for {model_id}...")
print("This will take a few minutes. Grabbing a coffee is recommended!")

# This triggers the download and saves it to your instance's hidden ~/.cache folder
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

print("Download complete! The model is cached and ready for your teammates.")
