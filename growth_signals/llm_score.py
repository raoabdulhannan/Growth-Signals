from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load model and tokenizer (use a smaller LLaMA model if running locally)
model_name = "deepseek-ai/deepseek-llm-7b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    offload_folder="offload_weights",
    offload_state_dict=True  # Offload parts of the model to CPU
)

# Use pipeline for text generation
text_gen = pipeline("text-generation", model=model, tokenizer=tokenizer)

original_text = "Some random words here."
reconstructed_text = "Some other random words."

prompt = f'''Compare the following two texts and provide a similarity score
from 0 to 1, where 1 means identical and 0 means completely different.

Text 1: {original_text}

Text 2: {reconstructed_text}

Score:'''

response = text_gen(prompt, max_length=500, temperature=0.01,
                    truncation=True)

# Extract score
score_text = response[0]["generated_text"].split("Score:")[-1].strip()
try:
    similarity_score = float(score_text)
except ValueError:
    similarity_score = None

print("Similarity Score:", similarity_score)
