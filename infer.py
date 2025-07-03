# infer.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from config import HF_TOKEN
from utils import format_instruction
from preprocess import clean_text

# Configuration
MODEL_PATH = "emotion_lora"
BASE_MODEL_ID = "meta-llama/Llama-3.2-1B"
EMOTIONS = ["anger", "fear", "joy", "sadness", "surprise"]

# Clear CUDA cache
torch.cuda.empty_cache()

# Quantization configuration
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,  # Match compute precision to training
    bnb_4bit_use_double_quant=True
)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, token=HF_TOKEN)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=quant_config,
    device_map="auto",
    token=HF_TOKEN
)
model = PeftModel.from_pretrained(base_model, MODEL_PATH)
model.eval()

def predict(text: str, max_new_tokens: int = 20) -> dict:
    cleaned = clean_text(text)
    prompt = format_instruction(cleaned)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.5,  # Lowered for more deterministic output
            top_p=0.85,      # Slightly tightened for focused sampling
            top_k=40,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    # Parse output
    output_text = ""
    if "### Output:" in decoded:
        output_text = decoded.split("### Output:")[-1].strip().lower()
    else:
        output_text = decoded.strip().lower()

    # Robust parsing
    predicted_tokens = [token.strip().strip(".,!?") for token in output_text.split(",")]
    predicted_emotions = [emo for emo in predicted_tokens if emo in EMOTIONS]
    prediction = {emo: int(emo in predicted_emotions) for emo in EMOTIONS}

    # Debug output
    print(f"\nTesting: {text}")
    print("Prompt:\n", prompt)
    print("Generated Text:\n", decoded)
    print("Parsed Labels:\n", predicted_emotions)
    print("Prediction dict:\n", prediction)

    return prediction

# Quick test
if __name__ == "__main__":
    test_sentences = [
        "I can't stop crying... I feel so alone.",
        "Thank you so much for your help!",
        "I just got the best surprise of my life!",
        "I'm really angry now."
    ]
    for sentence in test_sentences:
        predict(sentence)