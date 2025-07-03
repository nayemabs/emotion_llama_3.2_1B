# main.py
import torch
import pandas as pd
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

# Quantization config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
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


def predict_single(text: str, max_new_tokens: int = 20) -> list:
    cleaned = clean_text(text)
    prompt = format_instruction(cleaned)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.5,
            top_p=0.85,
            top_k=40,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    output_text = decoded.split("### Output:")[-1].strip().lower() if "### Output:" in decoded else decoded.strip().lower()

    predicted_tokens = [token.strip(" .,!?") for token in output_text.split(",")]
    predicted_emotions = [emo for emo in predicted_tokens if emo in EMOTIONS]
    return [int(emo in predicted_emotions) for emo in EMOTIONS]


def predict(csv_file_path: str) -> list[list[int]]:
    df = pd.read_csv(csv_file_path)

    # Assumes column with text is named 'text' or similar
    text_column = None
    for col in df.columns:
        if col.lower() in ["text", "sentence", "comment", "input"]:
            text_column = col
            break

    if text_column is None:
        raise ValueError("No appropriate text column found in the CSV. Expected one of: 'text', 'sentence', 'comment', 'input'.")

    predictions = [predict_single(text) for text in df[text_column]]
    return predictions


# Optional test
if __name__ == "__main__":
    preds = predict("data/emotions.csv")  # Replace with your actual test file
    for row in preds:
        print(row)
