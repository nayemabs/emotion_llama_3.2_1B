# train.py
import os
import random
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from preprocess import clean_text
from utils import format_instruction
from config import HF_TOKEN

# Set environment variable for memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_ID = "meta-llama/Llama-3.2-1B"
DATA_PATH = "data/emotions_balanced.csv"
OUTPUT_DIR = "emotion_lora"
LABELS = ["anger", "fear", "joy", "sadness", "surprise"]

# Clear CUDA cache
torch.cuda.empty_cache()

# Load and preprocess dataset
df = pd.read_csv(DATA_PATH)
df["cleaned_text"] = df["text"].astype(str).apply(clean_text)

def format_row(row):
    labels = [label for label in LABELS if row[label] == 1]
    random.shuffle(labels)
    return {
        "prompt": format_instruction(row["cleaned_text"]),
        "output": ", ".join(labels) if labels else "none"
    }

formatted_df = pd.DataFrame([format_row(row) for _, row in df.iterrows()])
dataset = Dataset.from_pandas(formatted_df)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(examples):
    full_texts = [prompt + output for prompt, output in zip(examples["prompt"], examples["output"])]
    tokenized = tokenizer(
        full_texts,
        truncation=True,
        padding="max_length",
        max_length=512
    )
    tokenized["labels"] = [ids[:] for ids in tokenized["input_ids"]]
    return tokenized

tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["prompt", "output"])

# Quantization configuration
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,  # Use float16 for faster computation
    bnb_4bit_use_double_quant=True  # Enable nested quantization for additional memory savings
)

# Model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quant_config,
    device_map="auto",
    token=HF_TOKEN
)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Training args
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    num_train_epochs=30,
    learning_rate=5e-5,
    fp16=True,
    logging_steps=50,
    save_strategy="epoch",
    save_total_limit=2,
    warmup_ratio=0.1,
    optim="adamw_torch",
    gradient_checkpointing=True
)

# Trainer
# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    processing_class=tokenizer
)

# Resume from latest checkpoint if exists
last_checkpoint = None
if os.path.isdir(OUTPUT_DIR):
    checkpoints = [os.path.join(OUTPUT_DIR, d) for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint")]
    if checkpoints:
        last_checkpoint = max(checkpoints, key=os.path.getctime)
        print(f"🧠 Resuming from checkpoint: {last_checkpoint}")
    else:
        print("🔁 No checkpoint found. Training from scratch.")
else:
    print("📁 Output directory does not exist. Training from scratch.")

print("✅ Starting training...")
trainer.train(resume_from_checkpoint=last_checkpoint)


print("✅ Starting training...")
trainer.train()

print("✅ Saving model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)