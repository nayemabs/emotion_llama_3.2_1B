# main.py
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import (
    confusion_matrix, 
    precision_recall_fscore_support,
    classification_report,
    accuracy_score,
    hamming_loss,
    jaccard_score,
    multilabel_confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
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
    
    # Find text column
    text_column = None
    for col in df.columns:
        if col.lower() in ["text", "sentence", "comment", "input"]:
            text_column = col
            break
    
    if text_column is None:
        raise ValueError("No appropriate text column found in the CSV. Expected one of: 'text', 'sentence', 'comment', 'input'.")
    
    predictions = [predict_single(text) for text in df[text_column]]
    return predictions

def load_ground_truth(csv_file_path: str) -> np.ndarray:
    """
    Load ground truth labels from CSV file.
    Assumes columns named with emotion names or in the same order as EMOTIONS.
    """
    df = pd.read_csv(csv_file_path)
    
    # Try to find emotion columns by name
    emotion_columns = []
    for emotion in EMOTIONS:
        if emotion in df.columns:
            emotion_columns.append(emotion)
        elif emotion.capitalize() in df.columns:
            emotion_columns.append(emotion.capitalize())
    
    if len(emotion_columns) == len(EMOTIONS):
        return df[emotion_columns].values
    
    # If not found by name, assume they're in order after text column
    text_column = None
    for col in df.columns:
        if col.lower() in ["text", "sentence", "comment", "input"]:
            text_column = col
            break
    
    if text_column is None:
        raise ValueError("Cannot determine ground truth format. Expected emotion columns or text column.")
    
    # Assume emotion columns come after text column
    text_col_idx = df.columns.get_loc(text_column)
    emotion_cols = df.columns[text_col_idx + 1:text_col_idx + 1 + len(EMOTIONS)]
    
    if len(emotion_cols) != len(EMOTIONS):
        raise ValueError(f"Expected {len(EMOTIONS)} emotion columns, found {len(emotion_cols)}")
    
    return df[emotion_cols].values

def plot_confusion_matrices(y_true, y_pred, emotions, save_path=None):
    """Plot confusion matrices for each emotion label."""
    n_labels = len(emotions)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Get multilabel confusion matrices
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    
    for i, (emotion, cm) in enumerate(zip(emotions, mcm)):
        if i < len(axes):
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Not ' + emotion, emotion],
                       yticklabels=['Not ' + emotion, emotion],
                       ax=axes[i])
            axes[i].set_title(f'{emotion.capitalize()} Confusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
    
    # Hide the extra subplot
    if len(axes) > len(emotions):
        axes[-1].set_visible(False)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def calculate_metrics(y_true, y_pred, emotions):
    """Calculate comprehensive evaluation metrics."""
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Overall metrics
    exact_match_ratio = accuracy_score(y_true, y_pred)
    hamming_loss_score = hamming_loss(y_true, y_pred)
    jaccard_score_macro = jaccard_score(y_true, y_pred, average='macro')
    jaccard_score_micro = jaccard_score(y_true, y_pred, average='micro')
    
    # Per-label metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Macro and micro averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0
    )
    
    # Weighted averages
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # Create results dictionary
    results = {
        'overall_metrics': {
            'exact_match_ratio': exact_match_ratio,
            'hamming_loss': hamming_loss_score,
            'jaccard_score_macro': jaccard_score_macro,
            'jaccard_score_micro': jaccard_score_micro,
        },
        'macro_averages': {
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
        },
        'micro_averages': {
            'precision_micro': precision_micro,
            'recall_micro': recall_micro,
            'f1_micro': f1_micro,
        },
        'weighted_averages': {
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
        },
        'per_label_metrics': {}
    }
    
    # Per-label results
    for i, emotion in enumerate(emotions):
        results['per_label_metrics'][emotion] = {
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i],
            'support': support[i]
        }
    
    return results

def print_detailed_results(results, emotions):
    """Print detailed evaluation results."""
    print("=" * 80)
    print("EMOTION CLASSIFICATION EVALUATION RESULTS")
    print("=" * 80)
    
    # Overall metrics
    print("\nOVERALL METRICS:")
    print("-" * 40)
    print(f"Exact Match Ratio: {results['overall_metrics']['exact_match_ratio']:.4f}")
    print(f"Hamming Loss: {results['overall_metrics']['hamming_loss']:.4f}")
    print(f"Jaccard Score (Macro): {results['overall_metrics']['jaccard_score_macro']:.4f}")
    print(f"Jaccard Score (Micro): {results['overall_metrics']['jaccard_score_micro']:.4f}")
    
    # Macro averages
    print("\nMACRO AVERAGES:")
    print("-" * 40)
    print(f"Precision (Macro): {results['macro_averages']['precision_macro']:.4f}")
    print(f"Recall (Macro): {results['macro_averages']['recall_macro']:.4f}")
    print(f"F1-Score (Macro): {results['macro_averages']['f1_macro']:.4f}")
    
    # Micro averages
    print("\nMICRO AVERAGES:")
    print("-" * 40)
    print(f"Precision (Micro): {results['micro_averages']['precision_micro']:.4f}")
    print(f"Recall (Micro): {results['micro_averages']['recall_micro']:.4f}")
    print(f"F1-Score (Micro): {results['micro_averages']['f1_micro']:.4f}")
    
    # Weighted averages
    print("\nWEIGHTED AVERAGES:")
    print("-" * 40)
    print(f"Precision (Weighted): {results['weighted_averages']['precision_weighted']:.4f}")
    print(f"Recall (Weighted): {results['weighted_averages']['recall_weighted']:.4f}")
    print(f"F1-Score (Weighted): {results['weighted_averages']['f1_weighted']:.4f}")
    
    # Per-label metrics
    print("\nPER-LABEL METRICS:")
    print("-" * 40)
    print(f"{'Emotion':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 52)
    for emotion in emotions:
        metrics = results['per_label_metrics'][emotion]
        print(f"{emotion:<12} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
              f"{metrics['f1']:<10.4f} {metrics['support']:<10}")

def save_results_to_csv(results, emotions, save_path="evaluation_results.csv"):
    """Save evaluation results to CSV file."""
    # Prepare data for CSV
    data = []
    
    # Overall metrics
    for metric, value in results['overall_metrics'].items():
        data.append(['Overall', metric, value, None, None])
    
    # Macro averages
    for metric, value in results['macro_averages'].items():
        data.append(['Macro', metric, value, None, None])
    
    # Micro averages
    for metric, value in results['micro_averages'].items():
        data.append(['Micro', metric, value, None, None])
    
    # Weighted averages
    for metric, value in results['weighted_averages'].items():
        data.append(['Weighted', metric, value, None, None])
    
    # Per-label metrics
    for emotion in emotions:
        metrics = results['per_label_metrics'][emotion]
        for metric, value in metrics.items():
            data.append(['Per-Label', metric, value, emotion, None])
    
    # Create DataFrame and save
    df = pd.DataFrame(data, columns=['Category', 'Metric', 'Value', 'Emotion', 'Extra'])
    df.to_csv(save_path, index=False)
    print(f"\nResults saved to {save_path}")

def evaluate_model(csv_file_path: str, save_plots=True, save_csv=True):
    """
    Complete evaluation pipeline.
    
    Args:
        csv_file_path: Path to CSV file with text and ground truth labels
        save_plots: Whether to save confusion matrix plots
        save_csv: Whether to save results to CSV
    """
    print("Loading data and generating predictions...")
    
    # Get predictions
    predictions = predict(csv_file_path)
    
    # Load ground truth
    y_true = load_ground_truth(csv_file_path)
    y_pred = np.array(predictions)
    
    print(f"Evaluated {len(y_true)} samples")
    print(f"Ground truth shape: {y_true.shape}")
    print(f"Predictions shape: {y_pred.shape}")
    
    # Calculate metrics
    print("\nCalculating metrics...")
    results = calculate_metrics(y_true, y_pred, EMOTIONS)
    
    # Print results
    print_detailed_results(results, EMOTIONS)
    
    # Plot confusion matrices
    if save_plots:
        print("\nGenerating confusion matrix plots...")
        plot_confusion_matrices(y_true, y_pred, EMOTIONS, "confusion_matrices.png")
    
    # Save results to CSV
    if save_csv:
        save_results_to_csv(results, EMOTIONS)
    
    return results

# Main execution
if __name__ == "__main__":
    # Example usage
    try:
        # Replace with your actual test file path
        csv_file_path = "data/track-a-test.csv"
        
        # Run complete evaluation
        results = evaluate_model(csv_file_path)
        
        # You can also run individual components:
        # predictions = predict(csv_file_path)
        # y_true = load_ground_truth(csv_file_path)
        # metrics = calculate_metrics(y_true, predictions, EMOTIONS)
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        print("Please ensure your CSV file has the correct format with text and emotion label columns.")