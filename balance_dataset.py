# balance_dataset.py
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

LABELS = ["anger", "fear", "joy", "sadness", "surprise"]
DATA_PATH = "data/emotions.csv"
OUTPUT_PATH = "data/emotions_balanced_new.csv"

def analyze_label_distribution(df, title="Label Distribution"):
    """Analyze and visualize label distribution"""
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    
    # Count samples per label
    label_counts = {label: df[df[label] == 1].shape[0] for label in LABELS}
    total_samples = df.shape[0]
    
    print(f"Total samples: {total_samples}")
    print(f"Label counts:")
    for label, count in label_counts.items():
        percentage = (count / total_samples) * 100
        print(f"  {label}: {count:,} ({percentage:.1f}%)")
    
    # Calculate imbalance metrics
    counts = list(label_counts.values())
    max_count = max(counts)
    min_count = min(counts)
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    print(f"\nImbalance Analysis:")
    print(f"  Max count: {max_count:,}")
    print(f"  Min count: {min_count:,}")
    print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")
    print(f"  Standard deviation: {np.std(counts):.2f}")
    
    return label_counts, imbalance_ratio

def analyze_multi_label_patterns(df):
    """Analyze multi-label patterns in the dataset"""
    print(f"\n{'='*50}")
    print("Multi-Label Pattern Analysis")
    print(f"{'='*50}")
    
    # Count labels per sample
    labels_per_sample = df[LABELS].sum(axis=1)
    label_count_dist = Counter(labels_per_sample)
    
    print("Labels per sample distribution:")
    for num_labels, count in sorted(label_count_dist.items()):
        percentage = (count / df.shape[0]) * 100
        print(f"  {num_labels} labels: {count:,} samples ({percentage:.1f}%)")
    
    # Find most common label combinations
    print("\nMost common label combinations:")
    label_combinations = []
    for _, row in df.iterrows():
        active_labels = [label for label in LABELS if row[label] == 1]
        if active_labels:
            label_combinations.append(tuple(sorted(active_labels)))
    
    combo_counts = Counter(label_combinations)
    for combo, count in combo_counts.most_common(10):
        percentage = (count / df.shape[0]) * 100
        combo_str = " + ".join(combo) if len(combo) > 1 else combo[0]
        print(f"  {combo_str}: {count:,} ({percentage:.1f}%)")
    
    return label_count_dist, combo_counts

def calculate_data_loss_metrics(original_counts, balanced_counts, final_shape):
    """Calculate data loss and efficiency metrics"""
    print(f"\n{'='*50}")
    print("Data Loss Analysis")
    print(f"{'='*50}")
    
    total_original = sum(original_counts.values())
    total_balanced = sum(balanced_counts.values())
    
    print(f"Original total label instances: {total_original:,}")
    print(f"Balanced total label instances: {total_balanced:,}")
    print(f"Final unique samples: {final_shape[0]:,}")
    
    # Calculate loss per label
    print(f"\nData loss per label:")
    for label in LABELS:
        original = original_counts[label]
        balanced = balanced_counts[label]
        loss = original - balanced
        loss_pct = (loss / original) * 100 if original > 0 else 0
        print(f"  {label}: -{loss:,} samples ({loss_pct:.1f}% loss)")
    
    # Overall metrics
    overall_loss = total_original - total_balanced
    overall_loss_pct = (overall_loss / total_original) * 100
    
    print(f"\nOverall metrics:")
    print(f"  Total data loss: {overall_loss:,} label instances ({overall_loss_pct:.1f}%)")
    print(f"  Data retention: {100 - overall_loss_pct:.1f}%")
    
    return overall_loss_pct

def visualize_balance_comparison(original_counts, balanced_counts):
    """Create visualization comparing original vs balanced distribution"""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original distribution
        labels = list(original_counts.keys())
        original_values = list(original_counts.values())
        
        ax1.bar(labels, original_values, color='lightcoral', alpha=0.7)
        ax1.set_title('Original Label Distribution')
        ax1.set_ylabel('Sample Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(original_values):
            ax1.text(i, v + max(original_values) * 0.01, str(v), ha='center')
        
        # Balanced distribution
        balanced_values = list(balanced_counts.values())
        ax2.bar(labels, balanced_values, color='lightgreen', alpha=0.7)
        ax2.set_title('Balanced Label Distribution')
        ax2.set_ylabel('Sample Count')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(balanced_values):
            ax2.text(i, v + max(balanced_values) * 0.01, str(v), ha='center')
        
        plt.tight_layout()
        plt.savefig('balance_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved as 'balance_comparison.png'")
        
    except ImportError:
        print("\nNote: matplotlib not available for visualization")

def balance_dataset():
    try:
        df = pd.read_csv(DATA_PATH)
        
        # Verify required columns
        if not all(col in df.columns for col in ["text"] + LABELS):
            raise ValueError("Dataset must contain 'text' and label columns: " + ", ".join(LABELS))
        
        print(f"Loaded dataset: {df.shape[0]:,} samples, {df.shape[1]} columns")
        
        # Filter out rows with no emotion labels
        original_shape = df.shape
        df = df[df[LABELS].sum(axis=1) > 0]
        filtered_count = original_shape[0] - df.shape[0]
        
        if filtered_count > 0:
            print(f"Filtered out {filtered_count:,} samples with no emotion labels")
        
        # Analyze original distribution
        original_counts, original_imbalance = analyze_label_distribution(df, "Original Dataset Analysis")
        
        # Analyze multi-label patterns
        label_dist, combo_counts = analyze_multi_label_patterns(df)
        
        # Balance the dataset
        min_count = min(original_counts.values())
        print(f"\nBalancing all labels to {min_count:,} samples each...")
        
        # Collect samples for each label
        balanced_rows = []
        for label in LABELS:
            label_df = df[df[label] == 1]
            sampled = label_df.sample(n=min_count, random_state=42, replace=False)
            balanced_rows.append(sampled)
        
        # Combine and drop duplicates
        balanced_df = pd.concat(balanced_rows).drop_duplicates(subset=["text"])
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Analyze balanced distribution
        balanced_counts, balanced_imbalance = analyze_label_distribution(balanced_df, "Balanced Dataset Analysis")
        
        # Calculate data loss metrics
        data_loss_pct = calculate_data_loss_metrics(original_counts, balanced_counts, balanced_df.shape)
        
        # Create visualization
        visualize_balance_comparison(original_counts, balanced_counts)
        
        # Summary statistics
        print(f"\n{'='*50}")
        print("BALANCING SUMMARY")
        print(f"{'='*50}")
        print(f"Strategy: Minimum-count downsampling")
        print(f"Original imbalance ratio: {original_imbalance:.2f}:1")
        print(f"Balanced imbalance ratio: {balanced_imbalance:.2f}:1")
        print(f"Data retention: {100 - data_loss_pct:.1f}%")
        print(f"Final dataset shape: {balanced_df.shape}")
        
        # Save balanced dataset
        print(f"\nSaving balanced dataset to: {OUTPUT_PATH}")
        balanced_df.to_csv(OUTPUT_PATH, index=False)
        
        return balanced_df
        
    except FileNotFoundError:
        print(f"Error: {DATA_PATH} not found.")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    balance_dataset()