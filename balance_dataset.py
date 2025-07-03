# balance_dataset.py
import pandas as pd

LABELS = ["anger", "fear", "joy", "sadness", "surprise"]
DATA_PATH = "data/emotions.csv"
OUTPUT_PATH = "data/emotions_balanced.csv"

def balance_dataset():
    try:
        df = pd.read_csv(DATA_PATH)
        # Verify required columns
        if not all(col in df.columns for col in ["text"] + LABELS):
            raise ValueError("Dataset must contain 'text' and label columns: " + ", ".join(LABELS))
        
        # Filter out rows with no emotion labels
        df = df[df[LABELS].sum(axis=1) > 0]

        # Count samples per label
        label_counts = {label: df[df[label] == 1].shape[0] for label in LABELS}
        min_count = min(label_counts.values())

        print("Original label counts:")
        for k, v in label_counts.items():
            print(f"{k}: {v}")
        print(f"\nBalancing all labels to {min_count} samples each...")

        # Collect samples for each label
        balanced_rows = []
        for label in LABELS:
            label_df = df[df[label] == 1]
            sampled = label_df.sample(n=min_count, random_state=42, replace=False)
            balanced_rows.append(sampled)

        # Combine and drop duplicates
        balanced_df = pd.concat(balanced_rows).drop_duplicates(subset=["text"])
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"\nFinal balanced dataset shape: {balanced_df.shape}")
        print("Saving to:", OUTPUT_PATH)
        balanced_df.to_csv(OUTPUT_PATH, index=False)
        
    except FileNotFoundError:
        print(f"Error: {DATA_PATH} not found.")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    balance_dataset()