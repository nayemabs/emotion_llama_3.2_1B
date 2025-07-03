# Emotion Detection with Fine-tuned Llama 3.2

A multi-label emotion detection system that fine-tunes Meta's Llama 3.2-1B model using LoRA (Low-Rank Adaptation) to classify text into five emotion categories: anger, fear, joy, sadness, and surprise.

## Features

- **Multi-label Classification**: Detects multiple emotions in a single text
- **Memory Efficient**: Uses 4-bit quantization and LoRA for efficient training
- **Balanced Dataset**: Automatic dataset balancing for fair emotion representation
- **Text Preprocessing**: Robust text cleaning and normalization
- **Hugging Face Integration**: Easy model sharing and deployment

## Project Structure

```
emotion_detection/
├── balance_dataset.py    # Dataset balancing utilities
├── config.py            # Configuration (HF token)
├── infer.py             # Single inference script
├── main.py              # Batch prediction script
├── preprocess.py        # Text preprocessing utilities
├── train.py             # Model training script
├── utils.py             # Helper functions and contractions
├── requirements.txt     # Python dependencies
└── data/
    ├── emotions.csv     # Original dataset
    └── emotions_balanced.csv  # Balanced dataset
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd emotion_detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your Hugging Face token in `config.py`:
```python
HF_TOKEN = "your_hugging_face_token_here"
```

## Usage

### 1. Prepare Your Dataset

Your CSV file should contain:
- A `text` column with the text to classify
- Binary columns for each emotion: `anger`, `fear`, `joy`, `sadness`, `surprise`

Example:
```csv
text,anger,fear,joy,sadness,surprise
"I'm so happy today!",0,0,1,0,0
"This is terrifying and makes me angry",1,1,0,0,0
```

### 2. Balance the Dataset

```bash
python balance_dataset.py
```

This will:
- Load data from `data/emotions.csv`
- Balance all emotion classes to have equal representation
- Save the balanced dataset to `data/emotions_balanced.csv`

### 3. Train the Model

```bash
python train.py
```

Training features:
- **Model**: Meta Llama 3.2-1B with LoRA fine-tuning
- **Quantization**: 4-bit quantization for memory efficiency
- **Epochs**: 30 epochs with warmup
- **Batch Size**: 2 per device with gradient accumulation
- **Checkpoint Support**: Automatically resumes from last checkpoint

### 4. Run Inference

#### Single Text Prediction
```bash
python infer.py
```

#### Batch Prediction from CSV
```bash
python main.py
```

## Model Architecture

- **Base Model**: Meta Llama 3.2-1B
- **Fine-tuning**: LoRA (Low-Rank Adaptation)
  - Rank: 16
  - Alpha: 32
  - Target modules: `q_proj`, `v_proj`
  - Dropout: 0.1
- **Quantization**: 4-bit NF4 with double quantization
- **Optimization**: AdamW with gradient checkpointing

## Emotion Categories

The model classifies text into these five emotions:
- **Anger**: Frustration, irritation, rage
- **Fear**: Anxiety, worry, terror
- **Joy**: Happiness, excitement, delight
- **Sadness**: Sorrow, melancholy, grief
- **Surprise**: Amazement, shock, wonder

## Performance Optimizations

- **Memory Efficiency**: 4-bit quantization reduces memory usage by ~75%
- **LoRA Training**: Only trains 0.1% of model parameters
- **Gradient Checkpointing**: Trades compute for memory
- **Mixed Precision**: FP16 training for faster computation

## Configuration

Key parameters in `train.py`:
- `per_device_train_batch_size`: Adjust based on GPU memory
- `gradient_accumulation_steps`: Increase for effective larger batch sizes
- `num_train_epochs`: Training duration
- `learning_rate`: LoRA learning rate (5e-5)

## Hardware Requirements

### Minimum
- **GPU**: 8GB VRAM (GTX 1070, RTX 3060)
- **RAM**: 16GB system RAM
- **Storage**: 5GB for model and data

### Recommended
- **GPU**: 16GB+ VRAM (RTX 4070, V100)
- **RAM**: 32GB system RAM
- **Storage**: 10GB+ for checkpoints and logs

## Troubleshooting

### CUDA Out of Memory
- Reduce `per_device_train_batch_size` to 1
- Increase `gradient_accumulation_steps`
- Enable `gradient_checkpointing=True`

### Slow Training
- Increase batch size if memory allows
- Use multiple GPUs with `device_map="auto"`
- Enable mixed precision with `fp16=True`

### Poor Performance
- Increase training epochs
- Adjust learning rate (try 1e-4 or 2e-5)
- Ensure dataset is properly balanced

## Output Format

The model outputs emotions as comma-separated values:
```
Input: "I'm so excited but also nervous!"
Output: "joy, fear"
```

## Dependencies

Core libraries:
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face model library
- **PEFT**: Parameter-efficient fine-tuning
- **BitsAndBytes**: Quantization library
- **Datasets**: Data loading and processing
- **Pandas**: Data manipulation

## License

This project uses the Llama 3.2 model which requires compliance with Meta's licensing terms. Please review the [Llama 3.2 license](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE) before use.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Acknowledgments

- Meta AI for the Llama 3.2 model
- Hugging Face for the transformers library
- Microsoft for the LoRA technique