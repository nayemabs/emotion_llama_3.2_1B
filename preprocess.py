# preprocess.py
import re
from utils import CONTRACTIONS

def clean_text(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces
    text = re.sub(r'[^\w\s.,!?\'\"-]', '', text)  # Keep basic punctuation
    # Expand contractions
    words = text.split()
    words = [CONTRACTIONS.get(word.lower(), word) for word in words]
    return ' '.join(words)