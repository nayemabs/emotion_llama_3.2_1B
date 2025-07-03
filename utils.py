# utils.py
CONTRACTIONS = {
    "don't": "do not", "won't": "will not", "can't": "cannot",
    "i'm": "i am", "he's": "he is", "she's": "she is",
    "it's": "it is", "that's": "that is", "there's": "there is"
}

def format_instruction(text: str) -> str:
    return f"""### Instruction:
Detect all applicable emotions from the following sentence. The only valid emotions are: anger, fear, joy, sadness, surprise. Output all that apply, separated by commas.
### Input:
{text}
### Output:
"""