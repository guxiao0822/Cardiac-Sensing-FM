import re

# ===============================================================
# Text Cleaning Utilities
# ===============================================================

def clean_text_basic(text: str) -> str:
    """Basic cleanup: remove sentences with 'previous tracing', fix punctuation."""
    # Remove sentences containing "previous tracing"
    text = re.sub(r'[^.,]*previous tracing[^.,]*[.,]', '', text, flags=re.IGNORECASE)
    # Replace newlines with commas
    text = re.sub(r'\n+', ', ', text)
    # Standardize commas and spaces
    text = re.sub(r'\s*,\s*', ', ', text).strip(', ')
    # Replace periods with commas (unify punctuation)
    text = text.replace('.', ',')
    # Remove consecutive commas
    text = re.sub(r',\s*,', ',', text).strip(', ')
    # Remove trailing comma if present
    if text.endswith(','):
        text = text[:-1]
    return text.strip()


def refine_text(text: str) -> str:
    """Advanced refinement for removing tracing/comparison references."""
    # Remove phrases containing 'Compared to' or 'tracing' between commas
    text = re.sub(
        r',\s*[^,]*(Compared to|tracing|TRACING|tracing of \[\*\*\d{4}-\d{1,2}-\d{1,2}\*\*\])[^,]*',
        '', text)
    # Remove 'TRACING #X' pattern
    text = re.sub(r',?\s*TRACING\s*#\d+', '', text)
    # Remove redundant commas/spaces
    text = re.sub(r',\s*,', ',', text)
    text = text.strip(', ')
    return text


def preprocess_text(text: str) -> str:
    """Full pipeline for text preprocessing."""
    text = clean_text_basic(text)
    text = refine_text(text)
    return text
