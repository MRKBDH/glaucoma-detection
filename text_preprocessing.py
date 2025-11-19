"""
Text preprocessing for clinical notes
"""

import re
import nltk
import os

# Set NLTK data path to project directory
nltk_data_dir = '/itch/MRK/dm/nltk_data'
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

if nltk_data_dir not in nltk.data.path:
    nltk.data.path.insert(0, nltk_data_dir)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_dir)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', download_dir=nltk_data_dir)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_dir)


class TextPreprocessor:
    """Handles text preprocessing for clinical notes"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess(self, text):
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short words
        tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(tokens)


def build_vocabulary(texts, min_freq=2):
    """Build vocabulary from texts"""
    word_freq = {}
    for text in texts:
        for word in text.split():
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Filter by minimum frequency
    vocab = ['<PAD>', '<UNK>'] + [word for word, freq in word_freq.items() if freq >= min_freq]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    
    print(f"Vocabulary size: {len(vocab)}")
    
    return word2idx, vocab


if __name__ == "__main__":
    # Test preprocessing
    preprocessor = TextPreprocessor()
    
    sample_text = "Patient presents with elevated intraocular pressure. Examination reveals glaucomatous changes."
    processed = preprocessor.preprocess(sample_text)
    
    print("Original:", sample_text)
    print("Processed:", processed)
    print("\n✓ Text preprocessing works!")
