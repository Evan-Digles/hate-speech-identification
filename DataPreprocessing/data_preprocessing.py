"""
Data preprocessing module for hate speech classification
"""

import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords


class DataPreprocessor:
    """
    Preprocessor for hate speech dataset
    """
    
    def __init__(self, max_words=10000, max_len=500):
        """
        Initialize preprocessor
        
        Args:
            max_words: Maximum number of words to keep in vocabulary
            max_len: Maximum length of sequences (padding/truncation)
        """
        self.max_words = max_words
        self.max_len = max_len
        self.tokenizer = None
        self.stop_words = set(stopwords.words('english'))
        
        # Class mappings
        self.class_names = {
            0: 'Hate Speech',
            1: 'Offensive Speech',
            2: 'Normal Speech'
        }
    
    def clean_text(self, text):
        """
        Clean and preprocess text
        
        Args:
            text: Input text string
            
        Returns:
            Cleaned text string
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove HTML entities
        text = re.sub(r'&\w+;', '', text)
        
        # Remove user mentions and hashtags (optional - keep for now)
        # text = re.sub(r'@\w+', '', text)
        # text = re.sub(r'#\w+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def load_data(self, filepath, sample_size=None):
        """
        Load data from CSV file
        
        Args:
            filepath: Path to CSV file
            sample_size: Optional - number of samples to load (for testing)
            
        Returns:
            DataFrame with loaded data
        """
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        
        if sample_size:
            df = df.sample(n=min(sample_size, len(df)), random_state=42)
        
        print(f"Loaded {len(df)} samples")
        print(f"Class distribution:\n{df['class'].value_counts().sort_index()}")
        
        return df
    
    def preprocess_data(self, df, fit_tokenizer=True):
        """
        Preprocess the dataframe
        
        Args:
            df: Input dataframe
            fit_tokenizer: Whether to fit the tokenizer (True for training data)
            
        Returns:
            Tuple of (X, y) where X is padded sequences and y is labels
        """
        # Clean texts
        print("Cleaning texts...")
        df['cleaned_text'] = df['tweet'].apply(self.clean_text)
        
        # Remove empty texts
        df = df[df['cleaned_text'].str.len() > 0]
        
        texts = df['cleaned_text'].values
        labels = df['class'].values
        
        # Tokenize
        if fit_tokenizer:
            print(f"Fitting tokenizer with max_words={self.max_words}...")
            self.tokenizer = Tokenizer(num_words=self.max_words, oov_token='<OOV>')
            self.tokenizer.fit_on_texts(texts)
            print(f"Vocabulary size: {len(self.tokenizer.word_index)}")
        
        # Convert texts to sequences
        sequences = self.tokenizer.texts_to_sequences(texts)
        
        # Pad sequences
        print(f"Padding sequences to max_len={self.max_len}...")
        X = pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')
        
        return X, labels
    
    def prepare_train_test_data(self, filepath, test_size=0.2, val_size=0.1, random_state=42):
        """
        Load and prepare train/validation/test splits
        
        Args:
            filepath: Path to dataset CSV
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with train, validation, and test data
        """
        # Load data
        df = self.load_data(filepath)
        
        # Preprocess
        X, y = self.preprocess_data(df, fit_tokenizer=True)
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Split train into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=random_state, stratify=y_train
        )
        
        print(f"\nData split:")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test
        }
    
    def get_class_name(self, class_id):
        """Get human-readable class name"""
        return self.class_names.get(class_id, f"Unknown ({class_id})")


if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = DataPreprocessor(max_words=5000, max_len=100)
    
    # Load and preprocess mini dataset
    data = preprocessor.prepare_train_test_data('Dataset-Mini.csv')
    
    print("\nPreprocessing complete!")
    print(f"Training data shape: {data['X_train'].shape}")
    print(f"Example class names: {[preprocessor.get_class_name(i) for i in range(3)]}")
