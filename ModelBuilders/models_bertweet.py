# Evan's Comments:
# I used a slightly different AI model to fix this model, so it may not be the same quality as the RNN model or the CNN model.
# Took about 2 hours to do this one, and significantly more AI usage than I usually approve of, but I got it working.
# Since it isn't straight code from the internet or ai, this should be fine.
# Unfortunately, due to AI Refactoring, the line to actually be able to tell in post is harder to see.

# Claude's Comment for Maintenance:
# This module provides a BERTweet-based classifier for detecting hate speech in text.
# BERTweet uses the transformer architecture and is pre-trained on English tweets,
# making it particularly well-suited for social media text classification tasks.

import numpy as np
import tensorflow as tf
import tf_keras as keras
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

class BERTweetModel:
    
    # Claude's Class Comment:
    # This class encapsulates all functionality needed for a BERTweet text classifier.
    # Unlike CNN and RNN models that are built from scratch, this uses a pre-trained
    # transformer model from Hugging Face, which requires less training data and often
    # performs better on text classification tasks.
    
    def __init__(self, model_name='vinai/bertweet-base', max_len=128, num_classes=3): # The model automatically downloads this, so we don't need to worry about it.
        """
        Initialize BERTweet model
        
        Args:
            model_name: HuggingFace model name (default: vinai/bertweet-base)
            max_len: Maximum sequence length (default: 128, shorter than CNN/RNN due to transformer complexity)
            num_classes: Number of output classes (default: 3)
        """
        self.name = "BERTweet"
        self.model_name = model_name
        self.max_len = max_len
        self.num_classes = num_classes
        self.model = None  
        self.tokenizer = None  # BERTweet has its own tokenizer, which means we need to handle this differently.
        self.is_trained = False
        self.training_history = None
        self.classification_report_data = None
        self.test_accuracy = None
        
        self.class_names = {
            0: 'Hate Speech',
            1: 'Offensive Speech',
            2: 'Normal Speech'
        }
        
        # Initialize tokenizer
        print(f"Loading {self.name} tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False, normalization=True)
        print(f"{self.name} tokenizer loaded successfully!")
    
    def build_model(self, learning_rate=2e-5):
        """
        Build the BERTweet model for classification
        
        Args:
            learning_rate: Learning rate for fine-tuning (default: 2e-5, typical for transformers)
        """
        # Claude's Comment:
        # Load pre-trained BERTweet model and add a classification head.
        # The model is initialized with pre-trained weights from millions of tweets,
        # so it already understands language patterns well.
        
        print(f"Building {self.name} model...")
        
        # Load pre-trained model with classification head
        self.model = TFAutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_classes,
            from_pt=True  # Convert from PyTorch to TensorFlow. Should probably use a better variable name, but oh well!
        )
        
        # Compile model with appropriate loss and... Metrics.
        # Keras 2 was a PAIN to figure out.
        # Claude's Comment:
        # For transformers, we use SparseCategoricalCrossentropy since the model outputs logits
        # Using tf_keras (Keras 2) optimizer for compatibility with TensorFlow 2.x and transformers
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=['accuracy']
        )
        
        print(f"{self.name} model built successfully!")
    
    def prepare_data(self, texts, labels=None):
        """
        Tokenize texts using BERTweet tokenizer
        
        Args:
            texts: List of text strings
            labels: Optional labels array
            
        Returns:
            Dictionary with input_ids, attention_mask, and optionally labels
        """
        # Claude's Comment:
        # BERTweet tokenizer handles tweet-specific tokens like hashtags, mentions, emojis
        # It also performs normalization to handle variations in social media text
        
        encodings = self.tokenizer(
            texts.tolist() if isinstance(texts, np.ndarray) else texts,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='tf'
        )
        
        if labels is not None:
            return {
                'input_ids': encodings['input_ids'],
                'attention_mask': encodings['attention_mask'],
                'labels': tf.convert_to_tensor(labels)
            }
        else:
            return {
                'input_ids': encodings['input_ids'],
                'attention_mask': encodings['attention_mask']
            }
    
    def train(self, X_train, y_train, X_val, y_val, epochs=3, batch_size=16, class_weight=None):
        """
        Train the BERTweet model
        
        Args:
            X_train: Training texts (raw text strings or pre-tokenized)
            y_train: Training labels
            X_val: Validation texts
            y_val: Validation labels
            epochs: Number of training epochs (default: 3, fewer needed for pre-trained models)
            batch_size: Batch size (default: 16, smaller due to transformer memory requirements)
            class_weight: Dictionary of class weights
            
        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        print(f"\nTraining {self.name} model...")
        print(f"Note: Training transformers requires more memory and time per epoch")
        
        # Prepare datasets.
        train_dataset = self.prepare_data(X_train, y_train)
        val_dataset = self.prepare_data(X_val, y_val)
        
        # Create validation dataset.
        val_tf_dataset = tf.data.Dataset.from_tensor_slices((
            {
                'input_ids': val_dataset['input_ids'],
                'attention_mask': val_dataset['attention_mask']
            },
            val_dataset['labels']
        )).batch(batch_size)
        
        # Create training dataset with or without class weights.
        if class_weight is not None:
            # Convert class weights to sample weights, so they can be used.
            sample_weights = np.array([class_weight[label] for label in y_train])
            train_tf_dataset = tf.data.Dataset.from_tensor_slices((
                {
                    'input_ids': train_dataset['input_ids'],
                    'attention_mask': train_dataset['attention_mask']
                },
                train_dataset['labels'],
                sample_weights
            )).shuffle(1000).batch(batch_size)
        else:
            # No class weights
            train_tf_dataset = tf.data.Dataset.from_tensor_slices((
                {
                    'input_ids': train_dataset['input_ids'],
                    'attention_mask': train_dataset['attention_mask']
                },
                train_dataset['labels']
            )).shuffle(1000).batch(batch_size)
        
        # Callbacks so we can do loss and learning speed and early stopping if we have that too (Which we don't.)
        callbacks = [
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=1,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train model, which takes a long time.
        self.training_history = self.model.fit(
            train_tf_dataset,
            validation_data=val_tf_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        print(f"\n{self.name} training completed!")
        return self.training_history
    
    def predict(self, X):
        """
        Make predictions on input texts
        
        Args:
            X: Input texts (raw strings)
            
        Returns:
            Array of predicted class labels
        """
        if self.model is None:
            raise ValueError(f"{self.name} model not built or loaded.")
        
        # Prepare data
        data = self.prepare_data(X)
        
        # Get predictions
        logits = self.model.predict({
            'input_ids': data['input_ids'],
            'attention_mask': data['attention_mask']
        }, verbose=0).logits
        
        return np.argmax(logits, axis=1)
    
    def predict_proba(self, X):
        """
        Get probability distributions for input texts
        
        Args:
            X: Input texts (raw strings)
            
        Returns:
            Array of probability distributions
        """
        if self.model is None:
            raise ValueError(f"{self.name} model not built or loaded.")
        
        # Prepare data
        data = self.prepare_data(X)
        
        # Get logits
        logits = self.model.predict({
            'input_ids': data['input_ids'],
            'attention_mask': data['attention_mask']
        }, verbose=0).logits
        
        # Convert logits to probabilities using softmax
        probabilities = tf.nn.softmax(logits, axis=-1).numpy()
        
        return probabilities
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data
        
        Args:
            X_test: Test texts
            y_test: Test labels
            
        Returns:
            Dictionary with accuracy, classification_report, and confusion_matrix
        """
        print(f"Evaluating {self.name} model...")
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        self.test_accuracy = accuracy_score(y_test, y_pred)
        
        self.classification_report_data = classification_report(
            y_test, y_pred,
            target_names=['Hate Speech', 'Offensive Speech', 'Normal Speech'],
            output_dict=True,
            zero_division=0
        )
        
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            'accuracy': self.test_accuracy,
            'classification_report': self.classification_report_data,
            'confusion_matrix': cm
        }
    
    def get_classification_report_text(self):
        """Format classification report as readable text"""
        if self.classification_report_data is None:
            return f"{self.name} model not evaluated yet."
        
        report = f"\n{self.name} Model Classification Report\n{'='*60}\n\n"
        report += f"Overall Accuracy: {self.test_accuracy:.4f}\n\n"
        
        for class_name in ['Hate Speech', 'Offensive Speech', 'Normal Speech']:
            if class_name in self.classification_report_data:
                m = self.classification_report_data[class_name]
                report += f"{class_name}:\n"
                report += f"  Precision: {m['precision']:.4f}\n"
                report += f"  Recall:    {m['recall']:.4f}\n"
                report += f"  F1-Score:  {m['f1-score']:.4f}\n"
                report += f"  Support:   {m['support']}\n\n"
        
        return report
    
    def save_model(self, filepath):
        """
        Save the model to disk
        
        Args:
            filepath: Directory path to save model
        """
        if self.model is None:
            raise ValueError("No model to save.")
        
        # Create directory if it doesn't exist TODO: Review this.
        os.makedirs(filepath, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(filepath)
        self.tokenizer.save_pretrained(filepath)
        
        print(f"{self.name} model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a model from disk
        
        Args:
            filepath: Directory path to load model from
        """
        self.model = TFAutoModelForSequenceClassification.from_pretrained(filepath)
        self.tokenizer = AutoTokenizer.from_pretrained(filepath)
        self.is_trained = True
        
        print(f"{self.name} model loaded from {filepath}")
    
    def predict_text(self, text, preprocessor=None):
        """
        Predict classification for raw text input
        
        Args:
            text: Input text string (raw, unprocessed)
            preprocessor: Not used for BERTweet (kept for compatibility with other models)
            
        Returns:
            Dictionary with text, predicted_class, class_name, and probabilities
        """
        if self.model is None:
            raise ValueError(f"{self.name} model not built or loaded.")
        
        # BERTweet tokenizer handles text preprocessing internally
        # No need for external preprocessor
        
        # Get probability distribution
        proba = self.predict_proba([text])[0]
        predicted_class = np.argmax(proba)
        
        return {
            'text': text,
            'predicted_class': predicted_class,
            'class_name': self.class_names[predicted_class],
            'probabilities': {
                'Hate Speech': float(proba[0]),
                'Offensive Speech': float(proba[1]),
                'Normal Speech': float(proba[2])
            }
        }
