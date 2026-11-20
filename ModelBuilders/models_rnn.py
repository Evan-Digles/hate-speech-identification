# Evan's Comments:
# Bug fixed and ready for use!

# Claude's Comment for Maintenance:
# This module provides an RNN (LSTM-based) classifier for detecting hate speech in text.
# RNNs are particularly good at capturing sequential patterns and long-range dependencies
# in text, which can be important for understanding context in hate speech detection.

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for thread safety with GUI
import matplotlib.pyplot as plt
import seaborn as sns

class RNNModel:
    
    # Claude's Class Comment:
    # This class encapsulates all functionality needed for an RNN text classifier.
    # It uses LSTM (Long Short-Term Memory) layers which are better at handling
    # long sequences compared to simple RNNs, and includes bidirectional processing
    # to capture context from both directions in the text.
    
    def __init__(self, vocab_size=10000, embedding_dim=128, max_len=500, num_classes=3):
        
        self.name = "RNN"
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.num_classes = num_classes
        self.model = None  # Will hold the keras model once built
        self.is_trained = False  # Tracks if it's trained
        self.training_history = None  # Stores training metrics over epochs
        self.classification_report_data = None  # For evaluation metrics
        self.test_accuracy = None  # Test accuracy
    
        self.class_names = {  # Needs to match the labels from the hateXplain dataset
            0: 'Hate Speech',
            1: 'Offensive Speech',
            2: 'Normal Speech'
        }
    
    def build_model(self, lstm_units=128, dropout_rate=0.5, use_bidirectional=True):
        # Makes the RNN model
        
        # Claude's Comment:
        # Constructs the neural network with LSTM layers. The architecture flows:
        # Embedding -> Bidirectional LSTM (optional) -> Dense layers -> Output.
        # Bidirectional processing helps capture context from both directions,
        # which is important for understanding the full meaning of text.
        
        # Args:
            # lstm_units: Number of LSTM units (default 128)
            # dropout_rate: Dropout rate for regularization (default 0.5)
            # use_bidirectional: Whether to use bidirectional LSTM (default True)
        
        self.model = Sequential([
            # Embedding layer converts word indices to dense vectors
            Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, 
                     input_length=self.max_len, name='embedding')
        ])
        
        # Add LSTM layer (bidirectional or regular)
        if use_bidirectional:
            # Claude's Comment:
            # Bidirectional LSTM processes sequence in both forward and backward directions
            # This helps capture context from both past and future words
            self.model.add(Bidirectional(
                LSTM(lstm_units, return_sequences=False, name='lstm'),
                name='bidirectional_lstm'
            ))
        else:
            # Regular LSTM only processes sequence in forward direction
            self.model.add(LSTM(lstm_units, return_sequences=False, name='lstm'))
        
        # Dropout for regularization
        self.model.add(Dropout(dropout_rate, name='dropout1'))
        
        # First fully connected layer
        self.model.add(Dense(64, activation='relu', name='dense1'))
        self.model.add(Dropout(dropout_rate, name='dropout2'))
        
        # Output layer with softmax for multi-class classification
        # Claude's Comment:
        # Output layer with softmax gives probability distribution over 3 classes
        self.model.add(Dense(self.num_classes, activation='softmax', name='output'))
        
        # Compile the model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',  # For integer labels (0, 1, 2)
            metrics=['accuracy']
        )
        
        print(f"{self.name} model built successfully!")
    
    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32, class_weight=None):
        # Trains the RNN model
        
        # Claude's Comment:
        # Training process with validation monitoring and learning rate reduction
        
        # Args:
            # X_train: Training sequences (padded word indices)
            # y_train: Training labels (0, 1, or 2)
            # X_val: Validation sequences
            # y_val: Validation labels
            # epochs: Number of training epochs (default 10)
            # batch_size: Batch size (default 32)
            # class_weight: Dictionary of class weights (optional)
            
        # Returns:
            # Training history object with metrics per epoch
        
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Callbacks for better training
        callbacks = [
            # Reduces learning rate when validation loss plateaus
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, 
                            min_lr=1e-7, verbose=1)
        ]
        
        print(f"\nTraining {self.name} model...")
        
        # Main training loop
        self.training_history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,  # Handles class imbalance
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        print(f"\n{self.name} training completed!")
        return self.training_history
    
    def predict(self, X):
        # Makes predictions on input data
        
        if self.model is None:
            raise ValueError(f"{self.name} model not built or loaded.")
        
        # Get probability distributions and select highest probability class
        proba = self.model.predict(X, verbose=0)
        return np.argmax(proba, axis=1)
    
    def predict_with_threshold(self, X, hate_speech_threshold=0.5): # Worked experimentally with CNN, so I'll be including a RNN version of that here.
        
        # Claude's Comment:
        # This method helps improve precision for Hate Speech by requiring higher confidence
        # before making that prediction. Higher threshold = fewer false positives = better precision.
        
        # Args:
            # X: Input sequences
            # hate_speech_threshold: Min probability to predict Hate Speech (default 0.5)
        
        if self.model is None:
            raise ValueError(f"{self.name} model not built or loaded.")
        
        # Get probability distributions for all samples
        proba = self.model.predict(X, verbose=0)
        predictions = []
        
        for prob_dist in proba:
            # Check if Hate Speech probability meets the threshold
            if prob_dist[0] >= hate_speech_threshold:
                predictions.append(0)  # Hate Speech
            else:
                # For other classes, pick the highest probability between Offensive and Normal
                predictions.append(1 if prob_dist[1] > prob_dist[2] else 2)
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        # Returns probability distributions for all classes
        
        if self.model is None:
            raise ValueError(f"{self.name} model not built or loaded.")
        
        return self.model.predict(X, verbose=0)
    
    def evaluate(self, X_test, y_test): # Very close to evaluate from CNN.
        
        # Predict for all test samples
        y_pred = self.predict(X_test)

        # Calculate overall accuracy
        self.test_accuracy = accuracy_score(y_test, y_pred)
        
        # Generate detailed per-class metrics
        self.classification_report_data = classification_report(
            y_test, y_pred,
            target_names=['Hate Speech', 'Offensive Speech', 'Normal Speech'],
            output_dict=True,
            zero_division=0
        )
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            'accuracy': self.test_accuracy,
            'classification_report': self.classification_report_data,
            'confusion_matrix': cm
        }
    
    def get_classification_report_text(self):
        # Formats the classification report as readable text
        
        if self.classification_report_data is None:
            return f"{self.name} model not evaluated yet."
        
        report = f"\n{self.name} Model Classification Report\n{'='*60}\n\n"
        report += f"Overall Accuracy: {self.test_accuracy:.4f}\n\n"
        
        # Display per-class metrics in readable format
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
        # Saves the model to disk
        
        if self.model is None:
            raise ValueError("No model to save.")
        self.model.save(filepath)
        print(f"{self.name} model saved to {filepath}")
    
    def load_model(self, filepath):
        # Loads a model from disk
        
        self.model = load_model(filepath)
        self.is_trained = True
        print(f"{self.name} model loaded from {filepath}")
    
    def predict_text(self, text, preprocessor):
        # End-to-end prediction for raw text input
        
        # Claude's Comment:
        # End-to-end prediction pipeline for raw text input. Handles cleaning,
        # tokenization, padding, and prediction. Returns structured dictionary
        # with predicted class and probability breakdown.
        
        # Args:
            # text: Input text string (raw, unprocessed)
            # preprocessor: DataPreprocessor instance for text cleaning/tokenization
            
        # Returns:
            # Dictionary with text, predicted_class, class_name, and probabilities
        
        # Clean text (remove URLs, special chars, etc.)
        cleaned = preprocessor.clean_text(text)
        
        # Convert text to sequence of word indices using trained tokenizer
        sequence = preprocessor.tokenizer.texts_to_sequences([cleaned])
        
        # Pad sequence to fixed length (required for RNN input)
        X = tf.keras.preprocessing.sequence.pad_sequences(
            sequence, maxlen=self.max_len, padding='post'
        )
        
        # Get probability distribution over classes
        proba = self.predict_proba(X)[0]
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
    
    def plot_confusion_matrix(self, X_test, y_test, filepath='Plots/RNN_confusion_matrix.png'):
        """Plot and save confusion matrix"""
        if self.model is None:
            raise ValueError("Model not built or loaded.")
        
        # Get predictions
        y_pred = self.predict(X_test)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Hate Speech', 'Offensive', 'Normal'],
                   yticklabels=['Hate Speech', 'Offensive', 'Normal'])
        plt.title(f'{self.name} Model - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to {filepath}")
        return filepath
    
    def plot_training_history(self, filepath='Plots/RNN_training_history.png'):
        """Plot and save training history"""
        if self.training_history is None:
            raise ValueError("No training history available. Train the model first.")
        
        history = self.training_history.history
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(history['accuracy'], label='Training Accuracy', marker='o')
        ax1.plot(history['val_accuracy'], label='Validation Accuracy', marker='s')
        ax1.set_title(f'{self.name} Model - Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(history['loss'], label='Training Loss', marker='o')
        ax2.plot(history['val_loss'], label='Validation Loss', marker='s')
        ax2.set_title(f'{self.name} Model - Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training history saved to {filepath}")
        return filepath
