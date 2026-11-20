# Evan's Comments:
# This is actually alright now.
# TODO: Model saving/loading selection. The selection for both should be done in gui manually but I just have it go
# directly to one place since I don't want to handle it right now. 
# Solving this issue is not very high on the list.
#
# Other than that, I'm trying to keep all the models self contained, so no shared abstraction or whatever.


# Claude's Comment for Maintenance:
# This module provides a complete CNN-based classifier for detecting hate speech in text.
# It's designed to work independently without requiring inheritance from base classes.
# The model can classify text into three categories: Hate Speech, Offensive Speech, and Normal Speech.
# See comment inside the class for further information.

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model # Need this for loading.
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.optimizers import Adam # yey adam
from tensorflow.keras.callbacks import ReduceLROnPlateau # Makes learning better.
# from tensorflow.keras.callbacks import EarlyStopping # This and ModelCheckpoint will be kept in comments for reference just in case
# from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score # For tracking. I would cry if I had to do these by scratch.
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for thread safety with GUI
import matplotlib.pyplot as plt
import seaborn as sns

class CNNModel:

    # Claude's Class Comment:
    # This class encapsulates all functionality needed for a CNN text classifier.
    # It handles model building, training, prediction, evaluation, and persistence.
    # The architecture uses 1D convolutions to process sequential text data.
    
    def __init__(self, vocab_size=10000, embedding_dim=128, max_len=500, num_classes=3): # Mostly from geeksforgeeks, fixed for the hateXplain dataset.
        
        self.name = "CNN"
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.num_classes = num_classes
        self.model = None  # Will hold the keras model once we have that.
        self.is_trained = False  # Tracks if it's trained.
        self.training_history = None  # Stores training metrics over epochs, for the graphs and stuff.
        self.classification_report_data = None  # Same as above
        self.test_accuracy = None  # Same as above
    
        self.class_names = { # Needs to match the labels from the hateXplain dataset.
            0: 'Hate Speech',
            1: 'Offensive Speech',
            2: 'Normal Speech'
        }
    
    def build_model(self, filters=128, kernel_size=5, dropout_rate=0.5): # Makes the actual model.
        
        # Claude's Comment:
        # Constructs the neural network layers in sequence. The architecture flows:
        # Embedding -> Conv1D -> MaxPooling -> GlobalMaxPooling -> Dense layers -> Output.
        # This design extracts local features through convolution, then global patterns
        # through pooling, before final classification through fully-connected layers.
        
        # Args:
            # filters: Number of convolutional filters (default 128)
            # kernel_size: Size of convolutional kernel (default 5). Mentioned here since Evan's having trouble understanding this.
            # dropout_rate: Dropout rate for regularization (default 0.5)

        self.model = Sequential([

            # Actually directs how the model understands the positioning of words.
            Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, input_length=self.max_len, name='embedding'),
            
            # Claude's Comment:
            # Conv1D slides a filter across the sequence to detect local patterns
            Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', name='conv1d'),

            # Claude's Comment:
            # MaxPooling reduces dimensionality by keeping only max values in windows
            MaxPooling1D(pool_size=2, name='maxpool'),

            # Claude's Comment:
            # GlobalMaxPooling takes the maximum value across the entire sequence
            GlobalMaxPooling1D(name='global_maxpool'),
            
            # First fully connected layer.
            Dense(128, activation='relu', name='dense1'),
    
            # Disable some of the neurons to help with overfitting.
            Dropout(dropout_rate, name='dropout1'),
        
            # Second fully connected layer, but simpler.
            Dense(64, activation='relu', name='dense2'),
            Dropout(dropout_rate, name='dropout2'),
        
    
            # You need an output layer.
            # Claude's Comment:
            # Output layer with softmax gives probability distribution over 3 classes.
            Dense(self.num_classes, activation='softmax', name='output')
        ])
        
        # Here's the compiler.
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),  # Adam is the learning rate optimizer we're using, if we want to use it in this specific way we need to import it.
            loss='sparse_categorical_crossentropy',  # For integer labels (0, 1, 2)
            metrics=['accuracy']  # Track accuracy during training
        )
        
        print(f"{self.name} model built successfully!") # yay message
    
    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32, class_weight=None): #, patience=5):
                                                                                                  # Patience is unused, but like above kept just in case.
        # Claude's Comment:
        # An explanation, since you're having some trouble with the arguments here
        
        # Args:
            # X_train: Training sequences (padded word indices)
            # y_train: Training labels (0, 1, or 2)
            # X_val: Validation sequences
            # y_val: Validation labels
            # epochs: Number of training epochs (default 10)
            # batch_size: Batch size (default 32)
            # class_weight: Dictionary of class weights (optional)
            # patience: Early stopping patience (default 5)
            
        # Returns:
            # Training history object with metrics per epoch
        
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Claude's Comments:
        # Callbacks modify training behavior dynamically based on performance
        # There used to be early stopping here, but it was removed due to some buggy interactions
        callbacks = [
            # Claude's Comments:
            # Reduces learning rate when validation loss plateaus
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
        ]
        
        print(f"\nTraining {self.name} model...") # Previously a debug, kept since it's nice for the GUI.
        
        # Main training loop iterates through data for specified epochs.
        self.training_history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,  # Handles class imbalance, with whatever weight we give it.
            callbacks=callbacks,        # Does NOT solve the issue of there being little samples of hate speech in the data set.
            verbose=1  # Prints progress bar - Claude
        )
        
        self.is_trained = True
        print(f"\n{self.name} training completed!")
        return self.training_history
    
    def predict(self, X):
        
        if self.model is None:
            raise ValueError(f"{self.name} model not built or loaded.")
        
        # Get percent probabilities for each class, then select the class with the highest probability
        proba = self.model.predict(X, verbose=0)
        return np.argmax(proba, axis=1)
    
    def predict_with_threshold(self, X, hate_speech_threshold=0.5): # Experimental prediction method.
        # Claude's Comment:
        # This method helps improve precision for Hate Speech by requiring higher confidence
        # before making that prediction. Higher threshold = fewer false positives = better precision.
        
        # Args:
            # X: Input sequences
            # hate_speech_threshold: Min probability to predict Hate Speech (default 0.5)
        
        if self.model is None:
            raise ValueError(f"{self.name} model not built or loaded.")
        
        # Claude Comment:
        # Get probability distributions for all samples
        proba = self.model.predict(X, verbose=0)
        predictions = []
        
        for prob_dist in proba:
            # Claude Comment:
            # Check if Hate Speech probability meets the threshold
            if prob_dist[0] >= hate_speech_threshold:
                predictions.append(0)  # Hate Speech
            else:
                # Claude Comment:
                # For other classes, pick the highest probability between Offensive and Normal
                predictions.append(1 if prob_dist[1] > prob_dist[2] else 2)
        
        return np.array(predictions)
    
    def predict_proba(self, X):
       
        if self.model is None:
            raise ValueError(f"{self.name} model not built or loaded.")
        
        return self.model.predict(X, verbose=0)
    
    def evaluate(self, X_test, y_test): # This does the evaluation, and makes the report.
        
        # Claude's Comment:
        # Computes comprehensive evaluation metrics including accuracy, precision,
        # recall, F1-score for each class, and a confusion matrix showing
        # prediction patterns. Stores results in the object for later retrieval.
        
        # Args:
            # X_test: Test sequences (padded word indices)
            # y_test: Test labels (ground truth)
            
        # Returns:
            # Dictionary with accuracy, classification_report, and confusion_matrix
        
        # Predict for all test samples.
        y_pred = self.predict(X_test)

        # Calculate overall accuracy. All we do is correct predictions / total predictions here.
        self.test_accuracy = accuracy_score(y_test, y_pred)
        
        # Claude Comment:
        # Generate detailed per-class metrics, output_dict=True returns dictionary format
        self.classification_report_data = classification_report(
            y_test, y_pred,
            target_names=['Hate Speech', 'Offensive Speech', 'Normal Speech'],
            output_dict=True,
            zero_division=0  # Handles when there isn't a predictions for a class.
        )
        
        # Claude's Comment:
        # Confusion matrix shows actual vs predicted in a grid format
        cm = confusion_matrix(y_test, y_pred)
        
        return { # Hands back the accuracy, classification report, and the confusion matrix now that we're done making all of them.
            'accuracy': self.test_accuracy,
            'classification_report': self.classification_report_data,
            'confusion_matrix': cm
        }
    
    def get_classification_report_text(self): # Formats the classification report. Made by Claude. This works better than
                                              # what I was trying before.
        if self.classification_report_data is None:
            return f"{self.name} model not evaluated yet." # Sometimes we do a dumb. Thanks Claude!
        
        report = f"\n{self.name} Model Classification Report\n{'='*60}\n\n"
        report += f"Overall Accuracy: {self.test_accuracy:.4f}\n\n"
        
        # Claude's Comments:
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
    
    def save_model(self, filepath): # Saves the model in keras format to wherever the filepath points to.
        
        if self.model is None: # Claude to the rescue once again! No more wondering where the model went when I accidentaly
                               # try to save not a model.
            raise ValueError("No model to save.")
        self.model.save(filepath)
        print(f"{self.name} model saved to {filepath}")
    
    def load_model(self, filepath): # Loads a keras model from the filepath spot.

        self.model = load_model(filepath)
        self.is_trained = True
        print(f"{self.name} model loaded from {filepath}")
    
    def predict_text(self, text, preprocessor): # Preprocesses text fed in and then predicts its classifcation with the model.
                                                # Much of this was broken before, and ALL of it has been refactored by Claude at this point.
        # Claude's Comment:
        # End-to-end prediction pipeline for raw text input. Handles cleaning,
        # tokenization, padding, and prediction. Returns structured dictionary
        # with predicted class and probability breakdown.
        
        # Args:
            # text: Input text string (raw, unprocessed)
            # preprocessor: DataPreprocessor instance for text cleaning/tokenization
            
        # Returns:
            # Dictionary with text, predicted_class, class_name, and probabilities

        # Claude's Comment:
        # Clean text (remove URLs, special chars, etc.)
        cleaned = preprocessor.clean_text(text)
        
        # Claude's Comment:
        # Convert text to sequence of word indices using trained tokenizer
        sequence = preprocessor.tokenizer.texts_to_sequences([cleaned])
        
        # Claude's Comment:
        # Pad sequence to fixed length (required for CNN input)
        X = tf.keras.preprocessing.sequence.pad_sequences(
            sequence, maxlen=self.max_len, padding='post'
        )

        # Claude's Comment:
        # Get probability distribution over classes
        proba = self.predict_proba(X)[0]
        predicted_class = np.argmax(proba)
        
        return { # Gives back what we got from the prediction.
            'text': text,
            'predicted_class': predicted_class,
            'class_name': self.class_names[predicted_class],
            'probabilities': {
                'Hate Speech': float(proba[0]),
                'Offensive Speech': float(proba[1]),
                'Normal Speech': float(proba[2])
            }
        }
    
    def plot_confusion_matrix(self, X_test, y_test, filepath='Plots/CNN_confusion_matrix.png'):
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
    
    def plot_training_history(self, filepath='Plots/CNN_training_history.png'):
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
