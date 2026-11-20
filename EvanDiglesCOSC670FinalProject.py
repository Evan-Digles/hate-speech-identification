# Evan Digles COSC 670 Social Networks Hate Speech Identification Project
# Most of my comments for this are in the readme and spread around, this is the main GUI-APP file, so this is what you'll
# actually run. Unfortunately I haven't done a lot of GUI work. So I'll leave an amount of AI generation analysis below.
# NOTE: The analysis only covers RAW CODE BULK, not logic or work flow. Also it's done by hand, so take it with a large grain of salt.
# There'll be a note next to each file too. 
# 
# Only for the main important files. Skewed to higher percentages due to printing code stuff and refactoring changes to variable/method names.

# EvanDiglesCOSC670FinalProject.py      ~80% This file is mostly gui and things not involved with the files. I did get some initial stuff mostly working, but there is a ton of bulk here.
#                                       NOTE: I did have to fix most of the important parts by hand. So I think I'm fine here.
# models_cnn.py             ~36% The AI part is mostly from re-structuring and printing code.
# data_preprocessing.py     ~89% IMPORTANT: I'm including internet code in that percentage. Most of the code for this file is AI or from the internet. 
# models_rnn.py             ~47% I thought this worked out of the box initially, instead it was an AI hallucination that I needed to fix so that LSTM was actually used.
# models_bertweet.py        ~51% A little over half. Much of it is also from their huggingface page. I'm not including the comments for this.
# rebalance_dataset.py      ~>90% Worked on second try without much changed to it. Only included in the submission so you can see how
#                           the original dataset was rebalanced. Just because it was mostly AI generated does NOT mean I didn't look over it.
#                           Though it is pretty simple, mostly just duplication with Hate Speech samples.


"""
GUI Application for Hate Speech Classification
Supports CNN, RNN, and BERTweet model training and text classification
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from ModelBuilders import CNNModel, RNNModel, BERTweetModel
from DataPreprocessing import DataPreprocessor


class HateSpeechClassifierGUI:
    """
    GUI Application for Hate Speech Classification
    Supports CNN, RNN, and BERTweet models
    """
    
    def __init__(self, root):
        """Initialize the GUI"""
        self.root = root
        self.root.title("Evan Digles Hate Speech Classifier - CNN, RNN & BERTweet")
        self.root.geometry("900x700") # Nice default.
        
        # Models and preprocessor
        self.cnn_model = None
        self.rnn_model = None
        self.bertweet_model = None
        self.preprocessor = None
        self.data = None
        self.data_raw = None  # For BERTweet which needs raw text
        self.class_weights = None
        self.last_dataset_path = None  # Track which dataset was last loaded
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Create main container frame with canvas and scrollbar
        main_container = ttk.Frame(self.root)
        main_container.pack(fill='both', expand=True)
        
        # Create canvas for scrolling
        canvas = tk.Canvas(main_container)
        scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        # Configure scrollable frame
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        # Create window in canvas
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Create notebook (tabs) inside scrollable frame
        self.notebook = ttk.Notebook(scrollable_frame)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Tab 1: Training
        self.training_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.training_tab, text='Train Models')
        self.setup_training_tab()
        
        # Tab 2: Classification
        self.classification_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.classification_tab, text='Classify Text')
        self.setup_classification_tab()
        
        # Tab 3: Results
        self.results_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.results_tab, text='Model Reports')
        self.setup_results_tab()
        
    def setup_training_tab(self):
        """Setup the training tab"""
        # Title
        title = tk.Label(self.training_tab, text="Model Training", 
                        font=('Arial', 16, 'bold'))
        title.pack(pady=10)
        
        # Model selection
        model_frame = ttk.LabelFrame(self.training_tab, text="Model Type", padding=10)
        model_frame.pack(fill='x', padx=20, pady=10)
        
        self.model_type_var = tk.StringVar(value="CNN")
        tk.Radiobutton(model_frame, text="CNN (Convolutional Neural Network)", 
                      variable=self.model_type_var, value="CNN", 
                      font=('Arial', 10)).pack(anchor='w', padx=10, pady=5)
        tk.Radiobutton(model_frame, text="RNN (Recurrent Neural Network - LSTM)", 
                      variable=self.model_type_var, value="RNN",
                      font=('Arial', 10)).pack(anchor='w', padx=10, pady=5)
        tk.Radiobutton(model_frame, text="BERTweet (Transformer - Pre-trained on Tweets)", 
                      variable=self.model_type_var, value="BERTweet",
                      font=('Arial', 10)).pack(anchor='w', padx=10, pady=5)
        
        # Dataset selection
        dataset_frame = ttk.LabelFrame(self.training_tab, text="Dataset", padding=10)
        dataset_frame.pack(fill='x', padx=20, pady=10)
        
        tk.Label(dataset_frame, text="Dataset:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.dataset_var = tk.StringVar(value="Datasets/Dataset-Rebalanced.csv")
        dataset_options = [
            "Datasets/Dataset-Rebalanced.csv",
            "Datasets/Dataset-Mini.csv", 
            "Datasets/Dataset-Mini-Rebalanced.csv",
            "Datasets/Dataset---Hate-Speech-Detection-using-Deep-Learning.csv"
        ]
        dataset_menu = ttk.Combobox(dataset_frame, textvariable=self.dataset_var, 
                                    values=dataset_options, width=50, state='readonly')
        dataset_menu.grid(row=0, column=1, padx=5, pady=5)
        
        # Training parameters
        params_frame = ttk.LabelFrame(self.training_tab, text="Training Parameters", padding=10)
        params_frame.pack(fill='x', padx=20, pady=10)
        
        tk.Label(params_frame, text="Epochs:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.epochs_var = tk.IntVar(value=10)
        tk.Spinbox(params_frame, from_=1, to=50, textvariable=self.epochs_var, width=10).grid(
            row=0, column=1, sticky='w', padx=5, pady=5)
        
        tk.Label(params_frame, text="Batch Size:").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.batch_var = tk.IntVar(value=32)
        tk.Spinbox(params_frame, from_=8, to=128, textvariable=self.batch_var, width=10).grid(
            row=1, column=1, sticky='w', padx=5, pady=5)
        
        tk.Label(params_frame, text="Max Vocab Size:").grid(row=2, column=0, sticky='w', padx=5, pady=5)
        self.vocab_var = tk.IntVar(value=10000)
        tk.Spinbox(params_frame, from_=1000, to=20000, increment=1000, 
                  textvariable=self.vocab_var, width=10).grid(
            row=2, column=1, sticky='w', padx=5, pady=5)
        
        self.use_class_weights_var = tk.BooleanVar(value=True)
        tk.Checkbutton(params_frame, text="Use Class Weights (for imbalanced data)", 
                      variable=self.use_class_weights_var).grid(
            row=3, column=0, columnspan=2, sticky='w', padx=5, pady=5)
        
        # Button frame for train and model management
        button_frame = ttk.Frame(self.training_tab)
        button_frame.pack(pady=20)
        
        # Train button
        self.train_btn = tk.Button(button_frame, text="Train Selected Model", 
                                   command=self.train_model, bg='#4CAF50', fg='white',
                                   font=('Arial', 12, 'bold'), padx=20, pady=10)
        self.train_btn.grid(row=0, column=0, padx=10)
        
        # Export model button
        self.export_btn = tk.Button(button_frame, text="Export Model", 
                                    command=self.export_model, bg='#FF9800', fg='white',
                                    font=('Arial', 12, 'bold'), padx=20, pady=10)
        self.export_btn.grid(row=0, column=1, padx=10)
        
        # Import model button
        self.import_btn = tk.Button(button_frame, text="Import Model", 
                                    command=self.import_model, bg='#2196F3', fg='white',
                                    font=('Arial', 12, 'bold'), padx=20, pady=10)
        self.import_btn.grid(row=0, column=2, padx=10)
        
        # Plot export frame
        plot_frame = ttk.LabelFrame(self.training_tab, text="Export Plots (After Training)", padding=10)
        plot_frame.pack(fill='x', padx=20, pady=10)
        
        plot_buttons = ttk.Frame(plot_frame)
        plot_buttons.pack(pady=5)
        
        # Confusion Matrix button
        self.export_cm_btn = tk.Button(plot_buttons, text="Export Confusion Matrix", 
                                       command=self.export_confusion_matrix, bg='#9C27B0', fg='white',
                                       font=('Arial', 11, 'bold'), padx=15, pady=8)
        self.export_cm_btn.grid(row=0, column=0, padx=10)
        
        # Training History button
        self.export_history_btn = tk.Button(plot_buttons, text="Export Training History", 
                                           command=self.export_training_history, bg='#9C27B0', fg='white',
                                           font=('Arial', 11, 'bold'), padx=15, pady=8)
        self.export_history_btn.grid(row=0, column=1, padx=10)
        
        # Progress
        self.progress_var = tk.StringVar(value="Ready to train")
        tk.Label(self.training_tab, textvariable=self.progress_var, 
                font=('Arial', 10)).pack(pady=5)
        
        # Training log
        log_frame = ttk.LabelFrame(self.training_tab, text="Training Log", padding=10)
        log_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        self.training_log = scrolledtext.ScrolledText(log_frame, height=10, wrap=tk.WORD)
        self.training_log.pack(fill='both', expand=True)
        
    def setup_classification_tab(self):
        """Setup the classification tab"""
        # Title
        title = tk.Label(self.classification_tab, text="Text Classification", 
                        font=('Arial', 16, 'bold'))
        title.pack(pady=10)
        
        # Model selection for classification
        classify_model_frame = ttk.LabelFrame(self.classification_tab, text="Select Model", padding=10)
        classify_model_frame.pack(fill='x', padx=20, pady=10)
        
        self.classify_model_var = tk.StringVar(value="CNN")
        tk.Radiobutton(classify_model_frame, text="Use CNN Model", 
                      variable=self.classify_model_var, value="CNN",
                      font=('Arial', 10)).pack(side='left', padx=15)
        tk.Radiobutton(classify_model_frame, text="Use RNN Model", 
                      variable=self.classify_model_var, value="RNN",
                      font=('Arial', 10)).pack(side='left', padx=15)
        tk.Radiobutton(classify_model_frame, text="Use BERTweet Model", 
                      variable=self.classify_model_var, value="BERTweet",
                      font=('Arial', 10)).pack(side='left', padx=15)
        
        # Input frame
        input_frame = ttk.LabelFrame(self.classification_tab, text="Enter Text to Classify", padding=10)
        input_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        self.input_text = scrolledtext.ScrolledText(input_frame, height=8, wrap=tk.WORD,
                                                    font=('Arial', 11))
        self.input_text.pack(fill='both', expand=True)
        self.input_text.insert('1.0', "Enter your text here...")
        
        # Classify button
        self.classify_btn = tk.Button(self.classification_tab, text="Classify Text",
                                      command=self.classify_text, bg='#2196F3', fg='white',
                                      font=('Arial', 12, 'bold'), padx=20, pady=10)
        self.classify_btn.pack(pady=10)
        
        # Results frame
        results_frame = ttk.LabelFrame(self.classification_tab, text="Classification Results", padding=10)
        results_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, height=10, wrap=tk.WORD,
                                                      font=('Arial', 11))
        self.results_text.pack(fill='both', expand=True)
        
    def setup_results_tab(self):
        """Setup the results tab"""
        # Title
        title = tk.Label(self.results_tab, text="Model Performance Reports", 
                        font=('Arial', 16, 'bold'))
        title.pack(pady=10)
        
        # Create notebook for model reports
        self.reports_notebook = ttk.Notebook(self.results_tab)
        self.reports_notebook.pack(fill='both', expand=True, padx=20, pady=10)
        
        # CNN Report tab
        self.cnn_report_frame = ttk.Frame(self.reports_notebook)
        self.reports_notebook.add(self.cnn_report_frame, text='CNN Report')
        
        cnn_scroll_frame = ttk.Frame(self.cnn_report_frame)
        cnn_scroll_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.cnn_report_text = scrolledtext.ScrolledText(cnn_scroll_frame, wrap=tk.WORD,
                                                         font=('Courier', 10))
        self.cnn_report_text.pack(fill='both', expand=True)
        self.cnn_report_text.insert('1.0', "CNN model not trained yet.")
        
        # RNN Report tab
        self.rnn_report_frame = ttk.Frame(self.reports_notebook)
        self.reports_notebook.add(self.rnn_report_frame, text='RNN Report')
        
        rnn_scroll_frame = ttk.Frame(self.rnn_report_frame)
        rnn_scroll_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.rnn_report_text = scrolledtext.ScrolledText(rnn_scroll_frame, wrap=tk.WORD,
                                                         font=('Courier', 10))
        self.rnn_report_text.pack(fill='both', expand=True)
        self.rnn_report_text.insert('1.0', "RNN model not trained yet.")
        
        # BERTweet Report tab
        self.bertweet_report_frame = ttk.Frame(self.reports_notebook)
        self.reports_notebook.add(self.bertweet_report_frame, text='BERTweet Report')
        
        bertweet_scroll_frame = ttk.Frame(self.bertweet_report_frame)
        bertweet_scroll_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.bertweet_report_text = scrolledtext.ScrolledText(bertweet_scroll_frame, wrap=tk.WORD,
                                                              font=('Courier', 10))
        self.bertweet_report_text.pack(fill='both', expand=True)
        self.bertweet_report_text.insert('1.0', "BERTweet model not trained yet.")
        
    def log_message(self, message):
        """Add message to training log"""
        self.training_log.insert(tk.END, message + "\n")
        self.training_log.see(tk.END)
        self.root.update()
        
    def train_model(self):
        """Train the selected model"""
        model_type = self.model_type_var.get()
        
        # Disable train button
        self.train_btn.config(state='disabled')
        self.progress_var.set(f"Training {model_type} model in progress...")
        self.training_log.delete('1.0', tk.END)
        
        # Run training in separate thread
        thread = threading.Thread(target=self._train_model_thread, args=(model_type,))
        thread.start()
        
    def _train_model_thread(self, model_type):
        """Training thread"""
        try:
            # Get parameters
            dataset_path = self.dataset_var.get()
            epochs = self.epochs_var.get()
            batch_size = self.batch_var.get()
            vocab_size = self.vocab_var.get()
            use_weights = self.use_class_weights_var.get()
            
            self.log_message(f"Training {model_type} model")
            self.log_message(f"Loading dataset: {dataset_path}")
            
            # Check if we need to reload data (first time or dataset changed)
            need_reload = (self.preprocessor is None or 
                          self.last_dataset_path != dataset_path)
            
            if need_reload:
                if self.last_dataset_path != dataset_path and self.last_dataset_path is not None:
                    self.log_message(f"Dataset changed from {self.last_dataset_path} to {dataset_path}")
                    self.log_message("Reloading data...")
                
                self.preprocessor = DataPreprocessor(max_words=vocab_size, max_len=500)
                
                # Load and preprocess data
                self.data = self.preprocessor.prepare_train_test_data(dataset_path)
                self.last_dataset_path = dataset_path
                self.log_message(f"Data loaded: {len(self.data['X_train'])} training samples")
                
                # Compute class weights if needed
                if use_weights:
                    unique_classes = np.unique(self.data['y_train'])
                    weights_array = compute_class_weight(
                        class_weight='balanced',
                        classes=unique_classes,
                        y=self.data['y_train']
                    )
                    self.class_weights = {i: w for i, w in enumerate(weights_array)}
                    self.log_message(f"Using class weights: {self.class_weights}")
                else:
                    self.class_weights = None
                    self.log_message("Not using class weights")
            else:
                self.log_message("Using previously loaded data and preprocessor")
            
            actual_vocab_size = min(vocab_size, len(self.preprocessor.tokenizer.word_index) + 1)
            
            # Build and train the selected model
            if model_type == "CNN":
                self.log_message("Building CNN model...")
                self.cnn_model = CNNModel(vocab_size=actual_vocab_size, embedding_dim=128, max_len=500)
                self.cnn_model.build_model()
                
                self.log_message(f"Starting CNN training for {epochs} epochs...")
                self.cnn_model.train(
                    self.data['X_train'], self.data['y_train'],
                    self.data['X_val'], self.data['y_val'],
                    epochs=epochs,
                    batch_size=batch_size,
                    class_weight=self.class_weights
                )
                
                # Evaluate model
                self.log_message("Evaluating CNN model on test data...")
                results = self.cnn_model.evaluate(self.data['X_test'], self.data['y_test'])
                
                self.log_message(f"CNN training completed! Test Accuracy: {results['accuracy']:.4f}")
                
                # Update CNN results tab
                report = self.cnn_model.get_classification_report_text()
                self.cnn_report_text.delete('1.0', tk.END)
                self.cnn_report_text.insert('1.0', report)
                
                messagebox.showinfo("Success", f"CNN model trained successfully!\nTest Accuracy: {results['accuracy']:.4f}")
                
            elif model_type == "RNN":
                self.log_message("Building RNN model...")
                self.rnn_model = RNNModel(vocab_size=actual_vocab_size, embedding_dim=128, max_len=500)
                self.rnn_model.build_model()
                
                self.log_message(f"Starting RNN training for {epochs} epochs...")
                self.rnn_model.train(
                    self.data['X_train'], self.data['y_train'],
                    self.data['X_val'], self.data['y_val'],
                    epochs=epochs,
                    batch_size=batch_size,
                    class_weight=self.class_weights
                )
                
                # Evaluate model
                self.log_message("Evaluating RNN model on test data...")
                results = self.rnn_model.evaluate(self.data['X_test'], self.data['y_test'])
                
                self.log_message(f"RNN training completed! Test Accuracy: {results['accuracy']:.4f}")
                
                # Update RNN results tab
                report = self.rnn_model.get_classification_report_text()
                self.rnn_report_text.delete('1.0', tk.END)
                self.rnn_report_text.insert('1.0', report)
                
                messagebox.showinfo("Success", f"RNN model trained successfully!\nTest Accuracy: {results['accuracy']:.4f}")
                
            else:  # BERTweet
                # BERTweet needs raw text data, not tokenized sequences
                # Check if we need to reload data (first time or dataset changed)
                need_reload_bertweet = (self.data_raw is None or 
                                       self.last_dataset_path != dataset_path)
                
                if need_reload_bertweet:
                    if self.last_dataset_path != dataset_path and self.last_dataset_path is not None:
                        self.log_message(f"Dataset changed from {self.last_dataset_path} to {dataset_path}")
                        self.log_message("Reloading raw text data for BERTweet...")
                    else:
                        self.log_message("Loading raw text data for BERTweet...")
                    
                    df = pd.read_csv(dataset_path)
                    
                    # Perform train/val/test splits
                    from sklearn.model_selection import train_test_split
                    
                    # Split into train and test
                    train_df, test_df = train_test_split(
                        df, test_size=0.2, random_state=42, stratify=df['class']
                    )
                    
                    # Split train into train and val
                    train_df, val_df = train_test_split(
                        train_df, test_size=0.1, random_state=42, stratify=train_df['class']
                    )
                    
                    self.data_raw = {
                        'X_train': train_df['tweet'].values,
                        'y_train': train_df['class'].values,
                        'X_val': val_df['tweet'].values,
                        'y_val': val_df['class'].values,
                        'X_test': test_df['tweet'].values,
                        'y_test': test_df['class'].values
                    }
                    
                    self.last_dataset_path = dataset_path
                    self.log_message(f"Raw data loaded: {len(self.data_raw['X_train'])} training samples")
                    
                    # Compute class weights if needed
                    if use_weights:
                        unique_classes = np.unique(self.data_raw['y_train'])
                        weights_array = compute_class_weight(
                            class_weight='balanced',
                            classes=unique_classes,
                            y=self.data_raw['y_train']
                        )
                        self.class_weights = {i: w for i, w in enumerate(weights_array)}
                        self.log_message(f"Using class weights: {self.class_weights}")
                    else:
                        self.class_weights = None
                else:
                    self.log_message("Using previously loaded raw data for BERTweet")
                
                # Warning about training parameters for BERTweet
                if epochs > 3:
                    self.log_message("WARNING: BERTweet typically performs well with 2-3 epochs.")
                    self.log_message(f"You selected {epochs} epochs, which may lead to overfitting.")
                    self.log_message("Recommended: 2-3 epochs for BERTweet fine-tuning")
                
                if batch_size > 16:
                    self.log_message(f"WARNING: BERTweet uses more memory. Batch size {batch_size} may cause memory issues.")
                    self.log_message("Recommended: 8-16 batch size for BERTweet")
                
                self.log_message("Initializing BERTweet model (this may take a moment)...")
                self.bertweet_model = BERTweetModel(max_len=128)
                
                self.log_message("Building BERTweet model...")
                self.bertweet_model.build_model()
                
                self.log_message(f"Starting BERTweet training for {epochs} epochs (batch size: {batch_size})...")
                self.log_message("Note: BERTweet training is more memory-intensive and may take longer per epoch")
                
                self.bertweet_model.train(
                    self.data_raw['X_train'], self.data_raw['y_train'],
                    self.data_raw['X_val'], self.data_raw['y_val'],
                    epochs=epochs,
                    batch_size=batch_size,
                    class_weight=self.class_weights if use_weights else None
                )
                
                # Evaluate model
                self.log_message("Evaluating BERTweet model on test data...")
                results = self.bertweet_model.evaluate(self.data_raw['X_test'], self.data_raw['y_test'])
                
                self.log_message(f"BERTweet training completed! Test Accuracy: {results['accuracy']:.4f}")
                
                # Update BERTweet results tab
                report = self.bertweet_model.get_classification_report_text()
                self.bertweet_report_text.delete('1.0', tk.END)
                self.bertweet_report_text.insert('1.0', report)
                
                messagebox.showinfo("Success", f"BERTweet model trained successfully!\nTest Accuracy: {results['accuracy']:.4f}")
            
            self.progress_var.set(f"{model_type} training completed successfully!")
            
        except Exception as e:
            self.log_message(f"Error during {model_type} training: {str(e)}")
            self.progress_var.set(f"{model_type} training failed!")
            messagebox.showerror("Error", f"{model_type} training failed:\n{str(e)}")
        
        finally:
            self.train_btn.config(state='normal')
            
    def classify_text(self):
        """Classify the input text using the selected model"""
        model_type = self.classify_model_var.get()
        
        # Check if selected model is trained
        if model_type == "CNN":
            if self.cnn_model is None or not self.cnn_model.is_trained:
                messagebox.showwarning("Warning", "Please train the CNN model first!")
                return
            model = self.cnn_model
            needs_preprocessor = True
        elif model_type == "RNN":
            if self.rnn_model is None or not self.rnn_model.is_trained:
                messagebox.showwarning("Warning", "Please train the RNN model first!")
                return
            model = self.rnn_model
            needs_preprocessor = True
        else:  # BERTweet
            if self.bertweet_model is None or not self.bertweet_model.is_trained:
                messagebox.showwarning("Warning", "Please train the BERTweet model first!")
                return
            model = self.bertweet_model
            needs_preprocessor = False  # BERTweet has its own tokenizer
        
        if needs_preprocessor and self.preprocessor is None:
            messagebox.showwarning("Warning", "Please train a CNN or RNN model first!")
            return
        
        # Get input text
        text = self.input_text.get('1.0', tk.END).strip()
        
        if not text or text == "Enter your text here...":
            messagebox.showwarning("Warning", "Please enter some text to classify!")
            return
        
        try:
            # Classify
            result = model.predict_text(text, self.preprocessor)
            
            # Display results
            self.results_text.delete('1.0', tk.END)
            
            output = "=" * 60 + "\n"
            output += f"CLASSIFICATION RESULTS ({model_type} Model)\n"
            output += "=" * 60 + "\n\n"
            output += f"Input Text:\n{text}\n\n"
            output += f"Predicted Class: {result['class_name']}\n\n"
            output += "Class Probabilities:\n"
            output += "-" * 40 + "\n"
            
            for class_name, prob in result['probabilities'].items():
                bar_length = int(prob * 30)
                bar = '█' * bar_length + '░' * (30 - bar_length)
                output += f"{class_name:20s} [{bar}] {prob:.4f}\n"
            
            self.results_text.insert('1.0', output)
            
        except Exception as e:
            messagebox.showerror("Error", f"Classification failed:\n{str(e)}")
    
    def export_model(self):
        """Export the currently selected model to SavedModels folder"""
        model_type = self.model_type_var.get()
        
        # Check if model is trained
        if model_type == "CNN":
            if self.cnn_model is None or not self.cnn_model.is_trained:
                messagebox.showwarning("Warning", "No trained CNN model to export!")
                return
            model = self.cnn_model
        elif model_type == "RNN":
            if self.rnn_model is None or not self.rnn_model.is_trained:
                messagebox.showwarning("Warning", "No trained RNN model to export!")
                return
            model = self.rnn_model
        else:  # BERTweet
            if self.bertweet_model is None or not self.bertweet_model.is_trained:
                messagebox.showwarning("Warning", "No trained BERTweet model to export!")
                return
            model = self.bertweet_model
        
        try:
            # Save model with simple name (users can rename after export)
            if model_type == "BERTweet":
                filepath = f"SavedModels/{model_type}"
            else:
                filepath = f"SavedModels/{model_type}.keras"
            
            model.save_model(filepath)
            messagebox.showinfo("Success", f"{model_type} model exported to {filepath}\nYou can rename the file if desired.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export {model_type} model:\n{str(e)}")
    
    def import_model(self):
        """Import a model from SavedModels folder"""
        model_type = self.model_type_var.get()
        
        # Open file dialog to select model file
        if model_type == "BERTweet":
            # BERTweet saves as a directory
            filepath = filedialog.askdirectory(
                initialdir="SavedModels",
                title=f"Select {model_type} Model Folder"
            )
        else:
            # CNN and RNN save as .keras files
            filepath = filedialog.askopenfilename(
                initialdir="SavedModels",
                title=f"Select {model_type} Model File",
                filetypes=[("Keras Model", "*.keras"), ("All Files", "*.*")]
            )
        
        if not filepath:
            return  # User cancelled
        
        try:
            # Initialize model object
            if model_type == "CNN":
                self.cnn_model = CNNModel(vocab_size=10000, embedding_dim=128, max_len=500)
                self.cnn_model.load_model(filepath)
                messagebox.showinfo("Success", f"CNN model loaded from {filepath}")
            elif model_type == "RNN":
                self.rnn_model = RNNModel(vocab_size=10000, embedding_dim=128, max_len=500)
                self.rnn_model.load_model(filepath)
                messagebox.showinfo("Success", f"RNN model loaded from {filepath}")
            else:  # BERTweet
                self.bertweet_model = BERTweetModel(max_len=128)
                self.bertweet_model.load_model(filepath)
                messagebox.showinfo("Success", f"BERTweet model loaded from {filepath}")
            
            # Note: Imported models won't have training history or test results
            self.log_message(f"{model_type} model imported successfully")
            self.log_message("Note: Imported models don't include training history or test metrics")
            
            # For CNN/RNN models, we need a preprocessor to classify text
            if model_type in ["CNN", "RNN"] and self.preprocessor is None:
                self.log_message("Note: CNN/RNN models require a preprocessor for text classification")
                
                # Ask user if they want to load a dataset to create the preprocessor
                response = messagebox.askyesno(
                    "Load Dataset?",
                    f"To classify text with the {model_type} model, a preprocessor is needed.\n\n" +
                    "Would you like to load a dataset now to create the preprocessor?\n\n" +
                    "(The preprocessor must use the same vocabulary as the model was trained with)"
                )
                
                if response:
                    # Ask user to select dataset
                    dataset_path = filedialog.askopenfilename(
                        initialdir="Datasets",
                        title="Select Dataset for Preprocessor",
                        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
                    )
                    
                    if dataset_path:
                        try:
                            self.log_message(f"Loading dataset for preprocessor: {dataset_path}")
                            vocab_size = self.vocab_var.get()
                            self.preprocessor = DataPreprocessor(max_words=vocab_size, max_len=500)
                            self.data = self.preprocessor.prepare_train_test_data(dataset_path)
                            self.last_dataset_path = dataset_path
                            
                            self.log_message("Preprocessor created successfully!")
                            self.log_message("You can now classify text with this model.")
                            messagebox.showinfo("Success", "Preprocessor created! You can now classify text.")
                        except Exception as e:
                            self.log_message(f"Error creating preprocessor: {str(e)}")
                            messagebox.showerror("Error", f"Failed to create preprocessor:\n{str(e)}")
                    else:
                        self.log_message("No dataset selected. Classification may not work without preprocessor.")
                        messagebox.showinfo("Info", 
                            f"{model_type} model loaded, but text classification will not work without a preprocessor.\n" +
                            "You can train the model or import it again and load a dataset.")
                else:
                    self.log_message("Preprocessor not created. Classification may not work.")
                    messagebox.showinfo("Info", 
                        f"{model_type} model loaded, but text classification will not work without a preprocessor.\n" +
                        "You can train the model to create the preprocessor.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to import {model_type} model:\n{str(e)}")
    
    def export_confusion_matrix(self):
        """Export confusion matrix plot for the selected model"""
        model_type = self.model_type_var.get()
        
        # Check if model is trained and has test data
        if model_type == "CNN":
            if self.cnn_model is None or not self.cnn_model.is_trained:
                messagebox.showwarning("Warning", "Please train the CNN model first!")
                return
            if self.data is None:
                messagebox.showwarning("Warning", "No test data available. Please train the model first!")
                return
            model = self.cnn_model
            X_test = self.data['X_test']
            y_test = self.data['y_test']
            
        elif model_type == "RNN":
            if self.rnn_model is None or not self.rnn_model.is_trained:
                messagebox.showwarning("Warning", "Please train the RNN model first!")
                return
            if self.data is None:
                messagebox.showwarning("Warning", "No test data available. Please train the model first!")
                return
            model = self.rnn_model
            X_test = self.data['X_test']
            y_test = self.data['y_test']
            
        else:  # BERTweet
            if self.bertweet_model is None or not self.bertweet_model.is_trained:
                messagebox.showwarning("Warning", "Please train the BERTweet model first!")
                return
            if self.data_raw is None:
                messagebox.showwarning("Warning", "No test data available. Please train the model first!")
                return
            
            messagebox.showinfo("BERTweet Note", 
                              "BERTweet confusion matrix plotting requires additional implementation.\n" +
                              "For now, use the Model Reports tab to view metrics.\n" +
                              "You can manually create plots from the report data if needed.")
            return
        
        try:
            # Generate and save confusion matrix plot
            filepath = model.plot_confusion_matrix(X_test, y_test)
            messagebox.showinfo("Success", f"Confusion matrix saved to {filepath}")
            self.log_message(f"Confusion matrix exported to {filepath}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export confusion matrix:\n{str(e)}")
    
    def export_training_history(self):
        """Export training history plot for the selected model"""
        model_type = self.model_type_var.get()
        
        # Check if model is trained and has history
        if model_type == "CNN":
            if self.cnn_model is None or not self.cnn_model.is_trained:
                messagebox.showwarning("Warning", "Please train the CNN model first!")
                return
            if self.cnn_model.training_history is None:
                messagebox.showwarning("Warning", "No training history available!")
                return
            model = self.cnn_model
            
        elif model_type == "RNN":
            if self.rnn_model is None or not self.rnn_model.is_trained:
                messagebox.showwarning("Warning", "Please train the RNN model first!")
                return
            if self.rnn_model.training_history is None:
                messagebox.showwarning("Warning", "No training history available!")
                return
            model = self.rnn_model
            
        else:  # BERTweet
            if self.bertweet_model is None or not self.bertweet_model.is_trained:
                messagebox.showwarning("Warning", "Please train the BERTweet model first!")
                return
            if self.bertweet_model.training_history is None:
                messagebox.showwarning("Warning", "No training history available!")
                return
            
            messagebox.showinfo("BERTweet Note", 
                              "BERTweet training history plotting requires additional implementation.\n" +
                              "For now, monitor training progress in the training log.\n" +
                              "You can manually create plots from the training history if needed.")
            return
        
        try:
            # Generate and save training history plot
            filepath = model.plot_training_history()
            messagebox.showinfo("Success", f"Training history saved to {filepath}")
            self.log_message(f"Training history exported to {filepath}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export training history:\n{str(e)}")


def main():
    """Main function to run the GUI"""
    root = tk.Tk()
    app = HateSpeechClassifierGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
