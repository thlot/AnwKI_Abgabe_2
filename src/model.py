from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np

class HateSpeechClassifier:
    def __init__(self, max_features=10000):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),  # Use both unigrams and bigrams
                min_df=2,  # Minimum document frequency
                max_df=0.95  # Maximum document frequency
            )),
            ('classifier', LinearSVC(
                dual=False,  # More efficient for large datasets
                C=1.0,  # Regularization parameter
                class_weight='balanced',  # Handle class imbalance
                max_iter=1000
            ))
        ])
        
    def train(self, X, y, validation_split=0.1):
        """Train the model with validation split"""
        # Split the data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=validation_split, 
            random_state=42,
            stratify=y
        )
        
        # Train the model
        self.pipeline.fit(X_train, y_train)
        
        # Validate
        val_predictions = self.pipeline.predict(X_val)
        return X_val, y_val, val_predictions
    
    def predict(self, X):
        """Make predictions on new data"""
        return self.pipeline.predict(X)