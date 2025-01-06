from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np

class HateSpeechClassifier:
    def __init__(self, max_features=12000):  # Slightly increased from original
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),  # Keep bigrams which worked well
                min_df=2,
                max_df=0.95,
                sublinear_tf=True,    # Add sublinear scaling
                strip_accents='unicode'
            )),
            ('classifier', LinearSVC(
                dual=False,
                C=1.2,               # Slightly increased regularization parameter
                class_weight='balanced',
                max_iter=1500,       # Increased from original
                tol=1e-4
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
        print("\nTraining model...")
        self.pipeline.fit(X_train, y_train)
        
        # Validate
        val_predictions = self.pipeline.predict(X_val)
        return X_val, y_val, val_predictions
    
    def predict(self, X):
        """Make predictions on new data"""
        return self.pipeline.predict(X)