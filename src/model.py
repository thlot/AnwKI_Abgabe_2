from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
import numpy as np

class HateSpeechClassifier:
    def __init__(self, max_features=13000):
        # Create base classifier
        base_svm = LinearSVC(
            dual=False,
            C=1.3,               
            class_weight='balanced',
            max_iter=2000,
            tol=1e-4
        )
        
        # Wrap with CalibratedClassifierCV for better probability estimates
        self.calibrated_svc = CalibratedClassifierCV(base_svm, cv=5)
        
        # Create pipeline
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.90,        # Slightly more restrictive
                sublinear_tf=True,
                strip_accents='unicode',
                analyzer='word',
                binary=False,       # Use term frequencies
                use_idf=True,
                smooth_idf=True,
                norm='l2'
            )),
            ('classifier', self.calibrated_svc)
        ])
    
    def train(self, X, y, validation_split=0.1):
        """Train the model with validation split"""
        # Split with stratification
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