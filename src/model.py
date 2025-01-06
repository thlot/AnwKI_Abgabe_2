from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
from scipy import stats

class EnsembleClassifier:
    def __init__(self):
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=13000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.90,
            sublinear_tf=True,
            strip_accents='unicode'
        )
        
        # Create base classifiers
        self.svm = CalibratedClassifierCV(
            LinearSVC(
                dual=False,
                C=1.3,
                class_weight='balanced',
                max_iter=2000,
                tol=1e-4
            ),
            cv=5
        )
        
        self.lr = LogisticRegression(
            C=1.0,
            class_weight='balanced',
            max_iter=1000,
            multi_class='ovr',
            n_jobs=-1
        )
        
    def train(self, X, y, validation_split=0.1):
        """Train the ensemble model"""
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=validation_split,
            random_state=42,
            stratify=y
        )
        
        # Transform text data
        print("\nVectorizing text data...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_val_vec = self.vectorizer.transform(X_val)
        
        # Train models
        print("Training SVM...")
        self.svm.fit(X_train_vec, y_train)
        
        print("Training Logistic Regression...")
        self.lr.fit(X_train_vec, y_train)
        
        # Get predictions for validation set
        svm_pred = self.svm.predict(X_val_vec)
        lr_pred = self.lr.predict(X_val_vec)
        
        # Combine predictions using weighted voting
        svm_proba = self.svm.predict_proba(X_val_vec)
        lr_proba = self.lr.predict_proba(X_val_vec)
        
        # Use confidence-weighted voting
        ensemble_proba = 0.6 * svm_proba + 0.4 * lr_proba
        val_predictions = np.argmax(ensemble_proba, axis=1)
        
        return X_val, y_val, val_predictions
    
    def predict(self, X):
        """Make predictions using the ensemble"""
        # Transform text data
        X_vec = self.vectorizer.transform(X)
        
        # Get predictions from both models
        svm_proba = self.svm.predict_proba(X_vec)
        lr_proba = self.lr.predict_proba(X_vec)
        
        # Combine predictions
        ensemble_proba = 0.6 * svm_proba + 0.4 * lr_proba
        predictions = np.argmax(ensemble_proba, axis=1)
        
        return predictions

class HateSpeechClassifier:
    def __init__(self):
        self.classifier = EnsembleClassifier()
    
    def train(self, X, y, validation_split=0.1):
        return self.classifier.train(X, y, validation_split)
    
    def predict(self, X):
        return self.classifier.predict(X)