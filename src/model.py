from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectFromModel
import numpy as np
from scipy import stats

class EnsembleClassifier:
    def __init__(self):
        # Base vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=15000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.85,  # More restrictive
            sublinear_tf=True,
            strip_accents='unicode'
        )
        
        # SVM with different configurations
        self.svm1 = CalibratedClassifierCV(
            LinearSVC(
                dual=False,
                C=1.3,
                class_weight='balanced',
                max_iter=2000,
                tol=1e-4
            ),
            cv=5
        )
        
        self.svm2 = CalibratedClassifierCV(
            LinearSVC(
                dual=False,
                C=1.0,
                class_weight={0: 1.5, 1: 1.0, 2: 1.0},  # Focus on hate speech
                max_iter=2000,
                tol=1e-4
            ),
            cv=5
        )
        
        self.lr = LogisticRegression(
            C=1.2,
            class_weight='balanced',
            max_iter=1000,
            multi_class='multinomial',
            solver='lbfgs',
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
        
        # Feature selection based on SVM weights
        print("Performing feature selection...")
        temp_svm = LinearSVC(dual=False, random_state=42)
        temp_svm.fit(X_train_vec, y_train)
        selector = SelectFromModel(temp_svm, prefit=True, max_features=12000)
        X_train_selected = selector.transform(X_train_vec)
        X_val_selected = selector.transform(X_val_vec)
        
        # Train models with selected features
        print("Training SVM 1...")
        self.svm1.fit(X_train_selected, y_train)
        
        print("Training SVM 2...")
        self.svm2.fit(X_train_selected, y_train)
        
        print("Training Logistic Regression...")
        self.lr.fit(X_train_selected, y_train)
        
        # Get predictions for validation set
        svm1_proba = self.svm1.predict_proba(X_val_selected)
        svm2_proba = self.svm2.predict_proba(X_val_selected)
        lr_proba = self.lr.predict_proba(X_val_selected)
        
        # Weighted average of probabilities
        # Give more weight to SVM1 for hate speech class
        ensemble_proba = np.zeros_like(svm1_proba)
        for i in range(len(y_val)):
            if np.argmax(svm1_proba[i]) == 0:  # Hate speech class
                weights = [0.5, 0.3, 0.2]  # Higher weight for SVM1
            else:
                weights = [0.4, 0.3, 0.3]  # More balanced weights
                
            ensemble_proba[i] = (
                weights[0] * svm1_proba[i] +
                weights[1] * svm2_proba[i] +
                weights[2] * lr_proba[i]
            )
        
        val_predictions = np.argmax(ensemble_proba, axis=1)
        
        # Store feature selector for prediction
        self.selector = selector
        return X_val, y_val, val_predictions
    
    def predict(self, X):
        """Make predictions using the ensemble"""
        # Transform text data
        X_vec = self.vectorizer.transform(X)
        X_selected = self.selector.transform(X_vec)
        
        # Get predictions from all models
        svm1_proba = self.svm1.predict_proba(X_selected)
        svm2_proba = self.svm2.predict_proba(X_selected)
        lr_proba = self.lr.predict_proba(X_selected)
        
        # Combine predictions with dynamic weights
        ensemble_proba = np.zeros_like(svm1_proba)
        for i in range(len(X)):
            if np.argmax(svm1_proba[i]) == 0:  # Hate speech class
                weights = [0.5, 0.3, 0.2]
            else:
                weights = [0.4, 0.3, 0.3]
                
            ensemble_proba[i] = (
                weights[0] * svm1_proba[i] +
                weights[1] * svm2_proba[i] +
                weights[2] * lr_proba[i]
            )
        
        return np.argmax(ensemble_proba, axis=1)

class HateSpeechClassifier:
    def __init__(self):
        self.classifier = EnsembleClassifier()
    
    def train(self, X, y, validation_split=0.1):
        return self.classifier.train(X, y, validation_split)
    
    def predict(self, X):
        return self.classifier.predict(X)