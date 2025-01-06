from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np

class HateSpeechClassifier:
    def __init__(self, max_features=15000):  # Increased features
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 3),  # Added trigrams
                min_df=2,
                max_df=0.95,
                sublinear_tf=True,    # Apply sublinear scaling
                strip_accents='unicode',
                norm='l2'
            )),
            ('classifier', LinearSVC(
                dual=False,
                C=1.0,
                class_weight='balanced',
                max_iter=2000,
                tol=1e-4
            ))
        ])
        
        # Define parameter grid for optimization
        self.param_grid = {
            'tfidf__max_df': [0.95, 0.9],
            'tfidf__min_df': [2, 3],
            'classifier__C': [0.8, 1.0, 1.2],
        }
        
    def optimize_hyperparameters(self, X, y):
        """Optimize model hyperparameters using grid search"""
        grid_search = GridSearchCV(
            self.pipeline,
            self.param_grid,
            cv=3,
            scoring='f1_weighted',
            n_jobs=-1,  # Use all CPU cores
            verbose=1
        )
        grid_search.fit(X, y)
        
        # Use best parameters
        self.pipeline = grid_search.best_estimator_
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_score_
    
    def train(self, X, y, validation_split=0.1, optimize=True):
        """Train the model with optional hyperparameter optimization"""
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=validation_split, 
            random_state=42,
            stratify=y
        )
        
        if optimize:
            print("\nOptimizing hyperparameters...")
            cv_score = self.optimize_hyperparameters(X_train, y_train)
        
        # Train on full training set
        print("\nTraining final model...")
        self.pipeline.fit(X_train, y_train)
        
        # Validate
        val_predictions = self.pipeline.predict(X_val)
        return X_val, y_val, val_predictions
    
    def predict(self, X):
        """Make predictions on new data"""
        return self.pipeline.predict(X)