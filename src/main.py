import os
import gc
from preprocessing import TextPreprocessor
from model import HateSpeechClassifier
from utils import load_data_in_chunks, evaluate_model, save_predictions

def main():
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Load and preprocess training data
    print("Loading and preprocessing training data...")
    train_data = load_data_in_chunks('data/train.csv')
    
    # Preprocess text
    train_data['processed_text'] = train_data['text'].apply(preprocessor.preprocess)
    
    # Free up memory
    X = train_data['processed_text']
    y = train_data['label']
    del train_data
    gc.collect()
    
    # Initialize and train model
    print("\nTraining model...")
    classifier = HateSpeechClassifier()  # Removed max_features parameter
    X_val, y_val, val_predictions = classifier.train(X, y)
    
    # Evaluate model
    weighted_f1 = evaluate_model(y_val, val_predictions)
    
    # Free up training data
    del X, y, X_val, y_val, val_predictions
    gc.collect()
    
    # Process test data if validation results are good
    if weighted_f1 >= 0.80:
        print("\nProcessing test data...")
        test_data = load_data_in_chunks('data/test_no_labels.csv')
        test_data['processed_text'] = test_data['text'].apply(preprocessor.preprocess)
        
        # Make predictions
        print("Making predictions...")
        predictions = classifier.predict(test_data['processed_text'])
        
        # Save predictions
        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'test_with_label.csv')
        save_predictions(test_data, predictions, output_path)

if __name__ == "__main__":
    main()