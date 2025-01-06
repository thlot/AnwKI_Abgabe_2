import pandas as pd
from sklearn.metrics import classification_report, f1_score
import gc

def load_data_in_chunks(filepath, chunk_size=5000):
    """Load and process data in chunks to manage memory"""
    chunks = []
    
    for chunk in pd.read_csv(filepath, chunksize=chunk_size):
        chunks.append(chunk)
        
    # Concatenate chunks
    data = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()
    
    return data

def evaluate_model(y_true, y_pred):
    """Print detailed evaluation metrics"""
    print("\nModel Evaluation:")
    print("-" * 50)
    print(classification_report(y_true, y_pred))
    
    # Calculate weighted F1 score
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"\nWeighted F1 Score: {weighted_f1:.4f}")
    
    if weighted_f1 >= 0.90:
        print("✨ Congratulations! The model achieved the target F1 score of 0.90 or higher!")
    elif weighted_f1 >= 0.85:
        print("🎉 Great! The model achieved an F1 score of 0.85 or higher!")
    elif weighted_f1 >= 0.80:
        print("👍 Good! The model achieved an F1 score of 0.80 or higher!")
    else:
        print("The model's performance might need improvement to reach the target F1 score.")
    
    return weighted_f1

def save_predictions(test_data, predictions, output_path):
    """Save predictions to CSV file"""
    test_data['label'] = predictions
    test_data.to_csv(output_path, index=False)
    print(f"\nPredictions saved to {output_path}")