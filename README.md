# Hate Speech Classification Project

This project implements a hate speech classifier using machine learning techniques. The goal is to classify text into three categories: hate speech, offensive language, or neither.

## Project Structure

```
ANWKI_ABGABE_2/
├── data/                    # Data directory
│   ├── train.csv           # Training data
│   └── test_no_labels.csv  # Test data
├── src/                    # Source code
│   ├── __init__.py
│   ├── main.py            # Main execution script
│   ├── preprocessing.py   # Text preprocessing
│   ├── model.py          # Model definition
│   └── utils.py          # Utility functions
├── output/                # Output directory for predictions
├── README.md             # Project documentation
├── requirements.txt      # Project dependencies
└── .gitignore           # Git ignore file
```

## Setup

1. Create and activate your conda environment:
```bash
conda create -n tlconda1 python=3.8
conda activate tlconda1
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your data files in the `data/` directory:
   - `train.csv`
   - `test_no_labels.csv`

2. Run the classifier:
```bash
python src/main.py
```

3. Find the predictions in `output/test_with_label.csv`

## Features

- Memory efficient data processing using chunks
- Text preprocessing including lemmatization and stopword removal
- TF-IDF vectorization with n-grams
- Linear SVM classifier with class balancing
- Model evaluation with weighted F1 score

## Model Details

The implementation uses:
- TF-IDF vectorization with unigrams and bigrams
- Linear SVM classifier
- Preprocessing steps:
  - Lowercase conversion
  - Special character removal
  - Stopword removal
  - Lemmatization

## Performance

The model aims for a weighted F1 score of:
- 0.90 or higher for maximum bonus points
- 0.85 or higher for medium bonus points
- 0.80 or higher for minimum bonus points