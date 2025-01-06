import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        # Get standard stopwords but remove negative words
        self.stop_words = set(stopwords.words('english')) - {'no', 'not', 'nor'}
    
    def preprocess(self, text):
        """Clean and preprocess text data"""
        try:
            if not isinstance(text, str):
                return ""
            
            # Handle empty or very short texts
            if not text or len(text.strip()) < 2:
                return ""
                
            # Convert to lowercase
            text = text.lower()
            
            # Handle URLs
            text = re.sub(r'http\S+|www\S+', '', text)
            
            # Remove special characters but keep some punctuation
            text = re.sub(r'[^a-zA-Z!?\s]', '', text)
            
            # Replace multiple spaces
            text = re.sub(r'\s+', ' ', text)
            
            # Tokenize
            tokens = word_tokenize(text)
            
            # Remove stopwords and lemmatize
            tokens = [
                self.lemmatizer.lemmatize(token) 
                for token in tokens 
                if token not in self.stop_words or token in {'!', '?'}
            ]
            
            return ' '.join(tokens)
            
        except Exception as e:
            print(f"Error preprocessing text: {str(e)}")
            return ""