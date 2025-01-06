import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = self._get_custom_stopwords()
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#\w+')
        
    def _get_custom_stopwords(self):
        """Get custom stopwords list with some words removed"""
        stop_words = set(stopwords.words('english'))
        # Keep negative words as they might be important for hate speech
        negative_words = {'no', 'not', 'nor', 'neither', 'never', 'none'}
        return stop_words - negative_words
    
    def _clean_text(self, text):
        """Clean text by removing URLs, mentions, hashtags"""
        text = self.url_pattern.sub('', text)
        text = self.mention_pattern.sub('', text)
        text = self.hashtag_pattern.sub('', text)
        return text
    
    def _normalize_text(self, text):
        """Normalize text by handling common patterns"""
        # Replace repeated characters (e.g., 'haaappy' -> 'happy')
        text = re.sub(r'(.)\1+', r'\1\1', text)
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def preprocess(self, text):
        """Clean and preprocess text data"""
        try:
            if not isinstance(text, str):
                return ""
            
            if not text or len(text.strip()) < 2:
                return ""
            
            # Initial cleaning
            text = text.lower()
            text = self._clean_text(text)
            text = self._normalize_text(text)
            
            # Remove special characters but keep exclamation marks and question marks
            # as they might indicate emotional content
            text = re.sub(r'[^a-zA-Z!?\s]', '', text)
            
            # Tokenize
            tokens = word_tokenize(text)
            
            # Remove stopwords and lemmatize
            tokens = [
                self.lemmatizer.lemmatize(token) 
                for token in tokens 
                if token not in self.stop_words or token in {'!', '?'}
            ]
            
            # Ensure some content remains after preprocessing
            if not tokens:
                return ""
            
            return ' '.join(tokens)
            
        except Exception as e:
            print(f"Error preprocessing text: {str(e)}")
            return ""