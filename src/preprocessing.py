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

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        # Customize stopwords
        self.stop_words = set(stopwords.words('english')) - {
            'no', 'not', 'nor', 'against', 'down', 'out', 'off', 'over', 
            'under', 'more', 'most', 'other', 'some', 'such', 'very'
        }
        
        # Regular expressions for cleaning
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.user_pattern = re.compile(r'@[\w_]+')
        self.emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "]+", flags=re.UNICODE)
    
    def _normalize_repeating(self, text):
        """Normalize repeating characters but preserve meaning"""
        # Replace 3 or more repetitions with 2 repetitions
        return re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    def preprocess(self, text):
        """Clean and preprocess text data"""
        try:
            if not isinstance(text, str) or not text or len(text.strip()) < 2:
                return ""
                
            # Basic cleaning
            text = text.lower().strip()
            
            # Remove URLs, @mentions, and emojis
            text = self.url_pattern.sub('', text)
            text = self.user_pattern.sub('', text)
            text = self.emoji_pattern.sub('', text)
            
            # Normalize repeating characters
            text = self._normalize_repeating(text)
            
            # Replace special characters but keep some meaningful punctuation
            text = re.sub(r'[^a-zA-Z!?.,\s]', '', text)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Tokenize
            tokens = word_tokenize(text)
            
            # Remove stopwords and lemmatize
            processed_tokens = []
            for token in tokens:
                # Keep punctuation as is
                if token in {'!', '?', '.', ','}:
                    processed_tokens.append(token)
                # Process other tokens
                elif token not in self.stop_words:
                    lemmatized = self.lemmatizer.lemmatize(token)
                    processed_tokens.append(lemmatized)
            
            # Ensure some content remains
            if not processed_tokens:
                return ""
            
            return ' '.join(processed_tokens)
            
        except Exception as e:
            print(f"Error preprocessing text: {str(e)}")
            return ""