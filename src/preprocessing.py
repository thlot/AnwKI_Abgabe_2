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
        # Custom stopwords for hate speech detection
        base_stopwords = set(stopwords.words('english'))
        self.keep_words = {
            'no', 'not', 'nor', 'against', 'hate', 'cant', 'cannot',
            'never', 'none', 'nothing', 'nobody', 'down', 'off', 'over',
            'under', 'more', 'most', 'other', 'some', 'such', 'very',
            'own', 'same', 'few', 'dont', 'won', 'ain', 'aren', 'couldn',
            'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'shouldn',
            'wasn', 'weren', 'wouldn'
        }
        self.stop_words = base_stopwords - self.keep_words
        
        # Compile regex patterns
        self.url_pattern = re.compile(r'http[s]?://\S+|www\.\S+')
        self.user_pattern = re.compile(r'@\w+')
        self.emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "]+", flags=re.UNICODE)
        
        # Hate speech specific patterns
        self.repeat_pattern = re.compile(r'(.)\1{2,}')
        self.emphasis_pattern = re.compile(r'[!?]{2,}')
    
    def _clean_repeating(self, text):
        """Handle repeating characters while preserving emphasis"""
        # Replace 3+ repetitions with 2 (preserves emphasis)
        text = self.repeat_pattern.sub(r'\1\1', text)
        # Replace multiple punctuation with single but preserve !! and ??
        text = self.emphasis_pattern.sub('!!', text)
        return text
    
    def _add_special_tokens(self, text):
        """Add special tokens for specific patterns"""
        # Add tokens for shouting (ALL CAPS)
        if any(c.isupper() for c in text) and len(text) > 3:
            text += ' <SHOUTING>'
        # Add token for emphasis
        if '!!' in text or '??' in text:
            text += ' <EMPHASIS>'
        return text
    
    def preprocess(self, text):
        """Clean and preprocess text data with focus on hate speech detection"""
        try:
            if not isinstance(text, str) or not text or len(text.strip()) < 2:
                return ""
            
            # preserve original case for shouting detection
            has_shouting = text.isupper() and len(text) > 3
            
            # Basic cleaning
            text = text.lower().strip()
            
            # Remove URLs, mentions, and emojis
            text = self.url_pattern.sub('', text)
            text = self.user_pattern.sub('', text)
            text = self.emoji_pattern.sub('', text)
            
            # Handle repeating characters
            text = self._clean_repeating(text)
            
            # Keep only letters, numbers, and important punctuation
            text = re.sub(r'[^a-zA-Z0-9!?,.\s]', ' ', text)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Tokenize
            tokens = word_tokenize(text)
            
            # Process tokens
            processed_tokens = []
            for token in tokens:
                if token in {'!', '?', '.', ','}:
                    processed_tokens.append(token)
                elif (token not in self.stop_words) or (token in self.keep_words):
                    lemmatized = self.lemmatizer.lemmatize(token)
                    processed_tokens.append(lemmatized)
            
            # Join tokens
            processed_text = ' '.join(processed_tokens)
            
            # Add special tokens if needed
            if has_shouting:
                processed_text += ' <SHOUTING>'
            if '!!' in text or '??' in text:
                processed_text += ' <EMPHASIS>'
            
            return processed_text.strip()
            
        except Exception as e:
            print(f"Error preprocessing text: {str(e)}")
            return ""