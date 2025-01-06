import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def download_nltk_data():
    """Download required NLTK data"""
    required_packages = [
        'punkt',
        'stopwords',
        'wordnet',
        'punkt_tab',
        'averaged_perceptron_tagger',
        'omw-1.4'
    ]
    
    print("Downloading required NLTK packages...")
    for package in required_packages:
        try:
            nltk.download(package)
            print(f"Successfully downloaded {package}")
        except Exception as e:
            print(f"Error downloading {package}: {str(e)}")
    
    print("\nSetup complete!")

if __name__ == "__main__":
    download_nltk_data()