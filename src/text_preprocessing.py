# Multilingual Text Processing Library for Google Colab
# Reusable setup for robust text processing across multiple languages

import os
import sys
import re
import warnings
warnings.filterwarnings('ignore')

class MultilingualTextProcessor:
    """
    A comprehensive text processing class that handles multiple languages
    with graceful fallbacks and error handling.
    """
    
    def __init__(self, install_dependencies=True, verbose=True):
        self.verbose = verbose
        self.polyglot_available = False
        self.language_detector = None
        self.spacy_models = {}
        self.stopwords_dict = {}
        self.lemmatizer = None
        
        if install_dependencies:
            self.setup_environment()
    
    def log(self, message, success=None):
        """Log messages with optional success/failure indicators"""
        if not self.verbose:
            return
        
        if success is True:
            print(f"✓ {message}")
        elif success is False:
            print(f"✗ {message}")
        else:
            print(message)
    
    def setup_environment(self):
        """Complete environment setup with all dependencies"""
        self.log("=== Multilingual Text Processing Setup ===")
        
        # Step 1: System dependencies
        self._install_system_dependencies()
        
        # Step 2: Python packages
        self._install_python_packages()
        
        # Step 3: SpaCy models
        self._download_spacy_models()
        
        # Step 4: NLTK resources
        self._setup_nltk()
        
        # Step 5: Initialize components
        self._initialize_components()
        
        self._print_summary()
    
    def _install_system_dependencies(self):
        """Install system-level dependencies"""
        self.log("Installing system dependencies...")
        
        try:
            os.system("apt-get update > /dev/null 2>&1")
            result = os.system("apt-get install -y build-essential python3-dev libicu-dev > /dev/null 2>&1")
            self.log("System dependencies installed", result == 0)
        except Exception as e:
            self.log(f"System dependency installation issues: {e}", False)
    
    def _install_python_packages(self):
        """Install required Python packages"""
        self.log("Installing Python packages...")
        
        # Core packages
        core_packages = [
            "spacy", "nltk", "pandas", "tqdm", 
            "langdetect", "textblob"
        ]
        
        for package in core_packages:
            try:
                result = os.system(f"pip install {package} > /dev/null 2>&1")
                self.log(f"{package} installed", result == 0)
            except Exception as e:
                self.log(f"Failed to install {package}: {e}", False)
        
        # Try polyglot with fallback
        self._try_install_polyglot()
    
    def _try_install_polyglot(self):
        """Attempt to install polyglot with graceful fallback"""
        self.log("Attempting to install polyglot...")
        
        try:
            os.system("apt-get install -y libicu-dev > /dev/null 2>&1")
            pyicu_result = os.system("pip install pyicu > /dev/null 2>&1")
            
            if pyicu_result == 0:
                polyglot_result = os.system("pip install polyglot > /dev/null 2>&1")
                if polyglot_result == 0:
                    self.polyglot_available = True
                    self.log("Polyglot installed successfully", True)
                else:
                    self.log("Polyglot installation failed, using langdetect", False)
            else:
                self.log("PyICU installation failed, using langdetect", False)
                
        except Exception as e:
            self.log(f"Polyglot installation error: {e}", False)
    
    def _download_spacy_models(self):
        """Download spaCy language models"""
        self.log("Downloading spaCy models...")
        
        models_config = [
            ('en_core_web_sm', 'English'),
            ('de_core_news_sm', 'German'),
            ('fr_core_news_sm', 'French'),
            ('es_core_news_sm', 'Spanish'),
            ('ru_core_news_sm', 'Russian'),
            ('zh_core_web_sm', 'Chinese'),
            ('it_core_news_sm', 'Italian'),
            ('pt_core_news_sm', 'Portuguese')
        ]
        
        self.successful_models = []
        for model, language in models_config:
            try:
                result = os.system(f"python -m spacy download {model} > /dev/null 2>&1")
                if result == 0:
                    self.successful_models.append(model)
                    self.log(f"{language} model ({model})", True)
                else:
                    self.log(f"{language} model ({model}) failed", False)
            except Exception as e:
                self.log(f"Error downloading {language} model: {e}", False)
    
    def _setup_nltk(self):
        """Setup NLTK resources"""
        self.log("Setting up NLTK resources...")
        
        import nltk
        
        resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' 
                              else f'corpora/{resource}')
                self.log(f"NLTK {resource} available", True)
            except LookupError:
                try:
                    nltk.download(resource, quiet=True)
                    self.log(f"NLTK {resource} downloaded", True)
                except Exception as e:
                    self.log(f"NLTK {resource} failed: {e}", False)
    
    def _initialize_components(self):
        """Initialize all text processing components"""
        self.log("Initializing components...")
        
        # Language detection
        self._init_language_detection()
        
        # SpaCy models
        self._init_spacy_models()
        
        # NLTK components
        self._init_nltk_components()
    
    def _init_language_detection(self):
        """Initialize language detection"""
        if self.polyglot_available:
            try:
                from polyglot.detect import Detector
                self.language_detector = "polyglot"
                self.log("Language detection: Polyglot", True)
            except ImportError:
                self._fallback_to_langdetect()
        else:
            self._fallback_to_langdetect()
    
    def _fallback_to_langdetect(self):
        """Fallback to langdetect for language detection"""
        try:
            from langdetect import detect
            self.language_detector = "langdetect"
            self.log("Language detection: langdetect", True)
        except ImportError:
            self.language_detector = None
            self.log("No language detection available", False)
    
    def _init_spacy_models(self):
        """Initialize spaCy models"""
        import spacy
        
        model_mapping = {
            'en_core_web_sm': 'en',
            'de_core_news_sm': 'de',
            'fr_core_news_sm': 'fr',
            'es_core_news_sm': 'es',
            'ru_core_news_sm': 'ru',
            'zh_core_web_sm': 'zh',
            'it_core_news_sm': 'it',
            'pt_core_news_sm': 'pt'
        }
        
        for model_name in self.successful_models:
            try:
                lang_code = model_mapping.get(model_name, model_name[:2])
                self.spacy_models[lang_code] = spacy.load(model_name)
                self.log(f"SpaCy model loaded: {lang_code}", True)
            except Exception as e:
                self.log(f"Failed to load {model_name}: {e}", False)
    
    def _init_nltk_components(self):
        """Initialize NLTK components"""
        try:
            from nltk.corpus import stopwords
            from nltk.stem import WordNetLemmatizer
            
            # Stopwords
            available_languages = stopwords.fileids()
            for lang in available_languages:
                try:
                    self.stopwords_dict[lang] = set(stopwords.words(lang))
                except:
                    continue
            
            # Lemmatizer
            self.lemmatizer = WordNetLemmatizer()
            self.log("NLTK components initialized", True)
            
        except Exception as e:
            self.log(f"NLTK initialization failed: {e}", False)
    
    def detect_language(self, text):
        """
        Detect the language of input text
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            str: Language code (e.g., 'en', 'de', 'fr') or 'unknown'
        """
        try:
            if not isinstance(text, str) or not text.strip():
                return 'unknown'
            
            if self.language_detector == "polyglot":
                from polyglot.detect import Detector
                detector = Detector(text, quiet=True)
                return detector.language.code
                
            elif self.language_detector == "langdetect":
                from langdetect import detect
                return detect(text)
                
            else:
                return 'en'  # Default fallback
                
        except Exception:
            return 'unknown'
    
    def clean_text(self, text, lang=None, remove_urls=True, remove_punctuation=True):
        """
        Clean and process text with language-specific processing
        
        Args:
            text (str): Input text to clean
            lang (str): Language code (auto-detected if None)
            remove_urls (bool): Whether to remove URLs
            remove_punctuation (bool): Whether to remove punctuation
            
        Returns:
            str: Cleaned text
        """
        try:
            text = str(text).lower().strip()
            
            if not text:
                return ""
            
            # Auto-detect language if not provided
            if lang is None:
                lang = self.detect_language(text)
            
            # Remove URLs if requested
            if remove_urls:
                text = re.sub(r'http\S+|www.\S+', '', text)
            
            # Normalize whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            if not text:
                return ""
            
            # Language-specific processing
            return self._process_by_language(text, lang, remove_punctuation)
            
        except Exception as e:
            if self.verbose:
                print(f"Warning: Error cleaning text: {e}")
            return str(text)
    
    def _process_by_language(self, text, lang, remove_punctuation):
        """Process text using language-specific tools"""
        
        # SpaCy processing for supported languages
        if lang in self.spacy_models:
            try:
                nlp = self.spacy_models[lang]
                doc = nlp(text)
                tokens = []
                
                for token in doc:
                    if (not token.is_punct and not token.is_space and 
                        not token.is_stop and len(token.text) > 1):
                        tokens.append(token.lemma_)
                
                return ' '.join(tokens)
                
            except Exception:
                pass  # Fall through to basic cleaning
        
        # NLTK processing for English
        if lang == 'en' and self.lemmatizer:
            try:
                import nltk
                
                if remove_punctuation:
                    text = re.sub(r'[^a-z\s]', '', text)
                
                tokens = nltk.word_tokenize(text)
                tokens = [
                    self.lemmatizer.lemmatize(word) 
                    for word in tokens
                    if (word not in self.stopwords_dict.get('english', set()) and 
                        len(word) > 1)
                ]
                return ' '.join(tokens)
                
            except Exception:
                pass  # Fall through to basic cleaning
        
        # Basic cleaning fallback
        if remove_punctuation:
            text = re.sub(r'[^\w\s]', '', text)
        
        return text
    
    def process_dataframe(self, df, text_columns=None, detect_lang=True, 
                         clean_text=True, add_word_count=False):
        """
        Process a pandas DataFrame with text data
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_columns (list): Column names to process (all text columns if None)
            detect_lang (bool): Whether to add language detection
            clean_text (bool): Whether to clean text
            add_word_count (bool): Whether to add word count columns
            
        Returns:
            pd.DataFrame: Processed dataframe
        """
        try:
            import pandas as pd
            from tqdm import tqdm
            
            df = df.copy()
            
            # Auto-detect text columns if not specified
            if text_columns is None:
                text_columns = df.select_dtypes(include=['object']).columns.tolist()
            
            # Ensure columns exist
            text_columns = [col for col in text_columns if col in df.columns]
            
            if not text_columns:
                self.log("No text columns found for processing", False)
                return df
            
            self.log(f"Processing columns: {text_columns}")
            
            # Language detection
            if detect_lang and text_columns:
                self.log("Detecting languages...")
                primary_col = text_columns[0]
                df['detected_language'] = df[primary_col].apply(
                    lambda x: self.detect_language(str(x)) if pd.notna(x) else 'unknown'
                )
            
            # Text cleaning
            if clean_text:
                self.log("Cleaning text...")
                for col in text_columns:
                    new_col = f"{col}_clean"
                    
                    if detect_lang and 'detected_language' in df.columns:
                        df[new_col] = [
                            self.clean_text(str(text), lang) 
                            for text, lang in zip(df[col], df['detected_language'])
                        ]
                    else:
                        df[new_col] = df[col].apply(
                            lambda x: self.clean_text(str(x)) if pd.notna(x) else ""
                        )
            
            # Word counts
            if add_word_count:
                self.log("Adding word counts...")
                for col in text_columns:
                    count_col = f"{col}_word_count"
                    source_col = f"{col}_clean" if clean_text else col
                    
                    if source_col in df.columns:
                        df[count_col] = df[source_col].apply(
                            lambda x: len(str(x).split()) if pd.notna(x) else 0
                        )
            
            self.log("DataFrame processing complete!", True)
            return df
            
        except Exception as e:
            self.log(f"Error processing DataFrame: {e}", False)
            return df
    
    def _print_summary(self):
        """Print setup summary"""
        self.log("\n" + "="*60)
        self.log("SETUP COMPLETE - MULTILINGUAL TEXT PROCESSOR")
        self.log("="*60)
        self.log(f"Language Detection: {self.language_detector or 'Not available'}")
        self.log(f"SpaCy Models: {list(self.spacy_models.keys()) or 'None'}")
        self.log(f"NLTK Lemmatizer: {'Available' if self.lemmatizer else 'Not available'}")
        self.log(f"Stopwords Languages: {list(self.stopwords_dict.keys()) or 'None'}")
        self.log("="*60)
    
    def get_supported_languages(self):
        """Return list of supported languages"""
        return {
            'spacy_models': list(self.spacy_models.keys()),
            'stopwords': list(self.stopwords_dict.keys()),
            'language_detection': self.language_detector is not None
        }


# Convenience function for quick setup
def setup_multilingual_processor(verbose=True):
    """
    Quick setup function - creates and returns a configured processor
    
    Args:
        verbose (bool): Whether to show setup progress
        
    Returns:
        MultilingualTextProcessor: Configured processor instance
    """
    return MultilingualTextProcessor(install_dependencies=True, verbose=verbose)


# Example usage functions
def example_basic_usage():
    """Example of basic text processing"""
    # Initialize processor
    processor = setup_multilingual_processor()
    
    # Process individual text
    text = "This is a sample English text with some noise! Visit https://example.com"
    cleaned = processor.clean_text(text)
    lang = processor.detect_language(text)
    
    print(f"Original: {text}")
    print(f"Cleaned: {cleaned}")
    print(f"Language: {lang}")


def example_dataframe_usage():
    """Example of DataFrame processing"""
    import pandas as pd
    
    # Initialize processor
    processor = setup_multilingual_processor()
    
    # Sample data
    data = {
        'title': [
            'Breaking News: Important Update',
            'Neueste Nachrichten aus Deutschland',
            'Dernières nouvelles de France',
            'Últimas noticias de España'
        ],
        'content': [
            'This is English content with details...',
            'Dies ist deutscher Inhalt mit Details...',
            'Ceci est du contenu français avec des détails...',
            'Este es contenido español con detalles...'
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Process DataFrame
    processed_df = processor.process_dataframe(
        df, 
        text_columns=['title', 'content'],
        detect_lang=True,
        clean_text=True,
        add_word_count=True
    )
    
    print("Processed DataFrame:")
    print(processed_df.head())
    
    return processed_df


if __name__ == "__main__":
    # Run examples
    print("=== BASIC USAGE EXAMPLE ===")
    example_basic_usage()
    
    print("\n=== DATAFRAME USAGE EXAMPLE ===")
    example_dataframe_usage()