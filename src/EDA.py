import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
import pandas as pd
import re
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class MultilingualNewsEDA:
    """
    A comprehensive EDA class for multilingual news datasets.
    
    This class provides methods for text cleaning, language detection,
    and various visualizations for news article analysis.
    """
    
    def __init__(self, df: pd.DataFrame, text_column: str = 'headline', 
                 body_column: str = 'body_text'):
        """
        Initialize the EDA pipeline.
        
        Args:
            df: Input DataFrame
            text_column: Name of the main text column (default: 'headline')
            body_column: Name of the body text column (default: 'body_text')
        """
        self.df = df.copy()
        self.text_column = text_column
        self.body_column = body_column
        self.spacy_models = {}
        self._load_nlp_libraries()
        
    def _load_nlp_libraries(self):
        """Load NLP libraries and models."""
        try:
            import spacy
            from langdetect import detect
            self.detect_lang = detect
            
            # Available spaCy models
            model_mapping = {
                'en': 'en_core_web_sm',
                'de': 'de_core_news_sm', 
                'fr': 'fr_core_news_sm',
                'es': 'es_core_news_sm',
                'ru': 'ru_core_news_sm',
                'zh-cn': 'zh_core_web_sm'
            }
            
            for lang, model_name in model_mapping.items():
                try:
                    self.spacy_models[lang] = spacy.load(model_name)
                except OSError:
                    continue
                    
        except ImportError:
            print("Warning: spaCy or langdetect not installed. Limited multilingual support.")
            self.detect_lang = None
    
    def detect_language(self, text: str) -> str:
        """Detect language of given text."""
        if self.detect_lang is None:
            return 'unknown'
        try:
            return self.detect_lang(str(text))
        except Exception:
            return 'unknown'
    
    def clean_text(self, text: str, lang: str = 'en') -> str:
        """
        Clean and preprocess text.
        
        Args:
            text: Input text
            lang: Language code
            
        Returns:
            Cleaned text string
        """
        text = str(text).lower()
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Use spaCy if available for the language
        if lang in self.spacy_models:
            nlp = self.spacy_models[lang]
            doc = nlp(text)
            tokens = [token.lemma_ for token in doc 
                     if not token.is_stop and not token.is_punct and not token.is_space]
            return ' '.join(tokens)
        else:
            # Basic cleaning
            text = re.sub(r'[^\w\s]', '', text)
            return text
    
    def prepare_data(self) -> None:
        """Prepare data by adding language detection and cleaned text columns."""
        print("Preparing data...")
        
        # Language detection
        if 'lang' not in self.df.columns:
            print("Detecting languages...")
            self.df['lang'] = self.df[self.text_column].apply(self.detect_language)
        
        # Clean text columns
        clean_text_col = f'{self.text_column}_clean'
        if clean_text_col not in self.df.columns:
            print(f"Creating '{clean_text_col}' column...")
            self.df[clean_text_col] = [
                self.clean_text(text, lang) 
                for text, lang in zip(self.df[self.text_column], self.df['lang'])
            ]
        
        # Clean body text if available
        if self.body_column in self.df.columns:
            clean_body_col = f'{self.body_column}_clean'
            if clean_body_col not in self.df.columns:
                print(f"Creating '{clean_body_col}' column...")
                self.df[clean_body_col] = [
                    self.clean_text(text, lang) 
                    for text, lang in zip(self.df[self.body_column], self.df['lang'])
                ]
        
        print("Data preparation complete!")
    
    def show_basic_info(self) -> None:
        """Display basic dataset information."""
        print("="*50)
        print("DATASET OVERVIEW")
        print("="*50)
        print(f"Shape: {self.df.shape}")
        print(f"Available columns: {list(self.df.columns)}")
        print(f"\nMissing values:")
        print(self.df.isnull().sum())
        print("\n")
    
    def plot_language_distribution(self, figsize: Tuple[int, int] = (10, 6)) -> None:
        """Plot distribution of languages."""
        if 'lang' not in self.df.columns:
            print("Language column not found. Run prepare_data() first.")
            return
            
        plt.figure(figsize=figsize)
        lang_counts = self.df['lang'].value_counts()
        lang_counts.plot(kind='bar', color='skyblue', edgecolor='black')
        plt.title('Article Count by Language', fontsize=14, fontweight='bold')
        plt.xlabel('Language')
        plt.ylabel('Number of Articles')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        print("Language Distribution:")
        print(lang_counts)
        print("\n")
    
    def plot_categorical_distribution(self, column: str, 
                                    figsize: Tuple[int, int] = (10, 6),
                                    color: str = 'salmon') -> None:
        """Plot distribution of a categorical column."""
        if column not in self.df.columns:
            print(f"Column '{column}' not found in dataset.")
            return
            
        plt.figure(figsize=figsize)
        counts = self.df[column].value_counts()
        counts.plot(kind='bar', color=color, edgecolor='black')
        plt.title(f'Distribution of {column.replace("_", " ").title()}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel(column.replace("_", " ").title())
        plt.ylabel('Number of Articles')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        print(f"{column} Distribution:")
        print(counts)
        print("\n")
    
    def plot_text_length_distribution(self, column: str = None,
                                    figsize: Tuple[int, int] = (10, 6)) -> None:
        """Plot distribution of text lengths."""
        if column is None:
            column = f'{self.body_column}_clean'
            
        if column not in self.df.columns:
            print(f"Column '{column}' not found. Run prepare_data() first.")
            return
            
        # Calculate word counts
        length_col = f'{column}_length'
        self.df[length_col] = self.df[column].str.split().str.len()
        
        plt.figure(figsize=figsize)
        sns.histplot(self.df[length_col].dropna(), bins=50, color='purple', edgecolor='black')
        plt.title(f'Distribution of {column.replace("_", " ").title()} Lengths', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Number of Words')  
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()
        
        avg_length = self.df[length_col].mean()
        median_length = self.df[length_col].median()
        print(f"Average {column} length: {avg_length:.2f} words")
        print(f"Median {column} length: {median_length:.2f} words")
        print("\n")
    
    def plot_cross_analysis(self, col1: str, col2: str,
                          figsize: Tuple[int, int] = (10, 6)) -> None:
        """Create heatmap for cross-analysis of two categorical columns."""
        if col1 not in self.df.columns or col2 not in self.df.columns:
            print(f"One or both columns '{col1}', '{col2}' not found.")
            return
            
        cross_tab = pd.crosstab(self.df[col1], self.df[col2])
        plt.figure(figsize=figsize)
        sns.heatmap(cross_tab, annot=True, fmt='d', cmap='YlGnBu', 
                   cbar_kws={'label': 'Count'})
        plt.title(f'Cross-Analysis: {col1.replace("_", " ").title()} vs {col2.replace("_", " ").title()}',
                 fontsize=14, fontweight='bold')
        plt.ylabel(col1.replace("_", " ").title())
        plt.xlabel(col2.replace("_", " ").title())
        plt.tight_layout()
        plt.show()
        print("\n")
    
    def analyze_common_words(self, column: str = None, top_n: int = 20,
                           figsize: Tuple[int, int] = (12, 6)) -> List[Tuple[str, int]]:
        """Analyze and plot most common words."""
        if column is None:
            column = f'{self.text_column}_clean'
            
        if column not in self.df.columns:
            print(f"Column '{column}' not found. Run prepare_data() first.")
            return []
        
        # Get all words
        all_words = ' '.join(self.df[column].dropna()).split()
        common_words = Counter(all_words).most_common(top_n)
        
        if not common_words:
            print("No words found.")
            return []
        
        words, counts = zip(*common_words)
        
        plt.figure(figsize=figsize)
        sns.barplot(x=list(counts), y=list(words), palette='mako', orient='h')
        plt.title(f'Top {top_n} Most Common Words in {column.replace("_", " ").title()}',
                 fontsize=14, fontweight='bold')
        plt.xlabel('Frequency')
        plt.ylabel('Words')
        plt.tight_layout()
        plt.show()
        
        print(f"Top {top_n} words in {column}:")
        for word, count in common_words:
            print(f"  {word}: {count}")
        print("\n")
        
        return common_words
    
    def analyze_words_by_category(self, category_col: str, text_col: str = None,
                                top_categories: int = 5, top_words: int = 10,
                                figsize: Tuple[int, int] = (10, 6)) -> None:
        """Analyze common words by category (e.g., language, political leaning)."""
        if text_col is None:
            text_col = f'{self.text_column}_clean'
            
        if category_col not in self.df.columns or text_col not in self.df.columns:
            print(f"Required columns not found.")
            return
        
        print(f"\nMost common words in {text_col.replace('_', ' ')} by {category_col.replace('_', ' ')}:")
        
        # Get top categories by frequency
        top_cats = self.df[category_col].value_counts().index[:top_categories]
        
        for category in top_cats:
            category_data = self.df[self.df[category_col] == category]
            words = ' '.join(category_data[text_col].dropna()).split()
            common_words = Counter(words).most_common(top_words)
            
            print(f"\n{category}:")
            for word, count in common_words:
                print(f"  {word}: {count}")
            
            if common_words:
                words_list, counts_list = zip(*common_words)
                plt.figure(figsize=figsize)
                sns.barplot(x=list(counts_list), y=list(words_list), 
                           palette='viridis', orient='h')
                plt.title(f'Top {top_words} Words - {category}',
                         fontsize=14, fontweight='bold')
                plt.xlabel('Frequency')
                plt.ylabel('Words')
                plt.tight_layout()
                plt.show()
    
    def show_sample_data(self, n_samples: int = 5, random_state: int = 42) -> None:
        """Display random samples from the dataset."""
        cols_to_show = []
        priority_cols = [self.text_column, 'lang', 'political_leaning', 'topic', 
                        f'{self.text_column}_clean']
        
        for col in priority_cols:
            if col in self.df.columns:
                cols_to_show.append(col)
        
        if len(cols_to_show) >= 2:
            print("Random samples from dataset:")
            sample_df = self.df[cols_to_show].sample(min(n_samples, len(self.df)), 
                                                   random_state=random_state)
            print(sample_df.to_string())
        print("\n")
    
    def run_complete_eda(self) -> None:
        """Run the complete EDA pipeline."""
        print("🚀 Starting Comprehensive EDA Pipeline")
        print("="*60)
        
        # 1. Basic info
        self.show_basic_info()
        
        # 2. Prepare data
        self.prepare_data()
        
        # 3. Language analysis
        self.plot_language_distribution()
        
        # 4. Other categorical distributions
        categorical_cols = ['political_leaning', 'topic', 'source']
        for col in categorical_cols:
            if col in self.df.columns:
                self.plot_categorical_distribution(col)
        
        # 5. Text length analysis
        self.plot_text_length_distribution()
        
        # 6. Cross-analysis
        if 'lang' in self.df.columns and 'political_leaning' in self.df.columns:
            self.plot_cross_analysis('lang', 'political_leaning')
        
        # 7. Word frequency analysis
        self.analyze_common_words()
        
        # 8. Words by category
        if 'lang' in self.df.columns:
            self.analyze_words_by_category('lang')
        
        if 'political_leaning' in self.df.columns:
            self.analyze_words_by_category('political_leaning')
        
        # 9. Sample data
        self.show_sample_data()
        
        print("✅ EDA Complete! Review the outputs above to inform your next steps.")


# Usage Example:
def run_eda_pipeline(df, text_column='headline', body_column='body_text'):
    """
    Convenience function to run EDA on a news dataset.
    
    Args:
        df: Input DataFrame
        text_column: Name of main text column
        body_column: Name of body text column
    
    Example:
        # Load your data
        df = pd.read_csv('your_news_data.csv')
        
        # Run EDA
        run_eda_pipeline(df, text_column='headline', body_column='body_text')
    """
    eda = MultilingualNewsEDA(df, text_column, body_column)
    eda.run_complete_eda()
    return eda

# If you want to run individual components:
# eda = MultilingualNewsEDA(df)
# eda.prepare_data()
# eda.plot_language_distribution()
# eda.analyze_common_words()
# etc.