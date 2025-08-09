"""
News Spectrum Analyzer - A reusable pipeline for political perspective analysis
Provides topic clustering, political leaning classification, and multilingual summarization
"""

import pandas as pd
import numpy as np
import pickle
import joblib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsSpectrumAnalyzer:
    """
    A comprehensive pipeline for analyzing news articles across political spectrum

    Features:
    - Topic clustering using LDA
    - Political leaning classification
    - Multilingual abstractive summarization
    - Spectrum visualization and comparison
    """

    def __init__(self, model_dir: str = "models", n_topics: int = 5):
        """
        Initialize the analyzer

        Args:
            model_dir: Directory to save/load models
            n_topics: Number of topics for LDA clustering
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.n_topics = n_topics

        # Model components
        self.lda_model = None
        self.topic_vectorizer = None
        self.political_classifier = None
        self.political_vectorizer = None
        self.label_encoder = None
        self.summarizer = None

        # Initialize summarizer
        self._initialize_summarizer()

    def _initialize_summarizer(self):
        """Initialize the summarization pipeline"""
        try:
            from transformers import pipeline
            # Try multilingual model first
            try:
                self.summarizer = pipeline(
                    "summarization",
                    model="facebook/mbart-large-50-many-to-many-mmt"
                )
                logger.info("Using mBART for multilingual summarization")
            except Exception:
                self.summarizer = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn"
                )
                logger.info("Using BART for English summarization")
        except ImportError:
            logger.warning("Transformers not available. Summarization will use truncation.")
            self.summarizer = None

    def fit_topic_model(self, texts: List[str], max_features: int = 1000) -> 'NewsSpectrumAnalyzer':
        """
        Fit LDA topic model on provided texts

        Args:
            texts: List of text documents
            max_features: Maximum features for TF-IDF vectorization

        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting topic model with {len(texts)} documents")

        # Vectorize texts
        self.topic_vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )

        X_topics = self.topic_vectorizer.fit_transform(texts)

        # Fit LDA model
        self.lda_model = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=42,
            max_iter=10,
            learning_method='batch'
        )

        self.lda_model.fit(X_topics)

        # Display topics
        self._display_topics()

        return self

    def _display_topics(self):
        """Display top keywords for each topic"""
        if not self.lda_model or not self.topic_vectorizer:
            return

        feature_names = self.topic_vectorizer.get_feature_names_out()
        logger.info("Top keywords for each topic:")

        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_words_idx = topic.argsort()[:-11:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            logger.info(f"Topic {topic_idx}: {' | '.join(top_words)}")

    def fit_political_classifier(self, texts: List[str], labels: List[str],
                                test_size: float = 0.2) -> 'NewsSpectrumAnalyzer':
        """
        Train political leaning classifier

        Args:
            texts: Training texts
            labels: Political leaning labels
            test_size: Test set proportion

        Returns:
            Self for method chaining
        """
        logger.info(f"Training political classifier with {len(texts)} samples")

        # Prepare data
        df_train = pd.DataFrame({'text': texts, 'label': labels})
        df_train = df_train.dropna()

        # Remove unwanted labels
        unwanted = ['political_leaning', 'unknown', 'mixed', '']
        df_train = df_train[~df_train['label'].str.lower().isin(unwanted)]

        if len(df_train) == 0:
            raise ValueError("No valid training data after filtering")

        # Encode labels
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(df_train['label'])

        # Vectorize texts
        self.political_vectorizer = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.95
        )

        X = self.political_vectorizer.fit_transform(df_train['text'])

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42,
            stratify=y if len(np.unique(y)) > 1 else None
        )

        # Train classifier
        self.political_classifier = LogisticRegression(
            max_iter=200,
            class_weight='balanced',
            random_state=42
        )

        self.political_classifier.fit(X_train, y_train)

        # Evaluate
        if X_test.shape[0] > 0:
            y_pred = self.political_classifier.predict(X_test)
            report = classification_report(
                y_test, y_pred,
                target_names=self.label_encoder.classes_,
                output_dict=True
            )
            logger.info(f"Classifier accuracy: {report['accuracy']:.3f}")

        return self

    def predict_topics(self, texts: List[str]) -> np.ndarray:
        """
        Predict topics for given texts

        Args:
            texts: List of texts to analyze

        Returns:
            Array of topic IDs
        """
        if not self.lda_model or not self.topic_vectorizer:
            raise ValueError("Topic model not fitted. Call fit_topic_model() first.")

        X = self.topic_vectorizer.transform(texts)
        topic_distribution = self.lda_model.transform(X)
        return topic_distribution.argmax(axis=1)

    def predict_political_leaning(self, texts: List[str]) -> List[str]:
        """
        Predict political leaning for given texts

        Args:
            texts: List of texts to analyze

        Returns:
            List of predicted political leanings
        """
        if not self.political_classifier or not self.political_vectorizer:
            raise ValueError("Political classifier not fitted. Call fit_political_classifier() first.")

        X = self.political_vectorizer.transform(texts)
        predictions = self.political_classifier.predict(X)
        return self.label_encoder.inverse_transform(predictions).tolist()

    def generate_summaries(self, texts, min_length=30, max_length=60, batch_size=8):
        """
        Batch summarization for a list of texts.
        """
        if not self.summarizer:
            # Fallback: truncate or return original
            return [t[:max_length] + "..." if t and len(str(t)) > max_length else t for t in texts]

        results = []
        to_summarize = []
        idx_map = []
        for i, t in enumerate(texts):
            if not t or pd.isna(t) or len(str(t).split()) < min_length:
                results.append(t)
            else:
                to_summarize.append(str(t))
                idx_map.append(i)
                results.append(None)

        # Batch summarize
        if to_summarize:
            summaries = self.summarizer(
                to_summarize, max_length=max_length, min_length=min_length, 
                do_sample=False, batch_size=batch_size
            )
            for idx, summary in zip(idx_map, summaries):
                results[idx] = summary['summary_text']

        return results

    def analyze_dataframe(self, df: pd.DataFrame,
                         text_column: str = 'combined_text',
                         headline_column: str = 'headline',
                         body_column: str = 'body_text') -> pd.DataFrame:
        """
        Perform complete analysis on a DataFrame

        Args:
            df: Input DataFrame
            text_column: Column containing text for analysis
            headline_column: Column containing headlines
            body_column: Column containing body text

        Returns:
            DataFrame with analysis results
        """
        df = df.copy()

        # Create combined text if not exists
        if text_column not in df.columns:
            if headline_column in df.columns and body_column in df.columns:
                df[text_column] = (df[headline_column].fillna('') + ' ' +
                                   df[body_column].fillna(''))
            else:
                raise ValueError(f"Text column '{text_column}' not found and cannot create it")

        texts = df[text_column].fillna('').tolist()

        # Predict topics
        if self.lda_model:
            logger.info("Predicting topics...")
            df['topic_id'] = self.predict_topics(texts)

        # Predict political leaning
        if self.political_classifier:
            logger.info("Predicting political leaning...")
            df['predicted_leaning'] = self.predict_political_leaning(texts)

        # Generate summaries (fast batch mode)
        if body_column in df.columns:
            logger.info("Generating summaries (batch mode)...")
            df['perspective_summary'] = self.generate_summaries(df[body_column].tolist())

        return df

    def create_spectrum_view(self, df: pd.DataFrame,
                             topic_filter: Optional[int] = None) -> Dict[str, Any]:
        """
        Create spectrum view for given topic

        Args:
            df: Analyzed DataFrame
            topic_filter: Specific topic to filter (None for all)

        Returns:
            Dictionary containing spectrum data
        """
        if topic_filter is not None:
            df = df[df['topic_id'] == topic_filter]

        spectrum_data = {}
        leanings = ['Left', 'Center-Left', 'Center', 'Center-Right', 'Right']

        for leaning in leanings:
            articles = df[df['predicted_leaning'] == leaning]
            if not articles.empty:
                spectrum_data[leaning] = {
                    'count': len(articles),
                    'articles': articles[['headline', 'perspective_summary']].head(3).to_dict('records')
                }

        return spectrum_data

    def save_models(self) -> None:
        """Save all trained models"""
        logger.info(f"Saving models to {self.model_dir}")

        models = {
            'lda_model.pkl': self.lda_model,
            'topic_vectorizer.pkl': self.topic_vectorizer,
            'political_classifier.pkl': self.political_classifier,
            'political_vectorizer.pkl': self.political_vectorizer,
            'label_encoder.pkl': self.label_encoder
        }

        for filename, model in models.items():
            if model is not None:
                joblib.dump(model, self.model_dir / filename)

        logger.info("Models saved successfully")

    def load_models(self) -> 'NewsSpectrumAnalyzer':
        """Load all saved models"""
        logger.info(f"Loading models from {self.model_dir}")

        model_files = {
            'lda_model.pkl': 'lda_model',
            'topic_vectorizer.pkl': 'topic_vectorizer',
            'political_classifier.pkl': 'political_classifier',
            'political_vectorizer.pkl': 'political_vectorizer',
            'label_encoder.pkl': 'label_encoder'
        }

        for filename, attr_name in model_files.items():
            filepath = self.model_dir / filename
            if filepath.exists():
                setattr(self, attr_name, joblib.load(filepath))
                logger.info(f"Loaded {filename}")
            else:
                logger.warning(f"Model file {filename} not found")

        return self

    def visualize_spectrum_distribution(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """
        Visualize political spectrum distribution

        Args:
            df: Analyzed DataFrame
            save_path: Optional path to save the plot
        """
        if 'predicted_leaning' not in df.columns:
            logger.error("Political leaning predictions not found in DataFrame")
            return

        plt.figure(figsize=(12, 6))

        # Distribution by political leaning
        plt.subplot(1, 2, 1)
        leaning_counts = df['predicted_leaning'].value_counts()
        plt.pie(leaning_counts.values, labels=leaning_counts.index, autopct='%1.1f%%')
        plt.title('Political Leaning Distribution')

        # Distribution by topic and leaning
        plt.subplot(1, 2, 2)
        if 'topic_id' in df.columns:
            topic_leaning = pd.crosstab(df['topic_id'], df['predicted_leaning'])
            topic_leaning.plot(kind='bar', stacked=True, ax=plt.gca())
            plt.title('Topics vs Political Leaning')
            plt.xlabel('Topic ID')
            plt.ylabel('Count')
            plt.legend(title='Political Leaning', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()


# Utility functions for easy usage
def quick_analysis(df: pd.DataFrame,
                  text_column: str = 'combined_text',
                  political_labels_column: Optional[str] = None,
                  model_dir: str = "models") -> Tuple[pd.DataFrame, NewsSpectrumAnalyzer]:
    """
    Perform quick analysis on a DataFrame

    Args:
        df: Input DataFrame
        text_column: Column containing text for analysis
        political_labels_column: Column with political labels for training
        model_dir: Directory for model storage

    Returns:
        Tuple of (analyzed_df, analyzer_instance)
    """
    analyzer = NewsSpectrumAnalyzer(model_dir=model_dir)

    # Try to load existing models
    try:
        analyzer.load_models()
        logger.info("Loaded existing models")
    except:
        logger.info("Training new models")

        # Fit topic model
        texts = df[text_column].fillna('').tolist()
        analyzer.fit_topic_model(texts)

        # Fit political classifier if labels provided
        if political_labels_column and political_labels_column in df.columns:
            valid_data = df.dropna(subset=[text_column, political_labels_column])
            analyzer.fit_political_classifier(
                valid_data[text_column].tolist(),
                valid_data[political_labels_column].tolist()
            )

        # Save models
        analyzer.save_models()

    # Analyze DataFrame
    analyzed_df = analyzer.analyze_dataframe(df, text_column=text_column)

    return analyzed_df, analyzer


# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'headline': [
            "Climate Change Action Needed Now",
            "Economic Growth Shows Strong Numbers",
            "Healthcare Reform Debate Continues",
            "Immigration Policy Under Review",
            "Education Funding Increased"
        ],
        'body_text': [
            "Scientists warn that immediate action is needed to address climate change...",
            "The economy shows robust growth with unemployment at historic lows...",
            "Healthcare reform remains a contentious issue with various proposals...",
            "New immigration policies are being debated in Congress...",
            "Education funding has been increased by 15% in the new budget..."
        ],
        'political_leaning': ['Left', 'Right', 'Center', 'Right', 'Left']
    }

    df = pd.DataFrame(sample_data)
    df['combined_text'] = df['headline'] + ' ' + df['body_text']

    # Quick analysis
    analyzed_df, analyzer = quick_analysis(
        df,
        text_column='combined_text',
        political_labels_column='political_leaning'
    )

    print("Analysis Results:")
    print(analyzed_df[['headline', 'topic_id', 'predicted_leaning']].head())

    # Create spectrum view
    spectrum = analyzer.create_spectrum_view(analyzed_df)
    print("\nSpectrum View:")
    for leaning, data in spectrum.items():
        print(f"{leaning}: {data['count']} articles")