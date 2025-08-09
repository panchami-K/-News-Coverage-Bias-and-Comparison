"""
Political Leaning Text Classification - Reusable Class
=====================================================

A comprehensive, reusable text classification pipeline for political leaning analysis.
Supports both TF-IDF and transformer-based embeddings with comprehensive evaluation.

Usage:
    from political_classifier import PoliticalTextClassifier
    
    # Initialize classifier
    classifier = PoliticalTextClassifier()
    
    # Load and train
    classifier.load_data(df, text_column='headline_clean', target_column='political_leaning')
    classifier.train(method='tfidf')
    
    # Evaluate and predict
    classifier.evaluate()
    predictions = classifier.predict(['Sample text to classify'])
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
from typing import List, Tuple, Optional, Union, Dict, Any
import os

warnings.filterwarnings('ignore')

# Optional transformer support
try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class PoliticalTextClassifier:
    """
    A comprehensive text classification pipeline for political leaning analysis.
    
    Features:
    - Support for both TF-IDF and transformer embeddings
    - Automatic data validation and cleaning
    - Comprehensive evaluation with visualizations
    - Model persistence (save/load functionality)
    - Robust error handling
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the classifier.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.df = None
        self.text_column = None
        self.target_column = None
        self.vectorizer = None
        self.classifier = None
        self.label_encoder = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.method = None
        self.is_trained = False
        
    def load_data(self, 
                  df: pd.DataFrame, 
                  text_column: str = 'headline_clean', 
                  target_column: str = 'political_leaning',
                  verbose: bool = True) -> 'PoliticalTextClassifier':
        """
        Load and validate the DataFrame for classification.
        
        Args:
            df: Input DataFrame
            text_column: Name of the text column
            target_column: Name of the target column
            verbose: Whether to print validation information
            
        Returns:
            Self for method chaining
        """
        if verbose:
            print("=== Data Validation ===")
            print(f"Original DataFrame shape: {df.shape}")

        # Validate columns exist
        if text_column not in df.columns:
            available_cols = [col for col in df.columns if 'headline' in col.lower() or 'text' in col.lower()]
            raise ValueError(f"Text column '{text_column}' not found. Available text columns: {available_cols}")

        if target_column not in df.columns:
            available_cols = [col for col in df.columns if 'lean' in col.lower() or 'politic' in col.lower()]
            raise ValueError(f"Target column '{target_column}' not found. Available target columns: {available_cols}")

        # Clean the data
        initial_size = len(df)
        df_clean = df.dropna(subset=[text_column, target_column]).copy()
        df_clean = df_clean[df_clean[text_column].str.strip() != ''].copy()

        # Clean target labels
        df_clean[target_column] = df_clean[target_column].astype(str).str.strip().str.lower()

        # Remove ambiguous labels
        exclude_labels = ['political_leaning', 'unknown', 'mixed', 'nan', 'none', '']
        df_clean = df_clean[~df_clean[target_column].isin(exclude_labels)].copy()

        if verbose:
            print(f"Removed {initial_size - len(df_clean)} rows with missing/invalid data")
            print(f"Final DataFrame shape: {df_clean.shape}")
            print(f"Target distribution:\n{df_clean[target_column].value_counts()}")

        self.df = df_clean
        self.text_column = text_column
        self.target_column = target_column
        
        return self
    
    def _create_tfidf_features(self, texts: pd.Series, fit: bool = True) -> np.ndarray:
        """Create TF-IDF features."""
        if fit or self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(
                max_features=3000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                stop_words='english'
            )
            X = self.vectorizer.fit_transform(texts)
        else:
            X = self.vectorizer.transform(texts)
        return X
    
    def _create_transformer_features(self, texts: pd.Series, fit: bool = True) -> np.ndarray:
        """Create transformer embeddings."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers not available. Install with: pip install sentence-transformers")
        
        if fit or self.vectorizer is None:
            self.vectorizer = SentenceTransformer('distiluse-base-multilingual-cased-v2')
        
        X = self.vectorizer.encode(texts.tolist(), show_progress_bar=True, batch_size=32)
        return X
    
    def train(self, 
              method: str = 'tfidf', 
              test_size: float = 0.2,
              classifier_params: Optional[Dict[str, Any]] = None,
              verbose: bool = True) -> 'PoliticalTextClassifier':
        """
        Train the classification model.
        
        Args:
            method: 'tfidf' or 'transformer'
            test_size: Proportion of data for testing
            classifier_params: Additional parameters for LogisticRegression
            verbose: Whether to print training information
            
        Returns:
            Self for method chaining
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if verbose:
            print(f"\n=== Model Training ({method.upper()}) ===")

        # Prepare text data
        texts = self.df[self.text_column].fillna('').astype(str)
        
        # Create features
        if method == 'tfidf':
            X = self._create_tfidf_features(texts, fit=True)
        elif method == 'transformer':
            X = self._create_transformer_features(texts, fit=True)
        else:
            raise ValueError("Method must be 'tfidf' or 'transformer'")
        
        if verbose:
            print(f"Feature matrix shape: {X.shape}")

        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(self.df[self.target_column])

        if verbose:
            print(f"Classes: {list(self.label_encoder.classes_)}")
            print(f"Class distribution: {np.bincount(y_encoded)}")

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y_encoded
        )

        if verbose:
            print(f"Train set: {self.X_train.shape}, Test set: {self.X_test.shape}")

        # Initialize classifier
        default_params = {
            'max_iter': 1000,
            'class_weight': 'balanced',
            'random_state': self.random_state,
            'solver': 'liblinear'
        }
        
        if classifier_params:
            default_params.update(classifier_params)
        
        self.classifier = LogisticRegression(**default_params)
        
        # Train
        self.classifier.fit(self.X_train, self.y_train)
        
        # Make predictions
        self.y_pred = self.classifier.predict(self.X_test)
        
        self.method = method
        self.is_trained = True
        
        if verbose:
            accuracy = accuracy_score(self.y_test, self.y_pred)
            print(f"Training completed. Test accuracy: {accuracy:.3f}")
        
        return self
    
    def evaluate(self, show_plots: bool = True, show_features: bool = True) -> Dict[str, Any]:
        """
        Evaluate the trained model.
        
        Args:
            show_plots: Whether to show confusion matrix plot
            show_features: Whether to show top features (TF-IDF only)
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        print("\n=== Model Evaluation ===")
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, self.y_pred)
        print(f"Accuracy: {accuracy:.3f}")
        
        # Classification report
        report = classification_report(
            self.y_test, self.y_pred, 
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        print("\nClassification Report:")
        print(classification_report(
            self.y_test, self.y_pred, 
            target_names=self.label_encoder.classes_
        ))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        if show_plots:
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=self.label_encoder.classes_, 
                        yticklabels=self.label_encoder.classes_)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            plt.tight_layout()
            plt.show()
        
        # Feature importance (TF-IDF only)
        if show_features and self.method == 'tfidf' and hasattr(self.vectorizer, 'get_feature_names_out'):
            print("\n=== Top Features per Class ===")
            feature_names = np.array(self.vectorizer.get_feature_names_out())
            
            for i, class_label in enumerate(self.label_encoder.classes_):
                if len(self.label_encoder.classes_) > 2:
                    top_indices = np.argsort(self.classifier.coef_[i])[-10:]
                else:
                    coef = self.classifier.coef_[0] if i == 1 else -self.classifier.coef_[0]
                    top_indices = np.argsort(coef)[-10:]
                
                print(f"\n{class_label.upper()}:")
                print(f"  {', '.join(feature_names[top_indices])}")
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }
    
    def predict(self, texts: Union[str, List[str]], return_probabilities: bool = False) -> Union[List[str], Tuple[List[str], np.ndarray]]:
        """
        Make predictions on new text data.
        
        Args:
            texts: Single text or list of texts to classify
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Predicted labels, optionally with probabilities
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
        
        texts_series = pd.Series(texts)
        
        # Transform texts using the fitted vectorizer
        if self.method == 'tfidf':
            X = self._create_tfidf_features(texts_series, fit=False)
        else:  # transformer
            X = self._create_transformer_features(texts_series, fit=False)
        
        # Make predictions
        predictions = self.classifier.predict(X)
        predicted_labels = self.label_encoder.inverse_transform(predictions).tolist()
        
        if return_probabilities:
            probabilities = self.classifier.predict_proba(X)
            return predicted_labels, probabilities
        
        return predicted_labels
    
    def predict_proba(self, texts: Union[str, List[str]]) -> Tuple[List[str], np.ndarray]:
        """Get prediction probabilities for all classes."""
        return self.predict(texts, return_probabilities=True)
    
    def save_model(self, filepath: str = 'political_classifier.pkl') -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        model_data = {
            'classifier': self.classifier,
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder,
            'method': self.method,
            'text_column': self.text_column,
            'target_column': self.target_column,
            'random_state': self.random_state
        }
        
        joblib.dump(model_data, filepath)
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> 'PoliticalTextClassifier':
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Self for method chaining
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.classifier = model_data['classifier']
        self.vectorizer = model_data['vectorizer']
        self.label_encoder = model_data['label_encoder']
        self.method = model_data['method']
        self.text_column = model_data.get('text_column', 'text')
        self.target_column = model_data.get('target_column', 'target')
        self.random_state = model_data.get('random_state', 42)
        self.is_trained = True
        
        print(f"✓ Model loaded from {filepath}")
        return self
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model."""
        if not self.is_trained:
            return {"status": "not_trained"}
        
        return {
            "status": "trained",
            "method": self.method,
            "classes": list(self.label_encoder.classes_),
            "n_classes": len(self.label_encoder.classes_),
            "text_column": self.text_column,
            "target_column": self.target_column,
            "train_size": self.X_train.shape[0] if self.X_train is not None else None,
            "test_size": self.X_test.shape[0] if self.X_test is not None else None,
            "accuracy": accuracy_score(self.y_test, self.y_pred) if self.y_test is not None else None
        }


# Example usage and utility functions
def quick_classify(df: pd.DataFrame, 
                  text_column: str = 'headline_clean', 
                  target_column: str = 'political_leaning',
                  method: str = 'tfidf') -> PoliticalTextClassifier:
    """
    Quick classification with default settings.
    
    Args:
        df: Input DataFrame
        text_column: Name of text column
        target_column: Name of target column
        method: Classification method ('tfidf' or 'transformer')
        
    Returns:
        Trained classifier
    """
    classifier = PoliticalTextClassifier()
    classifier.load_data(df, text_column, target_column)
    classifier.train(method=method)
    classifier.evaluate()
    return classifier


if __name__ == "__main__":
    # Example usage
    print("Political Text Classifier - Reusable Class")
    print("==========================================")
    print()
    print("Usage example:")
    print("""
    # Basic usage
    from political_classifier import PoliticalTextClassifier
    
    classifier = PoliticalTextClassifier()
    classifier.load_data(df, text_column='headline_clean', target_column='political_leaning')
    classifier.train(method='tfidf')
    classifier.evaluate()
    
    # Make predictions
    predictions = classifier.predict(['Sample headline text'])
    
    # Save model
    classifier.save_model('my_model.pkl')
    
    # Load model later
    new_classifier = PoliticalTextClassifier().load_model('my_model.pkl')
    """)
    
    print("\nTransformer support:", "✓ Available" if TRANSFORMERS_AVAILABLE else "✗ Not available")
    print("To install transformers: pip install sentence-transformers")