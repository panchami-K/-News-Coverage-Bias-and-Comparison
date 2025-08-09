import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any

class NewsDataCleaner:
    """
    A reusable class for cleaning and standardizing news data across multiple sources.
    """
    
    def __init__(self, required_columns: Optional[List[str]] = None):
        """
        Initialize the cleaner with required columns.
        
        Args:
            required_columns: List of columns that must be present in the data
        """
        self.required_columns = required_columns or [
            'article_id', 'headline', 'body_text', 'political_leaning', 'topic'
        ]
    
    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names by stripping whitespace, converting to lowercase,
        and replacing spaces with underscores.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with cleaned column names
        """
        df = df.copy()
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        return df
    
    def validate_required_columns(self, df: pd.DataFrame, 
                                raise_error: bool = False) -> List[str]:
        """
        Check for required columns and optionally raise error if missing.
        
        Args:
            df: Input DataFrame
            raise_error: Whether to raise exception for missing columns
            
        Returns:
            List of missing columns
        """
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        
        for col in missing_cols:
            print(f"Warning: Column '{col}' is missing from DataFrame!")
        
        if raise_error and missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        return missing_cols
    
    def remove_duplicates_and_nulls(self, df: pd.DataFrame, 
                                  duplicate_subset: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Remove duplicate rows and rows with null values in required columns.
        
        Args:
            df: Input DataFrame
            duplicate_subset: Columns to use for duplicate detection
            
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        
        # Default duplicate detection columns
        if duplicate_subset is None:
            duplicate_subset = ['article_id', 'headline']
        
        # Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates(subset=duplicate_subset)
        removed_duplicates = initial_rows - len(df)
        if removed_duplicates > 0:
            print(f"Removed {removed_duplicates} duplicate rows")
        
        # Remove rows with null values in required columns
        present_required_cols = [col for col in self.required_columns if col in df.columns]
        initial_rows = len(df)
        df = df.dropna(subset=present_required_cols)
        removed_nulls = initial_rows - len(df)
        if removed_nulls > 0:
            print(f"Removed {removed_nulls} rows with null values in required columns")
        
        return df
    
    def standardize_dates(self, df: pd.DataFrame, 
                         date_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Standardize date columns to datetime format.
        
        Args:
            df: Input DataFrame
            date_columns: List of date column names to standardize
            
        Returns:
            DataFrame with standardized dates
        """
        df = df.copy()
        
        if date_columns is None:
            date_columns = ['publication_date', 'date', 'pub_date']
        
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                print(f"Standardized date column: {col}")
        
        return df
    
    def standardize_categorical_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize categorical columns like political_leaning and source_name.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with standardized categorical values
        """
        df = df.copy()
        
        # Standardize political leaning
        if 'political_leaning' in df.columns:
            df['political_leaning'] = (df['political_leaning']
                                     .astype(str)
                                     .str.strip()
                                     .str.title()
                                     .replace(['Nan', 'NaN', 'None', '', 'nan'], 'Unknown'))
            print("Standardized political_leaning column")
        
        # Standardize source name
        if 'source_name' in df.columns:
            df['source_name'] = df['source_name'].astype(str).str.strip().str.title()
            print("Standardized source_name column")
        
        return df
    
    def fill_missing_summaries(self, df: pd.DataFrame, 
                              summary_col: str = 'summary',
                              fill_value: str = 'No summary available') -> pd.DataFrame:
        """
        Fill missing values in summary column.
        
        Args:
            df: Input DataFrame
            summary_col: Name of the summary column
            fill_value: Value to use for missing summaries
            
        Returns:
            DataFrame with filled summaries
        """
        df = df.copy()
        
        if summary_col in df.columns:
            missing_count = df[summary_col].isna().sum()
            df[summary_col] = df[summary_col].fillna(fill_value)
            if missing_count > 0:
                print(f"Filled {missing_count} missing values in {summary_col} column")
        
        return df
    
    def remove_blacklisted_sources(self, df: pd.DataFrame, 
                                 blacklist_df: pd.DataFrame,
                                 source_col: str = 'source_name',
                                 blacklist_col: int = 0) -> pd.DataFrame:
        """
        Remove sources that appear in blacklist.
        
        Args:
            df: Main DataFrame
            blacklist_df: DataFrame containing blacklisted sources
            source_col: Column name containing source names in main df
            blacklist_col: Column index/name in blacklist_df containing sources
            
        Returns:
            DataFrame with blacklisted sources removed
        """
        if blacklist_df.empty or source_col not in df.columns:
            return df
        
        df = df.copy()
        
        # Get blacklist set
        if isinstance(blacklist_col, int):
            blacklist_sources = blacklist_df.iloc[:, blacklist_col]
        else:
            blacklist_sources = blacklist_df[blacklist_col]
        
        blacklist_set = set(blacklist_sources.astype(str).str.strip().str.lower())
        
        # Filter out blacklisted sources
        initial_rows = len(df)
        df = df[~df[source_col].astype(str).str.strip().str.lower().isin(blacklist_set)]
        removed_sources = initial_rows - len(df)
        
        if removed_sources > 0:
            print(f"Removed {removed_sources} articles from blacklisted sources")
        
        return df
    
    def clean_all_dataframes(self, dataframes_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Apply column name cleaning to multiple DataFrames.
        
        Args:
            dataframes_dict: Dictionary of DataFrame name -> DataFrame
            
        Returns:
            Dictionary with cleaned DataFrames
        """
        cleaned_dfs = {}
        for name, df in dataframes_dict.items():
            cleaned_dfs[name] = self.clean_column_names(df)
            print(f"Cleaned column names for {name}")
        
        return cleaned_dfs
    
    def full_pipeline(self, df: pd.DataFrame, 
                     blacklist_df: Optional[pd.DataFrame] = None,
                     save_path: Optional[str] = None,
                     display_results: bool = True) -> pd.DataFrame:
        """
        Run the complete data cleaning pipeline.
        
        Args:
            df: Main DataFrame to clean
            blacklist_df: Optional blacklist DataFrame
            save_path: Optional path to save cleaned data
            display_results: Whether to display results
            
        Returns:
            Fully cleaned DataFrame
        """
        print("Starting data cleaning pipeline...")
        print(f"Initial data shape: {df.shape}")
        
        # Clean column names
        df = self.clean_column_names(df)
        
        # Validate required columns
        self.validate_required_columns(df)
        
        # Remove duplicates and nulls
        df = self.remove_duplicates_and_nulls(df)
        
        # Fill missing summaries
        df = self.fill_missing_summaries(df)
        
        # Standardize dates
        df = self.standardize_dates(df)
        
        # Standardize categorical values
        df = self.standardize_categorical_values(df)
        
        # Remove blacklisted sources
        if blacklist_df is not None:
            blacklist_df = self.clean_column_names(blacklist_df)
            df = self.remove_blacklisted_sources(df, blacklist_df)
        
        # Save cleaned data
        if save_path:
            df.to_csv(save_path, index=False)
            print(f"Saved cleaned data to {save_path}")
        
        print(f"Final cleaned data shape: {df.shape}")
        
        if display_results:
            print("\nFirst 5 rows of cleaned data:")
            print(df.head())
        
        return df


# Usage example and convenience functions
def quick_clean_news_data(df: pd.DataFrame, 
                         blacklist_df: Optional[pd.DataFrame] = None,
                         required_cols: Optional[List[str]] = None,
                         save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Quick function to clean news data with default settings.
    
    Args:
        df: DataFrame to clean
        blacklist_df: Optional blacklist DataFrame
        required_cols: Optional custom required columns
        save_path: Optional save path
        
    Returns:
        Cleaned DataFrame
    """
    cleaner = NewsDataCleaner(required_columns=required_cols)
    return cleaner.full_pipeline(df, blacklist_df, save_path)


def clean_multiple_dataframes(*dataframes, names=None):
    """
    Clean column names for multiple DataFrames at once.
    
    Args:
        *dataframes: Variable number of DataFrames
        names: Optional list of names for the DataFrames
        
    Returns:
        List of cleaned DataFrames
    """
    cleaner = NewsDataCleaner()
    
    if names is None:
        names = [f"df_{i}" for i in range(len(dataframes))]
    
    df_dict = dict(zip(names, dataframes))
    cleaned_dict = cleaner.clean_all_dataframes(df_dict)
    
    return list(cleaned_dict.values())



if __name__ == "__main__":
    # Example usage or test code
    print("This is a test run for NewsDataCleaner.")
    # Or call some function, etc.

  