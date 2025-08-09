import pandas as pd
import os
from typing import Dict, List, Optional, Union

class DataLoader:
    """
    A reusable utility class for loading and inspecting CSV datasets.
    """
    
    def __init__(self, working_directory: Optional[str] = None):
        """
        Initialize the DataLoader.
        
        Args:
            working_directory: Optional path to set as working directory
        """
        if working_directory:
            os.chdir(working_directory)
        self.datasets = {}
        
    def show_directory_info(self) -> None:
        """Display current working directory and available files."""
        print("Current working directory:", os.getcwd())
        files = [f for f in os.listdir() if f.endswith('.csv')]
        print(f"CSV files in directory: {files}")
        
    def load_and_inspect(self, file: str, name: Optional[str] = None, 
                        display_head: bool = True, **kwargs) -> Optional[pd.DataFrame]:
        """
        Load a CSV file and display inspection information.
        
        Args:
            file: Path to the CSV file
            name: Display name for the dataset (defaults to filename)
            display_head: Whether to display the first few rows
            **kwargs: Additional arguments to pass to pd.read_csv()
            
        Returns:
            DataFrame if successful, None if failed
        """
        if name is None:
            name = file
            
        try:
            # Default arguments for pd.read_csv with option to override
            default_args = {'low_memory': False}
            default_args.update(kwargs)
            
            df = pd.read_csv(file, **default_args)
            
            print(f"\n{'='*50}")
            print(f"Loaded '{name}': {df.shape[0]} rows, {df.shape[1]} columns")
            print(f"{'='*50}")
            print(f"Columns: {df.columns.tolist()}")
            
            # Missing values summary
            missing_values = df.isnull().sum()
            if missing_values.sum() > 0:
                print(f"\nMissing values:")
                for col, count in missing_values[missing_values > 0].items():
                    print(f"  {col}: {count} ({count/len(df)*100:.1f}%)")
            else:
                print("\nNo missing values found.")
            
            # Data types
            print(f"\nData types:")
            for col, dtype in df.dtypes.items():
                print(f"  {col}: {dtype}")
            
            if display_head:
                print(f"\nFirst 5 rows:")
                print(df.head().to_string())
            
            # Store in datasets dictionary
            self.datasets[name] = df
            return df
            
        except Exception as e:
            print(f"\nError loading {name}: {e}")
            return None
    
    def load_multiple(self, file_configs: List[Dict[str, Union[str, bool]]], 
                     display_summary: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Load multiple CSV files at once.
        
        Args:
            file_configs: List of dictionaries with file configuration
                Each dict should have 'file' key and optionally 'name', 'display_head'
            display_summary: Whether to display a summary at the end
            
        Returns:
            Dictionary of successfully loaded DataFrames
        """
        loaded_datasets = {}
        
        for config in file_configs:
            file_path = config['file']
            name = config.get('name', file_path)
            display_head = config.get('display_head', True)
            
            df = self.load_and_inspect(file_path, name, display_head)
            if df is not None:
                loaded_datasets[name] = df
        
        if display_summary:
            self.display_summary()
            
        return loaded_datasets
    
    def combine_datasets(self, dataset_names: List[str], 
                        combined_name: str = "Combined") -> Optional[pd.DataFrame]:
        """
        Combine multiple datasets vertically (concatenate).
        
        Args:
            dataset_names: List of dataset names to combine
            combined_name: Name for the combined dataset
            
        Returns:
            Combined DataFrame if successful, None if failed
        """
        try:
            datasets_to_combine = []
            for name in dataset_names:
                if name in self.datasets:
                    datasets_to_combine.append(self.datasets[name])
                else:
                    print(f"Warning: Dataset '{name}' not found in loaded datasets.")
            
            if not datasets_to_combine:
                print("No valid datasets found to combine.")
                return None
            
            combined_df = pd.concat(datasets_to_combine, ignore_index=True)
            print(f"\nCombined {len(datasets_to_combine)} datasets into '{combined_name}': "
                  f"{combined_df.shape[0]} rows, {combined_df.shape[1]} columns")
            
            self.datasets[combined_name] = combined_df
            return combined_df
            
        except Exception as e:
            print(f"Error combining datasets: {e}")
            return None
    
    def display_summary(self) -> None:
        """Display a summary of all loaded datasets."""
        print(f"\n{'='*60}")
        print("SUMMARY OF LOADED DATASETS")
        print(f"{'='*60}")
        
        if not self.datasets:
            print("No datasets loaded.")
            return
        
        for name, df in self.datasets.items():
            if df is not None:
                print(f"{name:<20}: {df.shape[0]:>6} rows, {df.shape[1]:>3} columns")
            else:
                print(f"{name:<20}: Not loaded")
    
    def get_dataset(self, name: str) -> Optional[pd.DataFrame]:
        """Get a specific dataset by name."""
        return self.datasets.get(name)
    
    def list_datasets(self) -> List[str]:
        """Get list of all loaded dataset names."""
        return list(self.datasets.keys())


# Example usage function
def load_news_datasets():
    """
    Example function showing how to use the DataLoader for news datasets.
    Modify the file configurations as needed for your specific files.
    """
    # Initialize the data loader
    loader = DataLoader()
    
    # Show directory information
    loader.show_directory_info()
    
    # Define file configurations
    file_configs = [
        {'file': 'connection.csv', 'name': 'Connection 1'},
        {'file': 'connection1.csv', 'name': 'Connection 2'},
        {'file': 'Newsdata_Records.csv', 'name': 'Newsdata Records'},
        {'file': 'allsides_data.csv', 'name': 'AllSides Data'},
        {'file': 'blacklist.csv', 'name': 'Blacklist'},
        {'file': 'gnews_top_headlines.csv', 'name': 'Google News Headlines'},
        {'file': 'newsdata_latest.csv', 'name': 'Newsdata Latest'}
    ]
    
    # Load all datasets
    loaded_data = loader.load_multiple(file_configs)
    
    # Combine connection datasets if both loaded successfully
    if 'Connection 1' in loaded_data and 'Connection 2' in loaded_data:
        combined_main = loader.combine_datasets(['Connection 1', 'Connection 2'], 'Main Combined')
        print(f"\nSuccessfully combined main datasets.")
    
    return loader


# Quick usage examples:
if __name__ == "__main__":
    # Method 1: Use the example function
    loader = load_news_datasets()
    
    # Method 2: Manual usage
    # loader = DataLoader()
    # df1 = loader.load_and_inspect('connection.csv', 'Connection Data')
    # df2 = loader.load_and_inspect('newsdata.csv', 'News Data')
    # combined = loader.combine_datasets(['Connection Data', 'News Data'], 'Combined Data')
    
    # Method 3: Load single file with custom parameters
    # loader = DataLoader()
    # df = loader.load_and_inspect('large_file.csv', 'Large Dataset', 
    #                             display_head=False, nrows=1000)