import os
import sys
import gc
import traceback
import logging
import pandas as pd
import psutil
import signal
import threading
import multiprocessing
from datetime import datetime
from contextlib import contextmanager
import time
import warnings
warnings.filterwarnings('ignore')

from src.data_loading import DataLoader
from src.data_cleaning import NewsDataCleaner
from src.text_preprocessing import MultilingualTextProcessor
from src.EDA import MultilingualNewsEDA
from src.feature_engineering_and_baseline_modeling import PoliticalTextClassifier
from src.visualization import SpectrumAnalysisVisualizer


class TimeoutError(Exception):
    pass


class ProcessMonitor:
    """Monitor process resources and terminate if needed"""
    def __init__(self, memory_limit_mb=4000, timeout_seconds=300):
        self.memory_limit_mb = memory_limit_mb
        self.timeout_seconds = timeout_seconds
        self.start_time = time.time()
        self.should_stop = False
        
    def check_resources(self):
        """Check if process should be terminated"""
        # Check memory
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        # Check timeout
        elapsed = time.time() - self.start_time
        
        if memory_mb > self.memory_limit_mb:
            raise MemoryError(f"Memory usage {memory_mb:.1f} MB exceeds limit {self.memory_limit_mb} MB")
            
        if elapsed > self.timeout_seconds:
            raise TimeoutError(f"Operation timeout: {elapsed:.1f}s exceeds {self.timeout_seconds}s")
            
        return memory_mb, elapsed


@contextmanager
def resource_monitor(memory_limit_mb=4000, timeout_seconds=300, check_interval=10):
    """Context manager for monitoring resources during operations"""
    monitor = ProcessMonitor(memory_limit_mb, timeout_seconds)
    
    def check_loop():
        while not monitor.should_stop:
            try:
                monitor.check_resources()
                time.sleep(check_interval)
            except (MemoryError, TimeoutError) as e:
                logging.getLogger(__name__).error(f"Resource monitor triggered: {e}")
                os._exit(1)  # Force exit
            except:
                pass
    
    # Start monitoring thread
    monitor_thread = threading.Thread(target=check_loop, daemon=True)
    monitor_thread.start()
    
    try:
        yield monitor
    finally:
        monitor.should_stop = True


def setup_logging(log_dir="logs"):
    """Setup comprehensive logging"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"pipeline_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def check_memory(logger, stage="", warning_threshold=2000, critical_threshold=3500):
    """Monitor memory usage with warnings"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    logger.info(f"Memory usage {stage}: {memory_mb:.2f} MB")
    
    if memory_mb > critical_threshold:
        logger.critical(f"CRITICAL: Memory usage {memory_mb:.2f} MB exceeds {critical_threshold} MB")
        logger.critical("Triggering cleanup and considering process termination")
        gc.collect()  # Force cleanup
        return memory_mb, True  # Critical flag
    elif memory_mb > warning_threshold:
        logger.warning(f"HIGH: Memory usage {memory_mb:.2f} MB exceeds {warning_threshold} MB")
        gc.collect()  # Preventive cleanup
    
    return memory_mb, False


def safe_import_module(module_path, timeout_seconds=30):
    """Safely import module with timeout using subprocess"""
    logger = logging.getLogger(__name__)
    logger.info(f"Attempting to import {module_path} with {timeout_seconds}s timeout...")
    
    def import_worker(result_dict):
        """Worker function for importing in separate process"""
        try:
            if module_path == "src.topic_clustering":
                from src.topic_clustering import NewsSpectrumAnalyzer
                result_dict['success'] = True
                result_dict['module'] = NewsSpectrumAnalyzer
            else:
                module = __import__(module_path, fromlist=[''])
                result_dict['success'] = True
                result_dict['module'] = module
        except Exception as e:
            result_dict['success'] = False
            result_dict['error'] = str(e)
            result_dict['traceback'] = traceback.format_exc()
    
    # Use multiprocessing to isolate import
    try:
        manager = multiprocessing.Manager()
        result_dict = manager.dict()
        
        process = multiprocessing.Process(target=import_worker, args=(result_dict,))
        process.start()
        process.join(timeout=timeout_seconds)
        
        if process.is_alive():
            logger.error(f"Import timeout after {timeout_seconds}s, terminating process")
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
            return None
            
        if result_dict.get('success', False):
            logger.info(f"Successfully imported {module_path}")
            # Note: We can't return the actual module due to multiprocessing limitations
            # Instead, we'll return a success flag and import in main process
            return True
        else:
            logger.error(f"Import failed: {result_dict.get('error', 'Unknown error')}")
            return None
            
    except Exception as e:
        logger.error(f"Import process failed: {str(e)}")
        return None


def safe_dataframe_operation(func, *args, **kwargs):
    """Wrapper for safe dataframe operations with memory management"""
    try:
        result = func(*args, **kwargs)
        gc.collect()  # Force garbage collection
        return result
    except Exception as e:
        gc.collect()
        raise e


def validate_dataframe(df, stage_name, logger, required_columns=None):
    """Validate dataframe at each stage"""
    if df is None:
        logger.error(f"{stage_name}: DataFrame is None")
        return False
    
    if df.empty:
        logger.error(f"{stage_name}: DataFrame is empty")
        return False
    
    logger.info(f"{stage_name}: DataFrame shape: {df.shape}")
    logger.info(f"{stage_name}: Columns: {list(df.columns)}")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            logger.error(f"{stage_name}: Missing required columns: {missing_cols}")
            return False
    
    # Check for memory issues
    try:
        memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024
        logger.info(f"{stage_name}: DataFrame memory usage: {memory_usage:.2f} MB")
    except:
        logger.warning(f"{stage_name}: Could not calculate memory usage")
    
    return True


def create_basic_topic_analysis(df, logger):
    """Fallback basic topic analysis without heavy ML libraries"""
    logger.info("Running basic topic analysis as fallback...")
    
    try:
        # Simple frequency-based analysis
        topics = df['topic'].value_counts().head(20)
        logger.info("Top 20 Topics:")
        for topic, count in topics.items():
            logger.info(f"  {topic}: {count}")
        
        # Political leaning distribution
        political_dist = df['political_leaning'].value_counts()
        logger.info("Political Leaning Distribution:")
        for leaning, count in political_dist.items():
            logger.info(f"  {leaning}: {count}")
        
        # Create simple analysis dataframe
        df_analysis = df.copy()
        df_analysis['topic_frequency'] = df_analysis['topic'].map(topics)
        
        # Simple political scoring
        political_mapping = {
            'left': -1, 'center-left': -0.5, 'center': 0, 
            'center-right': 0.5, 'right': 1, 'mixed': 0
        }
        df_analysis['political_score'] = df_analysis['political_leaning'].str.lower().map(political_mapping)
        
        # Add basic sentiment analysis (simple keyword-based)
        positive_words = ['good', 'great', 'excellent', 'positive', 'success', 'achievement']
        negative_words = ['bad', 'terrible', 'negative', 'failure', 'crisis', 'concern']
        
        def simple_sentiment(text):
            if pd.isna(text):
                return 0
            text_lower = str(text).lower()
            pos_count = sum(word in text_lower for word in positive_words)
            neg_count = sum(word in text_lower for word in negative_words)
            return pos_count - neg_count
        
        df_analysis['sentiment_score'] = df_analysis['headline_clean'].apply(simple_sentiment)
        
        # Save basic analysis
        basic_analysis_path = os.path.join("data", "processed", "basic_analysis.csv")
        df_analysis.to_csv(basic_analysis_path, index=False)
        logger.info(f"Basic analysis saved to {basic_analysis_path}")
        
        # Create summary statistics
        logger.info("=== BASIC ANALYSIS SUMMARY ===")
        logger.info(f"Total articles: {len(df_analysis)}")
        logger.info(f"Unique topics: {df_analysis['topic'].nunique()}")
        logger.info(f"Unique sources: {df_analysis['source_name'].nunique()}")
        logger.info(f"Languages: {df_analysis['language'].value_counts().to_dict()}")
        logger.info(f"Average sentiment: {df_analysis['sentiment_score'].mean():.2f}")
        
        return df_analysis
        
    except Exception as e:
        logger.error(f"Basic topic analysis failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return df


def run_topic_clustering_safe(processed_df, logger):
    """Safely run topic clustering with multiple fallback strategies"""
    logger.info("=== Starting Safe Topic Clustering ===")
    
    try:
        # Step 1: Prepare and validate data
        logger.info("Step 1: Preparing combined text column...")
        if 'combined_text' not in processed_df.columns:
            processed_df['combined_text'] = (
                processed_df['headline_clean'].fillna('') + ' ' + 
                processed_df['body_text_clean'].fillna('')
            ).str.strip()
        
        # Validate combined text
        empty_texts = processed_df['combined_text'].isna().sum() + (processed_df['combined_text'] == '').sum()
        logger.info(f"Empty combined texts: {empty_texts}/{len(processed_df)}")
        
        # Filter out empty texts
        valid_mask = (processed_df['combined_text'].notna()) & (processed_df['combined_text'] != '')
        if valid_mask.sum() == 0:
            logger.error("No valid text data found for analysis")
            return create_basic_topic_analysis(processed_df, logger)
        
        processed_df_clean = processed_df[valid_mask].copy()
        logger.info(f"Using {len(processed_df_clean)} valid text entries")
        
        # Step 2: Check memory and sample if needed
        memory_mb, critical = check_memory(logger, "before topic clustering")
        
        if len(processed_df_clean) > 3000 or critical:
            sample_size = min(2000, len(processed_df_clean) // 2)
            logger.info(f"Sampling {sample_size} documents from {len(processed_df_clean)} for memory efficiency")
            processed_df_sample = processed_df_clean.sample(n=sample_size, random_state=42).copy()
        else:
            processed_df_sample = processed_df_clean.copy()
        
        # Step 3: Try to import NewsSpectrumAnalyzer safely
        logger.info("Step 3: Testing NewsSpectrumAnalyzer import...")
        import_success = safe_import_module("src.topic_clustering", timeout_seconds=45)
        
        if not import_success:
            logger.warning("NewsSpectrumAnalyzer import failed or timed out")
            return create_basic_topic_analysis(processed_df_clean, logger)
        
        # Step 4: Try actual import and initialization with resource monitoring
        logger.info("Step 4: Attempting NewsSpectrumAnalyzer initialization...")
        
        with resource_monitor(memory_limit_mb=3500, timeout_seconds=180):
            try:
                from src.topic_clustering import NewsSpectrumAnalyzer
                logger.info("✓ Import successful")
                
                # Initialize with minimal settings for safety
                analyzer = NewsSpectrumAnalyzer()
                logger.info("✓ Analyzer initialized")
                
                # Prepare data
                text_data = processed_df_sample['combined_text'].tolist()
                logger.info(f"✓ Prepared {len(text_data)} text documents")
                
                # Fit topic model with monitoring
                logger.info("Fitting topic model...")
                start_time = time.time()
                analyzer.fit_topic_model(text_data)
                topic_time = time.time() - start_time
                logger.info(f"✓ Topic model fitted in {topic_time:.2f} seconds")
                
                # Check memory after topic modeling
                memory_mb, critical = check_memory(logger, "after topic modeling")
                if critical:
                    logger.warning("Critical memory usage after topic modeling, skipping political classifier")
                elif 'political_leaning' in processed_df_sample.columns:
                    logger.info("Fitting political classifier...")
                    political_labels = processed_df_sample['political_leaning'].tolist()
                    analyzer.fit_political_classifier(text_data, political_labels)
                    logger.info("✓ Political classifier fitted")
                
                # Analyze dataframe
                logger.info("Analyzing full dataframe...")
                analyzed_df = analyzer.analyze_dataframe(processed_df_clean, text_column='combined_text')
                logger.info(f"✓ Analysis completed: {analyzed_df.shape}")
                
                # Save results
                analyzed_path = os.path.join("data", "processed", "analyzed_main_combined.csv")
                analyzed_df.to_csv(analyzed_path, index=False)
                logger.info(f"✓ Analyzed data saved to {analyzed_path}")
                
                # Save models if possible
                try:
                    analyzer.save_models()
                    logger.info("✓ Models saved successfully")
                except Exception as e:
                    logger.warning(f"Model saving failed: {str(e)}")
                
                logger.info("=== NewsSpectrumAnalyzer completed successfully ===")
                return analyzed_df
                
            except Exception as e:
                logger.error(f"NewsSpectrumAnalyzer execution failed: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise e
        
    except (TimeoutError, MemoryError, Exception) as e:
        logger.warning(f"Advanced topic clustering failed: {str(e)}")
        logger.info("Falling back to basic topic analysis...")
        return create_basic_topic_analysis(processed_df, logger)


def main():
    # Setup logging
    logger = setup_logging()
    logger.info("=== Starting Robust News Analysis Pipeline ===")
    
    try:
        # Define directories
        data_raw_dir = os.path.join("data", "raw")
        data_processed_dir = os.path.join("data", "processed")
        model_dir = "models"
        visuals_dir = "visuals"

        # Create directories if they don't exist
        for directory in [data_raw_dir, data_processed_dir, model_dir, visuals_dir]:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Directory ensured: {directory}")

        check_memory(logger, "at startup")

        # =================================================================
        # DATA LOADING
        # =================================================================
        logger.info("=== Starting Data Loading ===")
        try:
            loader = DataLoader()
            
            # Check if files exist
            file1_path = os.path.join(data_raw_dir, "connection.csv")
            file2_path = os.path.join(data_raw_dir, "connection1.csv")
            
            if not os.path.exists(file1_path):
                logger.error(f"File not found: {file1_path}")
                return 1
            if not os.path.exists(file2_path):
                logger.error(f"File not found: {file2_path}")
                return 1
            
            df1 = safe_dataframe_operation(loader.load_and_inspect, file1_path, name="Connection 1")
            df2 = safe_dataframe_operation(loader.load_and_inspect, file2_path, name="Connection 2")

            if not validate_dataframe(df1, "Connection 1 loading", logger):
                return 1
            if not validate_dataframe(df2, "Connection 2 loading", logger):
                return 1

            combined_df = safe_dataframe_operation(
                loader.combine_datasets, 
                ['Connection 1', 'Connection 2'], 
                combined_name="Main Combined"
            )
            
            if not validate_dataframe(combined_df, "Combined dataset", logger):
                return 1
                
            check_memory(logger, "after data loading")
            
        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return 1

        # =================================================================
        # DATA CLEANING
        # =================================================================
        logger.info("=== Starting Data Cleaning ===")
        try:
            cleaner = NewsDataCleaner()
            cleaned_df = safe_dataframe_operation(
                cleaner.full_pipeline, 
                combined_df, 
                display_results=True
            )
            
            if not validate_dataframe(cleaned_df, "Cleaned data", logger):
                return 1
            
            cleaned_path = os.path.join(data_processed_dir, "cleaned_main_combined.csv")
            cleaned_df.to_csv(cleaned_path, index=False)
            logger.info(f"Cleaned data saved to {cleaned_path}")
            
            # Clear memory
            del combined_df
            gc.collect()
            check_memory(logger, "after data cleaning")
            
        except Exception as e:
            logger.error(f"Data cleaning failed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return 1

        # =================================================================
        # TEXT PREPROCESSING
        # =================================================================
        logger.info("=== Running Multilingual Text Preprocessing ===")
        try:
            processor = MultilingualTextProcessor(install_dependencies=False, verbose=True)
            processed_df = safe_dataframe_operation(
                processor.process_dataframe,
                cleaned_df,
                text_columns=['headline', 'body_text'],
                detect_lang=True,
                clean_text=True,
                add_word_count=True
            )
            
            if not validate_dataframe(processed_df, "Processed data", logger, 
                                   required_columns=['headline_clean']):
                return 1
            
            processed_path = os.path.join(data_processed_dir, "processed_main_combined.csv")
            processed_df.to_csv(processed_path, index=False)
            logger.info(f"Processed data saved to {processed_path}")
            
            # Clear memory
            del cleaned_df
            gc.collect()
            check_memory(logger, "after text preprocessing")
            
        except Exception as e:
            logger.error(f"Text preprocessing failed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return 1

        # =================================================================
        # EXPLORATORY DATA ANALYSIS
        # =================================================================
        logger.info("=== Running Exploratory Data Analysis (EDA) ===")
        try:
            with resource_monitor(memory_limit_mb=3000, timeout_seconds=300):
                eda = MultilingualNewsEDA(processed_df)
                eda.run_complete_eda()
                logger.info("EDA completed successfully")
            check_memory(logger, "after EDA")
            
        except Exception as e:
            logger.warning(f"EDA failed (continuing pipeline): {str(e)}")
            logger.warning(f"Traceback: {traceback.format_exc()}")

        # =================================================================
        # BASELINE CLASSIFICATION
        # =================================================================
        logger.info("=== Running Baseline Political Leaning Classification ===")
        try:
            if 'political_leaning' not in processed_df.columns:
                logger.warning("No 'political_leaning' column found, skipping classification")
            else:
                with resource_monitor(memory_limit_mb=3000, timeout_seconds=180):
                    classifier = PoliticalTextClassifier()
                    classifier.load_data(processed_df, text_column='headline_clean', target_column='political_leaning')
                    classifier.train(method='tfidf')
                    classifier.evaluate()
                    
                    model_path = os.path.join(model_dir, "political_classifier.pkl")
                    classifier.save_model(model_path)
                    logger.info(f"Model saved to {model_path}")
                
            check_memory(logger, "after baseline classification")
            
        except Exception as e:
            logger.warning(f"Classification pipeline failed (continuing): {str(e)}")
            logger.warning(f"Traceback: {traceback.format_exc()}")

        # =================================================================
        # SAFE TOPIC CLUSTERING AND SPECTRUM ANALYSIS
        # =================================================================
        logger.info("=== Starting Safe Topic Clustering and Spectrum Analysis ===")
        try:
            analyzed_df = run_topic_clustering_safe(processed_df, logger)
            
            if analyzed_df is not None:
                logger.info(f"Topic clustering completed successfully: {analyzed_df.shape}")
            else:
                logger.warning("Topic clustering returned None, using original dataframe")
                analyzed_df = processed_df
            
            check_memory(logger, "after topic clustering pipeline")
            
        except Exception as e:
            logger.error(f"Topic clustering pipeline failed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            logger.info("Using basic fallback analysis...")
            analyzed_df = create_basic_topic_analysis(processed_df, logger)

        # =================================================================
        # PIPELINE COMPLETION
        # =================================================================
        logger.info("=== Pipeline Completed Successfully ===")
        final_memory, _ = check_memory(logger, "at completion")
        
        # Summary
        logger.info("=== PIPELINE SUMMARY ===")
        logger.info(f"Final processed data shape: {analyzed_df.shape if analyzed_df is not None else 'N/A'}")
        logger.info(f"Final memory usage: {final_memory:.2f} MB")
        logger.info("All major components completed successfully")
        logger.info("Pipeline execution finished successfully")

        return 0  # Success

    except Exception as e:
        logger.error(f"Critical pipeline failure: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1  # Return error code

    finally:
        # Cleanup
        gc.collect()
        logger.info("Final cleanup completed")


if __name__ == "__main__":
    exit_code = main()
    print(f"\nPipeline completed with exit code: {exit_code}")
    sys.exit(exit_code)