#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script to run the entire Amazon Review Sentiment Analysis pipeline.
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Set path constants
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)  # Add project directory to Python path


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Amazon Review Sentiment Analysis Pipeline')
    
    parser.add_argument('--acquire', action='store_true',
                      help='Download and prepare the dataset')
    
    parser.add_argument('--preprocess', action='store_true',
                      help='Preprocess the dataset')
    
    parser.add_argument('--train', action='store_true',
                      help='Train the sentiment analysis models')
    
    parser.add_argument('--visualize', action='store_true',
                      help='Generate visualizations')
    
    parser.add_argument('--predict', action='store_true',
                      help='Run the prediction interface')
    
    parser.add_argument('--full', action='store_true',
                      help='Run the entire pipeline')
    
    parser.add_argument('--subset-size', type=int, default=1000,
                       help='Number of reviews to use (default: 1000)')
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not any(vars(args).values()):
        parser.print_help()
        sys.exit(1)
    
    return args


def run_data_acquisition(subset_size):
    """Run the data acquisition process."""
    print("\n" + "="*50)
    print("STEP 1: DATA ACQUISITION")
    print("="*50)
    
    # import here to avoid circular imports
    from src.data_acquisition import ensure_directories, download_dataset, extract_dataset, create_subset, SUBSET_SIZE
    
    # Override subset size if specified
    if subset_size != SUBSET_SIZE:
        print(f"Using custom subset size: {subset_size}")
        # This is a bit of a hack, but it works
        import src.data_acquisition
        src.data_acquisition.SUBSET_SIZE = subset_size
    
    # Run the data acquisition process
    ensure_directories()
    download_dataset()
    if extract_dataset():
        create_subset()
    
    print("Data acquisition completed.")


def run_preprocessing():
    """Run the text preprocessing."""
    print("\n" + "="*50)
    print("STEP 2: TEXT PREPROCESSING")
    print("="*50)
    
    # Import preprocessing module
    from src.text_preprocessing import TextPreprocessor
    from src.data_acquisition import PROCESSED_DATA_DIR
    
    # Load the datasets
    train_path = os.path.join(PROCESSED_DATA_DIR, "amazon_reviews_train.csv")
    test_path = os.path.join(PROCESSED_DATA_DIR, "amazon_reviews_test.csv")
    
    # Check if datasets exist
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Datasets not found. Please run data acquisition first.")
        return False
    
    # Process training data
    print("Processing training data...")
    import pandas as pd
    train_df = pd.read_csv(train_path)
    preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=True)
    
    # Preprocess the text
    preprocessed_train_df = preprocessor.preprocess_dataframe(train_df, text_column='review')
    
    # Add text features
    final_train_df = preprocessor.add_text_features(preprocessed_train_df)
    
    # Save the preprocessed data
    preprocessed_train_path = os.path.join(PROCESSED_DATA_DIR, "amazon_reviews_train_preprocessed.csv")
    final_train_df.to_csv(preprocessed_train_path, index=False)
    print(f"Preprocessed training data saved to {preprocessed_train_path}")
    
    # Process test data
    print("\nProcessing test data...")
    test_df = pd.read_csv(test_path)
    
    # Preprocess the text
    preprocessed_test_df = preprocessor.preprocess_dataframe(test_df, text_column='review')
    
    # Add text features
    final_test_df = preprocessor.add_text_features(preprocessed_test_df)
    
    # Save the preprocessed data
    preprocessed_test_path = os.path.join(PROCESSED_DATA_DIR, "amazon_reviews_test_preprocessed.csv")
    final_test_df.to_csv(preprocessed_test_path, index=False)
    print(f"Preprocessed test data saved to {preprocessed_test_path}")
    
    print("Text preprocessing completed.")
    return True


def run_model_training():
    """Run the model training."""
    print("\n" + "="*50)
    print("STEP 3: MODEL TRAINING")
    print("="*50)
    
    # Import training module
    from src.model_training import main as train_main
    
    # Run training
    train_main()
    
    print("Model training completed.")


def run_visualization():
    """Run the visualization scripts."""
    print("\n" + "="*50)
    print("STEP 4: VISUALIZATION")
    print("="*50)
    
    # Import visualization module
    from src.visualization import main as visualize_main
    
    # Run visualization
    visualize_main()
    
    print("Visualization completed.")


def run_prediction_interface():
    """Run the prediction interface."""
    print("\n" + "="*50)
    print("STEP 5: PREDICTION INTERFACE")
    print("="*50)
    
    # Import prediction module
    from src.predict import main as predict_main
    
    # Run prediction interface
    predict_main()


def run_full_pipeline(subset_size):
    """Run the entire pipeline."""
    print("\n" + "="*50)
    print("RUNNING FULL AMAZON REVIEW SENTIMENT ANALYSIS PIPELINE")
    print("="*50)
    
    # Measure execution time
    start_time = time.time()
    
    # Run all steps
    run_data_acquisition(subset_size)
    run_preprocessing()
    run_model_training()
    run_visualization()
    
    # Print execution time
    execution_time = time.time() - start_time
    print(f"\nFull pipeline completed in {execution_time:.2f} seconds.")
    
    # Ask if user wants to run the prediction interface
    print("\nWould you like to run the prediction interface? (y/n)")
    choice = input("> ").strip().lower()
    if choice == 'y':
        run_prediction_interface()


def main():
    """Main function to run the pipeline based on command line arguments."""
    args = parse_arguments()
    
    if args.full:
        run_full_pipeline(args.subset_size)
        return
    
    if args.acquire:
        run_data_acquisition(args.subset_size)
    
    if args.preprocess:
        run_preprocessing()
    
    if args.train:
        run_model_training()
    
    if args.visualize:
        run_visualization()
    
    if args.predict:
        run_prediction_interface()


if __name__ == "__main__":
    main()
