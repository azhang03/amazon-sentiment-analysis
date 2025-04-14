#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script for downloading and preparing a subset of the Amazon Reviews Polarity Dataset.
"""

import os
import pandas as pd
import numpy as np
import urllib.request
import tarfile
import random
from tqdm import tqdm
import sys

# Constants
DATASET_URL = "https://s3.amazonaws.com/fast-ai-nlp/amazon_review_polarity_csv.tgz"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
DATASET_PATH = os.path.join(RAW_DATA_DIR, "amazon_review_polarity_csv.tgz")
SUBSET_SIZE = 1000  # Number of reviews to sample (500 for each class)


def ensure_directories():
    """Create necessary directories if they don't exist."""
    for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR]:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory {directory} is ready.")


def download_dataset():
    """Download the Amazon Reviews Polarity dataset if not already present."""
    if os.path.exists(DATASET_PATH):
        print(f"Dataset already exists at {DATASET_PATH}")
        return
    
    print(f"Downloading dataset from {DATASET_URL}...")
    # using tqdm to show a progress bar during download
    with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1) as t:
        def reporthook(blocknum, blocksize, totalsize):
            t.total = totalsize
            t.update(blocknum * blocksize - t.n)
        
        urllib.request.urlretrieve(DATASET_URL, DATASET_PATH, reporthook=reporthook)
    
    print(f"Dataset downloaded to {DATASET_PATH}")


def extract_dataset():
    """Extract the downloaded dataset."""
    if not os.path.exists(DATASET_PATH):
        print("Dataset file not found. Please download it first.")
        return False
    
    extracted_dir = os.path.join(RAW_DATA_DIR, "amazon_review_polarity_csv")
    if os.path.exists(extracted_dir):
        print(f"Dataset already extracted to {extracted_dir}")
        return True
    
    print(f"Extracting dataset to {RAW_DATA_DIR}...")
    try:
        with tarfile.open(DATASET_PATH, "r:gz") as tar:
            # Extract with progress tracking
            members = tar.getmembers()
            for member in tqdm(members, desc="Extracting files"):
                tar.extract(member, RAW_DATA_DIR)
        print(f"Dataset extracted to {RAW_DATA_DIR}")
        return True
    except Exception as e:
        print(f"Error extracting dataset: {e}")
        return False


def create_subset():
    """Create a balanced subset of the dataset with SUBSET_SIZE reviews."""
    # Path to the train.csv file in the extracted dataset
    train_csv = os.path.join(RAW_DATA_DIR, "amazon_review_polarity_csv", "train.csv")
    
    if not os.path.exists(train_csv):
        print(f"Train dataset not found at {train_csv}")
        return
    
    print(f"Creating a subset of {SUBSET_SIZE} reviews...")
    
    # this is actually a headerless CSV with the following fields:
    # 1. class index (1 or 2, where 1 = negative, 2 = positive)
    # 2. title
    # 3. review text
    
    # read in chunks to handle large file efficiently
    chunks = []
    chunk_size = 100000  # adjust based on your memory constraints
    
    # Sample records from each class
    samples_per_class = SUBSET_SIZE // 2
    positive_samples = []
    negative_samples = []
    
    # using pd.read_csv with no header and specified column names
    column_names = ['sentiment', 'title', 'review']
    
    print("Reading and sampling from dataset...")
    for chunk in tqdm(pd.read_csv(train_csv, header=None, names=column_names, chunksize=chunk_size)):
        # in this dataset, 1 = negative, 2 = positive
        # i'd rather use 0 = negative, 1 = positive for ML convention
        chunk['sentiment'] = chunk['sentiment'].map({1: 0, 2: 1})
        
        # Sample from positive class (sentiment = 1)
        pos_chunk = chunk[chunk['sentiment'] == 1]
        if len(positive_samples) < samples_per_class and not pos_chunk.empty:
            samples_to_take = min(samples_per_class - len(positive_samples), len(pos_chunk))
            positive_samples.append(pos_chunk.sample(samples_to_take))
        
        # Sample from negative class (sentiment = 0)
        neg_chunk = chunk[chunk['sentiment'] == 0]
        if len(negative_samples) < samples_per_class and not neg_chunk.empty:
            samples_to_take = min(samples_per_class - len(negative_samples), len(neg_chunk))
            negative_samples.append(neg_chunk.sample(samples_to_take))
        
        # Check if we have enough samples from both classes
        if (len(positive_samples) >= samples_per_class and 
            len(negative_samples) >= samples_per_class):
            break
    
    # Combine samples
    positive_df = pd.concat(positive_samples) if positive_samples else pd.DataFrame(columns=column_names)
    negative_df = pd.concat(negative_samples) if negative_samples else pd.DataFrame(columns=column_names)
    
    # Ensure we have exactly samples_per_class from each class
    positive_df = positive_df.head(samples_per_class)
    negative_df = negative_df.head(samples_per_class)
    
    # Combine and shuffle
    subset_df = pd.concat([positive_df, negative_df])
    subset_df = subset_df.sample(frac=1, random_state=42)  # shuffle with fixed random seed
    
    # Save the subset
    subset_path = os.path.join(PROCESSED_DATA_DIR, "amazon_reviews_subset.csv")
    subset_df.to_csv(subset_path, index=False)
    
    print(f"Created subset with {len(subset_df)} reviews ({subset_df['sentiment'].value_counts()[1]} positive, "
          f"{subset_df['sentiment'].value_counts()[0]} negative)")
    print(f"Subset saved to {subset_path}")
    
    # Create train/test split
    train_df = subset_df.sample(frac=0.8, random_state=42)
    test_df = subset_df.drop(train_df.index)
    
    train_path = os.path.join(PROCESSED_DATA_DIR, "amazon_reviews_train.csv")
    test_path = os.path.join(PROCESSED_DATA_DIR, "amazon_reviews_test.csv")
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Created train set with {len(train_df)} reviews and test set with {len(test_df)} reviews")
    print(f"Train set saved to {train_path}")
    print(f"Test set saved to {test_path}")


def main():
    print("Starting Amazon Reviews dataset preparation...")
    
    # # Ensure necessary directories exist
    # ensure_directories()
    #
    # # Download dataset if not present
    # download_dataset()
    #
    # Extract dataset
    if extract_dataset():
        # Create subset for analysis
        create_subset()
        print("Dataset preparation completed successfully!")
    else:
        print("Dataset preparation failed.")


if __name__ == "__main__":
    main()
