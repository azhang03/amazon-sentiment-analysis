#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Text preprocessing utilities for the Amazon reviews sentiment analysis.
"""

import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
from tqdm import tqdm

# Download necessary NLTK resources
def download_nltk_resources():
    """Download NLTK resources needed for preprocessing."""
    resources = ['punkt', 'stopwords', 'wordnet']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            print(f"Downloading {resource}...")
            nltk.download(resource, quiet=True)
    print("NLTK resources checked and downloaded if needed.")


class TextPreprocessor:
    """Class for preprocessing text data in preparation for sentiment analysis."""
    
    def __init__(self, remove_stopwords=True, lemmatize=True):
        """
        Initialize the text preprocessor.
        
        Args:
            remove_stopwords (bool): Whether to remove stopwords
            lemmatize (bool): Whether to lemmatize tokens
        """
        # Ensure NLTK resources are available
        download_nltk_resources()
        
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        
        # Initialize tools
        if remove_stopwords:
            self.stopwords = set(stopwords.words('english'))
        
        if lemmatize:
            self.lemmatizer = WordNetLemmatizer()
        
        # Compile regex patterns for efficiency
        # Pattern to find URLs
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        # Pattern to find HTML tags
        self.html_pattern = re.compile(r'<.*?>')
        # Pattern to find multiple spaces
        self.space_pattern = re.compile(r'\s+')
        
        print("Text preprocessor initialized.")
    
    def _clean_text(self, text):
        """
        Basic text cleaning.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            # handle NaN or other non-string values
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = self.url_pattern.sub(' ', text)
        
        # Remove HTML tags
        text = self.html_pattern.sub(' ', text)
        
        # Remove punctuation
        # kept as a separate step instead of using a translate table
        # cuz sometimes i like to keep track of certain punctuation
        text = ''.join([char for char in text if char not in string.punctuation])
        
        # Remove extra whitespace
        text = self.space_pattern.sub(' ', text)
        
        return text.strip()
    
    def _tokenize_and_process(self, text):
        """
        Tokenize and further process text.
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of processed tokens
        """
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords if enabled
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stopwords]
        
        # Lemmatize if enabled
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return tokens
    
    def preprocess_text(self, text):
        """
        Preprocess a single text.
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of preprocessed tokens
        """
        # Clean the text
        cleaned_text = self._clean_text(text)
        
        # Tokenize and process
        processed_tokens = self._tokenize_and_process(cleaned_text)
        
        return processed_tokens
    
    def preprocess_dataframe(self, df, text_column='review', new_column='processed_text'):
        """
        Preprocess text in a dataframe column.
        
        Args:
            df (pandas.DataFrame): Input dataframe
            text_column (str): Column containing text to preprocess
            new_column (str): Column to store preprocessed text
            
        Returns:
            pandas.DataFrame: Dataframe with preprocessed text
        """
        print(f"Preprocessing text in column '{text_column}'...")
        
        # Make a copy to avoid modifying the original
        processed_df = df.copy()
        
        # Apply preprocessing to each row with progress tracking
        tqdm.pandas(desc="Preprocessing")
        processed_df[new_column] = processed_df[text_column].progress_apply(
            lambda x: self.preprocess_text(x)
        )
        
        # Add a column with the joined tokens for easier inspection
        processed_df['processed_text_str'] = processed_df[new_column].apply(lambda x: ' '.join(x))
        
        print(f"Preprocessing complete. New columns: '{new_column}' and 'processed_text_str' added.")
        return processed_df
    
    def add_text_features(self, df, text_column='review', processed_column='processed_text'):
        """
        Add text-based features to the dataframe.
        
        Args:
            df (pandas.DataFrame): Input dataframe
            text_column (str): Column containing original text
            processed_column (str): Column containing preprocessed tokens
            
        Returns:
            pandas.DataFrame: Dataframe with additional features
        """
        print("Adding text features...")
        
        # Make a copy to avoid modifying the original
        features_df = df.copy()
        
        # Original text length
        features_df['text_length'] = features_df[text_column].str.len()
        
        # Word count
        features_df['word_count'] = features_df[text_column].apply(lambda x: len(str(x).split()))
        
        # Average word length
        features_df['avg_word_length'] = features_df[text_column].apply(
            lambda x: sum(len(word) for word in str(x).split()) / len(str(x).split()) if len(str(x).split()) > 0 else 0
        )
        
        # Exclamation mark count (from original text, before cleaning)
        features_df['exclamation_count'] = features_df[text_column].str.count('!')
        
        # Question mark count
        features_df['question_count'] = features_df[text_column].str.count('\?')
        
        # Capital letters ratio
        features_df['capitals_ratio'] = features_df[text_column].apply(
            lambda x: sum(1 for c in str(x) if c.isupper()) / len(str(x)) if len(str(x)) > 0 else 0
        )
        
        # Count of tokens after preprocessing (gives insight into stopword removal)
        features_df['processed_token_count'] = features_df[processed_column].apply(len)
        
        print("Text features added.")
        return features_df


if __name__ == "__main__":
    # Example usage
    from data_acquisition import PROCESSED_DATA_DIR
    import os
    
    print("Running text preprocessing script...")
    
    # Load the subset
    subset_path = os.path.join(PROCESSED_DATA_DIR, "amazon_reviews_train.csv")
    if os.path.exists(subset_path):
        print(f"Loading dataset from {subset_path}")
        df = pd.read_csv(subset_path)
        
        # Initialize preprocessor
        preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=True)
        
        # Preprocess the text
        preprocessed_df = preprocessor.preprocess_dataframe(df, text_column='review')
        
        # Add text features
        final_df = preprocessor.add_text_features(preprocessed_df)
        
        # Save the preprocessed data
        preprocessed_path = os.path.join(PROCESSED_DATA_DIR, "amazon_reviews_train_preprocessed.csv")
        final_df.to_csv(preprocessed_path, index=False)
        print(f"Preprocessed data saved to {preprocessed_path}")
        
        # Print a sample
        print("\nSample of preprocessed data:")
        print(final_df[['sentiment', 'text_length', 'word_count', 'processed_text_str']].head(3))
    else:
        print(f"Dataset not found at {subset_path}. Please run data_acquisition.py first.")
