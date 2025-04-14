#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script for visualizing the results of the sentiment analysis.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import pickle
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import nltk
from nltk.corpus import stopwords

# Set path constants
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
VISUALIZATIONS_DIR = os.path.join(PROJECT_DIR, "visualizations")


def load_data():
    """Load the preprocessed datasets."""
    train_path = os.path.join(PROCESSED_DATA_DIR, "amazon_reviews_train_preprocessed.csv")
    test_path = os.path.join(PROCESSED_DATA_DIR, "amazon_reviews_test_preprocessed.csv")
    
    print(f"Loading training data from {train_path}")
    train_df = pd.read_csv(train_path)
    
    print(f"Loading test data from {test_path}")
    test_df = pd.read_csv(test_path)
    
    return train_df, test_df


def load_model(model_path):
    """Load a trained model."""
    print(f"Loading model from {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def generate_wordcloud(text_series, title, filename):
    """
    Generate a word cloud from text.
    
    Args:
        text_series (pd.Series): Series of preprocessed text
        title (str): Title for the word cloud
        filename (str): Filename to save the word cloud
    """
    print(f"Generating word cloud for {title}...")
    
    # Combine all text
    text = ' '.join(text_series.astype(str))
    
    # Generate word cloud
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        max_words=100,
        contour_width=3,
        colormap='viridis',
        random_state=42
    ).generate(text)
    
    # Plot
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16)
    
    # Save
    save_path = os.path.join(VISUALIZATIONS_DIR, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Word cloud saved to {save_path}")


def generate_word_frequency_plot(text_series, title, filename, top_n=20):
    """
    Generate a bar plot of word frequencies.
    
    Args:
        text_series (pd.Series): Series of preprocessed text
        title (str): Title for the plot
        filename (str): Filename to save the plot
        top_n (int): Number of top words to include
    """
    print(f"Generating word frequency plot for {title}...")
    
    # Ensure we have NLTK stopwords
    try:
        stops = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords', quiet=True)
        stops = set(stopwords.words('english'))
    
    # Combine and tokenize all text
    all_words = []
    for text in text_series:
        try:
            # if text is a string, split it
            if isinstance(text, str):
                words = text.split()
            # if text is a list, use it directly
            else:
                words = text
            
            # filter out stopwords if any slipped through
            all_words.extend([word.lower() for word in words if word.lower() not in stops])
        except:
            continue
    
    # Count word frequencies
    word_counts = Counter(all_words)
    
    # Convert to DataFrame and sort
    word_freq_df = pd.DataFrame({
        'word': list(word_counts.keys()),
        'frequency': list(word_counts.values())
    })
    word_freq_df = word_freq_df.sort_values('frequency', ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='frequency', y='word', data=word_freq_df)
    plt.title(title, fontsize=16)
    plt.xlabel('Frequency', fontsize=14)
    plt.ylabel('Word', fontsize=14)
    
    # Save
    save_path = os.path.join(VISUALIZATIONS_DIR, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Word frequency plot saved to {save_path}")


def visualize_text_features(df):
    """
    Visualize text features and their relationship with sentiment.
    
    Args:
        df (pd.DataFrame): DataFrame with text features and sentiment
    """
    print("Visualizing text features...")
    
    # Map sentiment labels
    sentiment_map = {0: 'Negative', 1: 'Positive'}
    if 'sentiment' in df.columns:
        df['sentiment_label'] = df['sentiment'].map(sentiment_map)
    
    # Features to visualize
    features = ['text_length', 'word_count', 'avg_word_length', 
              'exclamation_count', 'question_count', 'capitals_ratio']
    
    # Create a figure with subplots
    fig, axs = plt.subplots(3, 2, figsize=(15, 12))
    axs = axs.flatten()
    
    # Plot each feature distribution by sentiment
    for i, feature in enumerate(features):
        if feature in df.columns:
            sns.boxplot(x='sentiment_label', y=feature, data=df, ax=axs[i])
            axs[i].set_title(f'{feature.replace("_", " ").title()} by Sentiment')
            axs[i].set_xlabel('Sentiment')
            axs[i].set_ylabel(feature.replace('_', ' ').title())
    
    # Adjust layout
    plt.tight_layout()
    
    # Save
    save_path = os.path.join(VISUALIZATIONS_DIR, "text_features_by_sentiment.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Text features visualization saved to {save_path}")


def plot_roc_curve(model, X_test, y_test, model_name):
    """
    Plot ROC curve for model evaluation.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name (str): Name of the model
    """
    # Check if model supports predict_proba
    if not hasattr(model, 'predict_proba'):
        print(f"Model {model_name} doesn't support probability predictions, skipping ROC curve.")
        return
    
    print(f"Generating ROC curve for {model_name}...")
    
    # Get probability predictions
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Plot
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    
    # Save
    save_path = os.path.join(VISUALIZATIONS_DIR, f"{model_name.replace(' ', '_')}_roc_curve.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ROC curve saved to {save_path}")


def plot_precision_recall_curve(model, X_test, y_test, model_name):
    """
    Plot precision-recall curve for model evaluation.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name (str): Name of the model
    """
    # Check if model supports predict_proba
    if not hasattr(model, 'predict_proba'):
        print(f"Model {model_name} doesn't support probability predictions, skipping precision-recall curve.")
        return
    
    print(f"Generating precision-recall curve for {model_name}...")
    
    # Get probability predictions
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    
    # Plot
    plt.figure(figsize=(8, 8))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend(loc="lower left")
    
    # Save
    save_path = os.path.join(VISUALIZATIONS_DIR, f"{model_name.replace(' ', '_')}_pr_curve.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Precision-recall curve saved to {save_path}")


def plot_sentiment_distribution(df):
    """
    Plot the distribution of sentiments in the dataset.
    
    Args:
        df (pd.DataFrame): DataFrame with sentiment labels
    """
    print("Plotting sentiment distribution...")
    
    # Map sentiment labels
    sentiment_map = {0: 'Negative', 1: 'Positive'}
    if 'sentiment' in df.columns:
        df['sentiment_label'] = df['sentiment'].map(sentiment_map)
    
    # Plot
    plt.figure(figsize=(10, 6))
    sentiment_counts = df['sentiment_label'].value_counts()
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
    plt.title('Distribution of Sentiments', fontsize=16)
    plt.xlabel('Sentiment', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    
    # Add count labels on top of bars
    for i, count in enumerate(sentiment_counts.values):
        plt.text(i, count + 5, str(count), ha='center', fontsize=12)
    
    # Save
    save_path = os.path.join(VISUALIZATIONS_DIR, "sentiment_distribution.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Sentiment distribution plot saved to {save_path}")


def main():
    print("Running visualization script...")
    
    # Ensure visualization directory exists
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
    
    # Load data
    train_df, test_df = load_data()
    
    # Combine data for some visualizations
    all_data = pd.concat([train_df, test_df])
    
    # Visualize sentiment distribution
    plot_sentiment_distribution(all_data)
    
    # Visualize text features
    visualize_text_features(all_data)
    
    # Generate word clouds
    # For positive reviews
    positive_reviews = all_data[all_data['sentiment'] == 1]['processed_text_str']
    generate_wordcloud(positive_reviews, 'Positive Reviews Word Cloud', 'positive_wordcloud.png')
    
    # For negative reviews
    negative_reviews = all_data[all_data['sentiment'] == 0]['processed_text_str']
    generate_wordcloud(negative_reviews, 'Negative Reviews Word Cloud', 'negative_wordcloud.png')
    
    # Generate word frequency plots
    generate_word_frequency_plot(positive_reviews, 'Top Words in Positive Reviews', 'positive_word_freq.png')
    generate_word_frequency_plot(negative_reviews, 'Top Words in Negative Reviews', 'negative_word_freq.png')
    
    # Load best model (assuming it's saved as logistic_regression_tfidf_pipeline.pkl)
    model_path = os.path.join(MODELS_DIR, "logistic_regression_tfidf_pipeline.pkl")
    if os.path.exists(model_path):
        model = load_model(model_path)
        
        # Prepare test data
        X_test = test_df['processed_text_str']
        y_test = test_df['sentiment']
        
        # Plot ROC and precision-recall curves
        plot_roc_curve(model, X_test, y_test, "Logistic Regression")
        plot_precision_recall_curve(model, X_test, y_test, "Logistic Regression")
    else:
        print(f"Model not found at {model_path}. Please run model_training.py first.")
    
    print("Visualization completed.")


if __name__ == "__main__":
    main()
