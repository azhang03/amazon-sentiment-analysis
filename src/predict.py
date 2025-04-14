#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script for making predictions on new Amazon reviews using the trained model.
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from text_preprocessing import TextPreprocessor

# Set path constants
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
VISUALIZATIONS_DIR = os.path.join(PROJECT_DIR, "visualizations")


def load_model(model_path):
    """
    Load a trained model from a pickle file.
    
    Args:
        model_path (str): Path to the model file
        
    Returns:
        object: Loaded model
    """
    print(f"Loading model from {model_path}")
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def preprocess_new_review(review_text, preprocessor=None):
    """
    Preprocess a new review.
    
    Args:
        review_text (str): Text of the review
        preprocessor (TextPreprocessor, optional): Preprocessor instance
        
    Returns:
        str: Preprocessed text
    """
    if preprocessor is None:
        # i'm recreating the preprocessor here so we don't need to pass it in
        # not the most efficient, but it's fine for one-off predictions
        preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=True)
    
    # convert the review to a DataFrame with the right column name
    review_df = pd.DataFrame({'review': [review_text]})
    
    # preprocess it
    processed_df = preprocessor.preprocess_dataframe(review_df)
    
    # return the processed text string
    return processed_df['processed_text_str'].iloc[0]


def predict_sentiment(model, review_text, preprocessor=None):
    """
    Predict the sentiment of a review.

    Args:
        model: Trained model
        review_text (str): Text of the review
        preprocessor (TextPreprocessor, optional): Preprocessor instance

    Returns:
        dict: Prediction results including sentiment and probability
    """
    # Preprocess the review
    processed_text = preprocess_new_review(review_text, preprocessor)

    # Make prediction
    # most of our models should be pipelines that include vectorization
    sentiment = model.predict([processed_text])[0]

    # Get probability if available
    probability = None
    if hasattr(model, 'predict_proba'):
        probability = model.predict_proba([processed_text])[0]

    # Map sentiment to label
    sentiment_label = "Positive" if sentiment == 1 else "Negative"

    # Format and return results
    results = {
        'sentiment': sentiment,
        'sentiment_label': sentiment_label,
        'processed_text': processed_text,
        'review_text': review_text  # <<< ADD THIS LINE
    }

    # add probability info if available
    if probability is not None:
        results['probability'] = probability
        results['confidence'] = probability[sentiment]

    return results


def analyze_batch_reviews(model, reviews_list, preprocessor=None):
    """
    Analyze a batch of reviews and visualize the results.
    
    Args:
        model: Trained model
        reviews_list (list): List of review texts
        preprocessor (TextPreprocessor, optional): Preprocessor instance
        
    Returns:
        pd.DataFrame: DataFrame with prediction results
    """
    # if we don't have a preprocessor yet, create one
    if preprocessor is None:
        preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=True)
    
    # Prepare a DataFrame to store results
    results = []
    
    print("Analyzing reviews...")
    for i, review in enumerate(reviews_list):
        # Predict sentiment
        prediction = predict_sentiment(model, review, preprocessor)
        
        # Add review info
        prediction['review_id'] = i + 1
        prediction['review_text'] = review
        
        # Append to results
        results.append(prediction)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Visualize sentiment distribution
    if len(results_df) > 1:  # Only visualize if we have multiple reviews
        plt.figure(figsize=(8, 6))
        sentiment_counts = results_df['sentiment_label'].value_counts()
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
        plt.title('Sentiment Analysis Results', fontsize=16)
        plt.xlabel('Sentiment', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        
        # Add count labels
        for i, count in enumerate(sentiment_counts.values):
            plt.text(i, count + 0.1, str(count), ha='center', fontsize=12)
        
        # Save plot
        os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
        save_path = os.path.join(VISUALIZATIONS_DIR, "batch_analysis_results.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Results visualization saved to {save_path}")
    
    return results_df


def print_prediction_result(prediction):
    """
    Print a formatted prediction result.
    
    Args:
        prediction (dict): Prediction result from predict_sentiment
    """
    print("\n" + "="*50)
    print("SENTIMENT ANALYSIS RESULT")
    print("="*50)
    print(f"Sentiment: {prediction['sentiment_label']}")
    
    if 'confidence' in prediction:
        print(f"Confidence: {prediction['confidence']*100:.2f}%")
    
    # Add some insight based on the sentiment
    if prediction['sentiment'] == 1:
        if prediction.get('confidence', 1.0) > 0.8:
            print("This review is strongly positive.")
        else:
            print("This review is somewhat positive.")
    else:
        if prediction.get('confidence', 1.0) > 0.8:
            print("This review is strongly negative.")
        else:
            print("This review is somewhat negative.")
    
    print("\nOriginal Review:")
    print("-"*50)
    print(prediction['review_text'])
    print("="*50)


def main():
    # Set default model path (can be overridden by command line argument)
    model_path = os.path.join(MODELS_DIR, "logistic_regression_tfidf_pipeline.pkl")
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    # Load model
    model = load_model(model_path)
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=True)
    
    # Interactive mode
    print("Amazon Review Sentiment Analyzer")
    print("="*50)
    print("Enter a review to analyze its sentiment, or type 'exit' to quit.")
    print("You can also type 'batch' to analyze multiple reviews at once.")
    
    while True:
        print("\nEnter your choice:")
        choice = input("> ").strip().lower()
        
        if choice == 'exit':
            print("Goodbye!")
            break
        
        elif choice == 'batch':
            print("\nBatch Analysis Mode")
            print("Enter reviews one by one. Type 'done' when finished.")
            
            reviews = []
            i = 1
            while True:
                print(f"\nReview #{i}:")
                review = input("> ").strip()
                
                if review.lower() == 'done':
                    break
                
                reviews.append(review)
                i += 1
            
            if reviews:
                results_df = analyze_batch_reviews(model, reviews, preprocessor)
                
                print("\nAnalysis Results:")
                for i, row in results_df.iterrows():
                    print(f"\nReview #{row['review_id']}:")
                    print(f"Sentiment: {row['sentiment_label']}")
                    if 'confidence' in row:
                        print(f"Confidence: {row['confidence']*100:.2f}%")
        
        else:
            # Treat as a review
            review_text = choice
            prediction = predict_sentiment(model, review_text, preprocessor)
            print_prediction_result(prediction)


if __name__ == "__main__":
    main()
