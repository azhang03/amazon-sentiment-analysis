"""
Script for training sentiment analysis models on the Amazon reviews dataset.
"""

import os
import pandas as pd
import numpy as np
import pickle
import time
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Set path constants
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
VISUALIZATIONS_DIR = os.path.join(PROJECT_DIR, "visualizations")


def load_data():
    """Load the preprocessed training and testing datasets."""
    train_path = os.path.join(PROCESSED_DATA_DIR, "amazon_reviews_train_preprocessed.csv")
    test_path = os.path.join(PROCESSED_DATA_DIR, "amazon_reviews_test.csv")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Preprocessed training data not found at {train_path}. "
                               "Please run text_preprocessing.py first.")
    
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found at {test_path}. "
                               "Please run data_acquisition.py first.")
    
    print(f"Loading training data from {train_path}")
    train_df = pd.read_csv(train_path)
    
    print(f"Loading test data from {test_path}")
    test_df = pd.read_csv(test_path)
    
    return train_df, test_df


def preprocess_test_data(test_df, preprocessor):
    """Apply preprocessing to the test data."""
    print("Preprocessing test data...")
    test_df = preprocessor.preprocess_dataframe(test_df, text_column='review')
    test_df = preprocessor.add_text_features(test_df)
    return test_df


class SentimentModel:
    """Class for training and evaluating sentiment analysis models."""
    
    def __init__(self, model_type='naive_bayes', vectorizer_type='tfidf'):
        """
        Initialize the sentiment model.
        
        Args:
            model_type (str): Type of model to use ('naive_bayes', 'logistic_regression', 'svm', 'random_forest')
            vectorizer_type (str): Type of vectorizer to use ('count', 'tfidf')
        """
        self.model_type = model_type
        self.vectorizer_type = vectorizer_type
        
        # making sure we can save the model files
        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
        
        # initialize the model based on type
        if model_type == 'naive_bayes':
            self.model = MultinomialNB()
            # this will be our basic model name
            self.model_name = "naive_bayes"
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(random_state=42, max_iter=1000)
            self.model_name = "logistic_regression"
        elif model_type == 'svm':
            self.model = LinearSVC(random_state=42, max_iter=1000)
            self.model_name = "svm"
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(random_state=42, n_estimators=100)
            self.model_name = "random_forest"
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # initialize the vectorizer based on type
        if vectorizer_type == 'count':
            self.vectorizer = CountVectorizer(max_features=5000)
            self.vectorizer_name = "count"
        elif vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(max_features=5000)
            self.vectorizer_name = "tfidf"
        else:
            raise ValueError(f"Unsupported vectorizer type: {vectorizer_type}")
        
        # creating a pipeline to combine vectorizer and model
        # this makes it easier to train and evaluate
        self.pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('model', self.model)
        ])
        
        print(f"Initialized {model_type} model with {vectorizer_type} vectorizer.")
    
    def train(self, X_train, y_train, use_grid_search=False):
        """
        Train the model.
        
        Args:
            X_train: Training data (text)
            y_train: Training labels
            use_grid_search (bool): Whether to use grid search for hyperparameter tuning
        """
        start_time = time.time()
        print(f"Training {self.model_type} model...")
        
        if use_grid_search:
            print("Performing grid search for hyperparameter tuning...")
            
            # Define parameter grid based on model type
            if self.model_type == 'naive_bayes':
                param_grid = {
                    'model__alpha': [0.1, 0.5, 1.0, 2.0]
                }
            elif self.model_type == 'logistic_regression':
                param_grid = {
                    'model__C': [0.1, 1.0, 10.0],
                    'model__solver': ['liblinear', 'saga']
                }
            elif self.model_type == 'svm':
                param_grid = {
                    'model__C': [0.1, 1.0, 10.0],
                }
            elif self.model_type == 'random_forest':
                param_grid = {
                    'model__n_estimators': [50, 100],
                    'model__max_depth': [None, 10, 20]
                }
            
            # Add vectorizer parameters to the grid
            if self.vectorizer_type == 'tfidf':
                param_grid.update({
                    'vectorizer__max_features': [3000, 5000],
                    'vectorizer__ngram_range': [(1, 1), (1, 2)]
                })
            else:  # count vectorizer
                param_grid.update({
                    'vectorizer__max_features': [3000, 5000],
                    'vectorizer__ngram_range': [(1, 1), (1, 2)]
                })
            
            # Create grid search object
            # we're using cv=3 for speed, but in real projects you might use 5 or more
            grid_search = GridSearchCV(
                self.pipeline, param_grid, cv=3, n_jobs=-1, verbose=1, scoring='accuracy'
            )
            
            # Fit the grid search
            grid_search.fit(X_train, y_train)
            
            # Print results
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
            
            # Update the pipeline with the best model
            self.pipeline = grid_search.best_estimator_
        else:
            # Regular training without grid search
            self.pipeline.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds.")
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test data (text)
            y_test: Test labels
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        print("Evaluating model on test data...")
        
        # Predict on test data
        y_pred = self.pipeline.predict(X_test)
        
        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Visualization of confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix - {self.model_type.replace("_", " ").title()}')
        
        # Save confusion matrix plot
        cm_path = os.path.join(VISUALIZATIONS_DIR, f"{self.model_name}_{self.vectorizer_name}_confusion_matrix.png")
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {cm_path}")
        
        # Store evaluation results
        evaluation_results = {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': conf_matrix,
            'predictions': y_pred
        }
        
        return evaluation_results
    
    def save_model(self):
        """Save the trained model and vectorizer."""
        model_path = os.path.join(MODELS_DIR, f"{self.model_name}_{self.vectorizer_name}_pipeline.pkl")
        
        print(f"Saving model to {model_path}")
        with open(model_path, 'wb') as f:
            pickle.dump(self.pipeline, f)
        
        print(f"Model saved successfully to {model_path}")
        return model_path
    
    @classmethod
    def load_model(cls, model_path):
        """
        Load a trained model from a file.
        
        Args:
            model_path (str): Path to the model file
            
        Returns:
            SentimentModel: Loaded model
        """
        print(f"Loading model from {model_path}")
        with open(model_path, 'rb') as f:
            pipeline = pickle.load(f)
        
        # Extract model and vectorizer types from the pipeline
        vectorizer = pipeline.named_steps['vectorizer']
        model = pipeline.named_steps['model']
        
        # Determine model and vectorizer types
        if isinstance(model, MultinomialNB):
            model_type = 'naive_bayes'
        elif isinstance(model, LogisticRegression):
            model_type = 'logistic_regression'
        elif isinstance(model, LinearSVC):
            model_type = 'svm'
        elif isinstance(model, RandomForestClassifier):
            model_type = 'random_forest'
        else:
            model_type = 'unknown'
        
        if isinstance(vectorizer, TfidfVectorizer):
            vectorizer_type = 'tfidf'
        elif isinstance(vectorizer, CountVectorizer):
            vectorizer_type = 'count'
        else:
            vectorizer_type = 'unknown'
        
        # Create a new instance with the determined types
        sentiment_model = cls(model_type=model_type, vectorizer_type=vectorizer_type)
        sentiment_model.pipeline = pipeline
        
        print(f"Model loaded successfully ({model_type} with {vectorizer_type} vectorizer)")
        return sentiment_model


def generate_feature_importance_plot(model, feature_names, top_n=20):
    """
    Generate and save a feature importance plot.
    
    Args:
        model: The trained model
        feature_names: Names of the features
        top_n (int): Number of top features to display
    """
    # Skip if model doesn't support feature importances
    if not hasattr(model, 'coef_') and not hasattr(model, 'feature_importances_'):
        print("Model doesn't support feature importance extraction.")
        return
    
    print(f"Generating feature importance plot for top {top_n} features...")
    
    # Extract feature importances
    if hasattr(model, 'coef_'):
        # For linear models (LogisticRegression, LinearSVC)
        if len(model.coef_.shape) > 1:
            # For multi-class models, use the positive class
            importances = model.coef_[0]
        else:
            importances = model.coef_
    else:
        # For tree-based models (RandomForest)
        importances = model.feature_importances_
    
    # Create a DataFrame of feature importances
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': np.abs(importances)  # Using absolute values for importance
    })
    
    # Sort by importance and get top N
    feature_importance = feature_importance.sort_values('importance', ascending=False).head(top_n)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title(f'Top {top_n} Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    # Save the plot
    model_name = type(model).__name__
    importance_path = os.path.join(VISUALIZATIONS_DIR, f"{model_name}_feature_importance.png")
    plt.savefig(importance_path, dpi=300, bbox_inches='tight')
    print(f"Feature importance plot saved to {importance_path}")


def compare_models(models, X_test, y_test):
    """
    Compare multiple trained models.
    
    Args:
        models (list): List of trained SentimentModel instances
        X_test: Test data
        y_test: Test labels
    """
    print("Comparing model performance...")
    
    # Track results for each model
    results = []
    
    # Evaluate each model
    for model in models:
        eval_results = model.evaluate(X_test, y_test)
        results.append({
            'model_name': f"{model.model_type} ({model.vectorizer_type})",
            'accuracy': eval_results['accuracy'],
            'precision': eval_results['report']['1']['precision'],  # Positive class precision
            'recall': eval_results['report']['1']['recall'],        # Positive class recall
            'f1_score': eval_results['report']['1']['f1-score']     # Positive class F1
        })
    
    # Convert to DataFrame for easier visualization
    results_df = pd.DataFrame(results)
    
    # Create a comparison plot
    plt.figure(figsize=(12, 8))
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    # Set up the positions
    x = np.arange(len(results_df))
    width = 0.2
    
    # Plot each metric as a group of bars
    for i, metric in enumerate(metrics):
        offset = (i - len(metrics)/2 + 0.5) * width
        plt.bar(x + offset, results_df[metric], width=width, label=metric.replace('_', ' ').title())
    
    # Customize the plot
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Model Comparison')
    plt.xticks(x, results_df['model_name'], rotation=45, ha='right')
    plt.legend()
    plt.ylim(0, 1)
    plt.tight_layout()
    
    # Save the plot
    comparison_path = os.path.join(VISUALIZATIONS_DIR, "model_comparison.png")
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"Model comparison plot saved to {comparison_path}")
    
    # Return the results DataFrame
    return results_df


def main():
    print("Running sentiment analysis model training...")
    
    # Load data
    train_df, test_df = load_data()
    
    # import preprocessor from text_preprocessing.py
    from text_preprocessing import TextPreprocessor
    
    # Preprocess test data
    preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=True)
    test_df = preprocess_test_data(test_df, preprocessor)
    
    # Save the preprocessed test data
    test_processed_path = os.path.join(PROCESSED_DATA_DIR, "amazon_reviews_test_preprocessed.csv")
    test_df.to_csv(test_processed_path, index=False)
    
    # Prepare training data
    X_train = train_df['processed_text_str']
    y_train = train_df['sentiment']
    
    # Prepare test data
    X_test = test_df['processed_text_str']
    y_test = test_df['sentiment']
    
    # Train and evaluate different models
    models = [
        SentimentModel(model_type='naive_bayes', vectorizer_type='tfidf'),
        SentimentModel(model_type='logistic_regression', vectorizer_type='tfidf')
    ]
    
    trained_models = []
    for model in models:
        # Train the model (with grid search for the second model)
        use_grid_search = model.model_type != 'naive_bayes'  # Use grid search for models other than Naive Bayes
        model.train(X_train, y_train, use_grid_search=use_grid_search)
        
        # Evaluate the model
        model.evaluate(X_test, y_test)
        
        # Save the model
        model.save_model()
        
        trained_models.append(model)
    
    # Compare model performance
    compare_models(trained_models, X_test, y_test)
    
    # Get the best model (highest accuracy on test data)
    best_model = trained_models[0]  # Start with the first model
    best_accuracy = best_model.evaluate(X_test, y_test)['accuracy']
    
    for model in trained_models[1:]:
        accuracy = model.evaluate(X_test, y_test)['accuracy']
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
    
    print(f"\nBest model: {best_model.model_type} with {best_model.vectorizer_type} vectorizer "
          f"(Accuracy: {best_accuracy:.3f})")
    
    # For the best model, if it's a linear model, generate feature importance plot
    if hasattr(best_model.pipeline.named_steps['model'], 'coef_') or \
       hasattr(best_model.pipeline.named_steps['model'], 'feature_importances_'):
        # Get feature names from the vectorizer
        feature_names = best_model.pipeline.named_steps['vectorizer'].get_feature_names_out()
        
        # Generate feature importance plot
        generate_feature_importance_plot(
            best_model.pipeline.named_steps['model'], 
            feature_names,
            top_n=20
        )
    
    print("\nModel training and evaluation completed.")


if __name__ == "__main__":
    main()
