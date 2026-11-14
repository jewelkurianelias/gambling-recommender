import os
from pathlib import Path
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from src.data import preprocess

# Point this to your AWS EC2 instance running the MLflow tracking server
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("sports-betting-recommender")

# Dynamically resolve the absolute path to the project root
BASE_DIR = Path(__file__).resolve().parent.parent

def train_model():
    # 1. Load BOTH files using dynamic absolute paths
    print("Loading datasets...")
    train_path = BASE_DIR / "data" / "spreadspoke_scores.csv"
    test_path = BASE_DIR / "data" / "test_data.csv"
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # 2. Prepare historical data and split for evaluation
    X, y = preprocess(train_df, is_training=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Prepare future data for actual predictions (No labels expected)
    X_future = preprocess(test_df, is_training=False)
    
    with mlflow.start_run():
        n_estimators = 150
        max_depth = 7
        
        mlflow.log_params({"n_estimators": n_estimators, "max_depth": max_depth})
        
        # 3. Train the model
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        clf.fit(X_train, y_train)
        
        # 4. Evaluate against the historical test split
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        
        mlflow.log_metrics({"accuracy": accuracy, "precision": precision})
        
        # Log the model artifact to EC2
        mlflow.sklearn.log_model(clf, "recommender_model")
        print(f"Model trained successfully.")
        print(f"Accuracy on historical split: {accuracy:.3f}")
        print(f"Precision on historical split: {precision:.3f}")
        
        # 5. Predict on future, unplayed games (test_data.csv)
        future_predictions = clf.predict(X_future)
        print(f"Successfully generated {len(future_predictions)} predictions for upcoming games.")

if __name__ == "__main__":
    train_model()