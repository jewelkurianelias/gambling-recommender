import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.impute import SimpleImputer

# Point this to your AWS EC2 instance running the MLflow tracking server
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("sports-betting-recommender")

def prep_data(df):
    """Cleans the spreadspoke dataset and engineers betting features."""
    # Convert lines to numeric, coercing string errors (like " ") to NaN
    df['over_under_line'] = pd.to_numeric(df['over_under_line'], errors='coerce')
    
    # Drop rows where games haven't happened yet or lack betting lines
    df = df.dropna(subset=['score_home', 'score_away', 'spread_favorite'])
    
    # Feature: Was the home team the favorite?
    df['home_favorite'] = (df['team_home'] == df['team_favorite_id']).astype(int)
    
    # Did the favorite cover the spread? 
    # (Spread is usually negative for the favorite, e.g., -7 means they must win by > 7)
    # We define a binary target: 1 if favorite covered, 0 if they didn't
    conditions = [
        (df['home_favorite'] == 1) & ((df['score_home'] - df['score_away']) > abs(df['spread_favorite'])),
        (df['home_favorite'] == 0) & ((df['score_away'] - df['score_home']) > abs(df['spread_favorite']))
    ]
    df['favorite_covered'] = np.select(conditions, [1, 1], default=0)
    
    # Select features for the model
    features = ['schedule_season', 'schedule_week', 'spread_favorite', 'over_under_line', 'home_favorite']
    X = df[features].copy()
    
    # Clean up schedule_week (convert strings like "Wildcard" to numeric equivalents)
    X['schedule_week'] = pd.to_numeric(X['schedule_week'], errors='coerce').fillna(18)
    
    # Impute any remaining missing values (like missing over/under lines in older games)
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    return X_imputed, df['favorite_covered']

def train_model():
    # 1. Load BOTH files provided by the repo
    print("Loading datasets...")
    train_df = pd.read_csv("../data/spreadspoke_scores.csv")
    test_df = pd.read_csv("../data/test_data.csv")
    
    # 2. Prepare and clean datasets
    X_train, y_train = prep_data(train_df)
    X_test, y_test = prep_data(test_df)
    
    with mlflow.start_run():
        n_estimators = 150
        max_depth = 7
        
        mlflow.log_params({"n_estimators": n_estimators, "max_depth": max_depth})
        
        # 3. Train the model
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        clf.fit(X_train, y_train)
        
        # 4. Evaluate against the dedicated test_data.csv
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        
        mlflow.log_metrics({"accuracy": accuracy, "precision": precision})
        
        # Log the model artifact to EC2
        mlflow.sklearn.log_model(clf, "recommender_model")
        print(f"Model trained successfully.")
        print(f"Accuracy on test_data.csv: {accuracy:.3f}")
        print(f"Precision on test_data.csv: {precision:.3f}")

if __name__ == "__main__":
    train_model()