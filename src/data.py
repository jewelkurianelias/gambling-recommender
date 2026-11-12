import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def clean_data(df):
    """Handles missing values, incorrect data types, and anomalies."""
    df = df.copy()
    
    # Convert lines to numeric, coercing string errors (like " ") to NaN
    df['over_under_line'] = pd.to_numeric(df['over_under_line'], errors='coerce')
    
    # Clean up schedule_week (convert strings like "Wildcard" to numeric equivalents)
    df['schedule_week'] = pd.to_numeric(df['schedule_week'], errors='coerce').fillna(18)
    
    # Drop rows where games haven't happened yet or lack crucial betting lines
    df = df.dropna(subset=['score_home', 'score_away', 'spread_favorite'])
    
    return df

def engineer_features(df):
    """Creates target variables and betting-specific features."""
    df = df.copy()
    
    # Feature: Was the home team the favorite?
    df['home_favorite'] = (df['team_home'] == df['team_favorite_id']).astype(int)
    
    # Target: Did the favorite cover the spread? 
    conditions = [
        (df['home_favorite'] == 1) & ((df['score_home'] - df['score_away']) > abs(df['spread_favorite'])),
        (df['home_favorite'] == 0) & ((df['score_away'] - df['score_home']) > abs(df['spread_favorite']))
    ]
    df['favorite_covered'] = np.select(conditions, [1, 1], default=0)
    
    return df

def preprocess(df):
    """Full pipeline: cleans, engineers features, and imputes."""
    df_cleaned = clean_data(df)
    df_engineered = engineer_features(df_cleaned)
    
    # Select features for the model
    features = ['schedule_season', 'schedule_week', 'spread_favorite', 'over_under_line', 'home_favorite']
    X = df_engineered[features].copy()
    y = df_engineered['favorite_covered']
    
    # Impute any remaining missing values (like missing over/under lines in older games)
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    return X_imputed, y