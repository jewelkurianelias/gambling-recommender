import pandas as pd
import numpy as np
from src.data import clean_data, engineer_features, preprocess

def test_clean_data():
    """
    Unit Test 1: Verifies data cleaning logic handles missing values, 
    incorrect data types, and string anomalies correctly.
    """
    mock_data = {
        'score_home': [24, 14, np.nan], # Row 3 is missing scores
        'score_away': [10, 21, np.nan],
        'spread_favorite': [-7.0, -3.0, -3.0],
        'over_under_line': ['45.5', ' ', '50.0'], # ' ' should become NaN
        'schedule_week': ['1', 'Wildcard', '2'] # 'Wildcard' should become 18
    }
    df = pd.DataFrame(mock_data)
    
    cleaned_df = clean_data(df)
    
    # Assertions
    assert len(cleaned_df) == 2, "Should have dropped the row with missing scores"
    assert pd.isna(cleaned_df.iloc[1]['over_under_line']), "Should have coerced ' ' to NaN"
    assert cleaned_df.iloc[1]['schedule_week'] == 18.0, "Should have mapped 'Wildcard' to week 18"


def test_engineer_features():
    """
    Unit Test 2: Verifies the business logic for calculating if a team 
    is the home favorite and if they successfully covered the spread.
    """
    mock_data = {
        'score_home': [24, 10, 10],
        'score_away': [10, 20, 12],
        'team_home': ['TeamA', 'TeamB', 'TeamC'],
        'team_favorite_id': ['TeamA', 'TeamZ', 'TeamC'], 
        'spread_favorite': [-7.0, -3.0, -5.0]
    }
    df = pd.DataFrame(mock_data)
    
    engineered_df = engineer_features(df)
    
    # Assertions for home_favorite
    assert list(engineered_df['home_favorite']) == [1, 0, 1]
    
    # Assertions for favorite_covered
    # Game 1: Home fav wins 24-10 (diff 14 > 7) -> 1 (Covered)
    # Game 2: Away fav wins 20-10 (diff 10 > 3) -> 1 (Covered)
    # Game 3: Home fav loses 10-12 -> 0 (Did not cover)
    assert list(engineered_df['favorite_covered']) == [1, 1, 0]


def test_preprocess():
    """
    Unit Test 3: Verifies the master pipeline stitches everything together,
    imputes missing values (NaNs), and outputs the correct shapes.
    """
    mock_data = {
        'score_home': [24, 10],
        'score_away': [10, 20],
        'team_home': ['TeamA', 'TeamB'],
        'team_favorite_id': ['TeamA', 'TeamC'], 
        'spread_favorite': [-7.0, -3.0], 
        'over_under_line': ['45.5', np.nan], # One missing line
        'schedule_season': [2023, 2023],
        'schedule_week': ['1', '2']
    }
    df = pd.DataFrame(mock_data)
    
    # Process data
    X, y = preprocess(df)
    
    # Assertions for X (Features)
    assert X.shape[0] == 2 # 2 rows
    assert X.shape[1] == 5 # 5 expected features
    
    # Verify the imputer filled the NaN in 'over_under_line' (Column index 3)
    assert not np.isnan(X[1][3]), "Imputer failed to fill NaN values"