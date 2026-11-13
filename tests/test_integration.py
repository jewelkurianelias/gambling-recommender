from fastapi.testclient import TestClient
from unittest.mock import patch
import numpy as np

# Import the FastAPI app
from src.serve import app

client = TestClient(app)

def test_health_check():
    """
    Integration Test: Ensure the API spins up and responds to health checks.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "healthy"

# Mock the machine learning model so we can test the API logic 
# without needing a live MLflow server running during CI/CD.
class MockModel:
    def predict(self, df):
        return np.array([1]) # Always predicts favorite covers
        
    def predict_proba(self, df):
        return np.array([[0.2, 0.8]]) # 80% confidence

@patch('src.serve.model', MockModel())
def test_predict_endpoint_success():
    """
    Integration Test: Ensure the /predict endpoint accepts valid JSON,
    processes it, and returns the expected schema.
    """
    payload = {
        "schedule_season": 2023,
        "schedule_week": 1,
        "spread_favorite": -4.5,
        "over_under_line": 48.5,
        "home_favorite": 1
    }
    
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "favorite_covers_spread" in data
    assert "confidence_score" in data
    assert data["favorite_covers_spread"] == True
    assert data["confidence_score"] == 0.8

def test_predict_endpoint_validation_error():
    """
    Integration Test: Ensure Pydantic catches missing required fields.
    """
    payload = {
        "schedule_season": 2023
        # Missing other required fields
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422 # Unprocessable Entity (Validation Error)