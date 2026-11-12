import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn

app = FastAPI(title="Sports Betting Recommender API")

# Environment variable pointing to your EC2 MLflow server
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# In a production scenario, you would fetch the specific "Production" tagged model.
# For this project, we specify the model name logged in train.py
MODEL_NAME = "recommender_model"
MODEL_URI = f"models:/{MODEL_NAME}/latest"

print(f"Loading model from {MODEL_URI}...")
try:
    model = mlflow.sklearn.load_model(MODEL_URI)
except Exception as e:
    print(f"Warning: Could not load model immediately. Ensure MLflow is reachable. Error: {e}")
    model = None

class GameFeatures(BaseModel):
    schedule_season: int
    schedule_week: int
    spread_favorite: float
    over_under_line: float
    home_favorite: int

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict_coverage(game: GameFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    
    # Convert incoming JSON payload to a DataFrame
    input_data = pd.DataFrame([game.model_dump()])
    
    # Make prediction
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1] # Probability of class 1 (Covering)
    
    return {
        "favorite_covers_spread": bool(prediction[0]),
        "confidence_score": round(float(probability), 3),
        "recommendation": "Bet Favorite to Cover" if prediction[0] == 1 else "Bet Underdog to Cover"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)