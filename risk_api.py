# risk_api.py
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

class Issue(BaseModel):
    text: str

app = FastAPI()
model = joblib.load("models/risk_model.joblib")  # pre-trained .joblib file

@app.post("/predict")
def predict(issue: Issue):
    prob = model.predict_proba([issue.text])[0][1]
    return {"risk_score": float(prob)}
