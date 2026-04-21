from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="AI Salary predictor API")

# Позволяваме на нашия HTML файл да комуникира с API-то
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Позволява заявки от всякъде (добро за локална разработка)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load('salary_model.pkl')

class UserInput(BaseModel):
    years_experience: float

@app.post("/predict")
def predict_salary(data: UserInput):
    X_new = np.array([[data.years_experience]])
    

    prediction = model.predict(X_new)
    

    return {
        "years_experience": data.years_experience,
        "predicted_salary_bgn": round(prediction[0], 2)
    }