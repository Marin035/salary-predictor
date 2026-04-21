from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="AI Salary Predictor API v2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Зареждаме новия модел
model = joblib.load('salary_model_v2.pkl')

# Вече очакваме 3 параметъра от фронтенда
class UserInput(BaseModel):
    experience: float
    education: str
    role: str

@app.post("/predict")
def predict_salary(data: UserInput):
    input_df = pd.DataFrame([{
        'Experience': data.experience,
        'Education': data.education,
        'Role': data.role
    }])
    
    # 1. Моделът предсказва в базовата валута (BGN)
    prediction_bgn = model.predict(input_df)[0]
    
    # 2. Бизнес логика: Конвертираме в Евро
    prediction_eur = prediction_bgn / 1.95583
    
    # 3. Връщаме новия JSON
    return {
        "predicted_salary_eur": round(prediction_eur, 2)
    }