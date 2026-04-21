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
model = joblib.load('salary_model_bg_gross.pkl')

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
    
    # Предсказание на Бруто (BGN)
    gross_bgn = model.predict(input_df)[0]
    
    # Конвертиране в EUR (фиксиран курс 1.95583)
    gross_eur = gross_bgn / 1.95583
    
    # Примерно изчисление на Нето (за България - приблизително 78% от бруто след макс. осиг. праг)
    # За по-голяма точност тук може да се вгради пълната формула за ДОД и осигуровки
    net_eur = gross_eur * 0.77 

    return {
        "gross_eur": round(gross_eur, 2),
        "net_eur": round(net_eur, 2)
    }