import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Създаваме нашия dataset (сурови данни)
# Обновени данни с новите професии (приблизителни стойности в BGN за обучението)
data = {
    'Experience': [1, 5, 10, 2, 7, 3, 6, 8, 4, 12, 2, 5, 10, 3, 8],
    'Education': ['Средно', 'Магистър', 'Магистър', 'Бакалавър', 'Магистър', 'Средно', 'Бакалавър', 'Магистър', 'Бакалавър', 'Магистър', 'Средно', 'Бакалавър', 'Магистър', 'Бакалавър', 'Магистър'],
    'Role': [
        'QA Engineer', 'AI Engineer', 'AI Engineer', 'Software Developer', 'Software Developer',
        'Sales Expert', 'Sales Expert', 'Sales Expert', # Нова роля
        'Logistics Specialist', 'Logistics Specialist', 'Logistics Specialist', # Нова роля
        'Customer Service', 'Customer Service', 'Customer Service', 'Data Scientist' # Нова роля
    ],
    'Salary': [
        1500, 6000, 8500, 2500, 4800, 
        1800, 3500, 5500, # Sales (с бонуси)
        1600, 3200, 4200, # Logistics
        1400, 2200, 3000, 4500  # Customer Service
    ]
}
df = pd.DataFrame(data)

# Разделяме входа (X) от това, което предсказваме (y)
X = df[['Experience', 'Education', 'Role']]
y = df['Salary']

# 2. Правила за превод (Encoding) от думи към числа
edu_categories = [['Средно', 'Бакалавър', 'Магистър']]

preprocessor = ColumnTransformer(
    transformers=[
        ('edu', OrdinalEncoder(categories=edu_categories), ['Education']), # Образованието има йерархия
        ('role', OneHotEncoder(handle_unknown='ignore'), ['Role'])        # Професиите са просто различни флагове
    ],
    remainder='passthrough' # Опитът (Experience) вече е число, пропускаме го директно
)

# 3. Сглобяваме Тръбопровода (Pipeline)
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# 4. Тренираме всичко наведнъж
model_pipeline.fit(X, y)

# Запазваме целия Pipeline (и преводача, и самия AI)
joblib.dump(model_pipeline, 'salary_model_v2.pkl')
print("Моделът v2 е успешно трениран и запазен!")