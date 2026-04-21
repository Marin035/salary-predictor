import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Реалистични данни за България (Бруто в BGN)
# Тук добавяме по-голям обем данни за по-добра статистика
data = {
    'Experience': [0, 1, 2, 3, 5, 8, 10, 0, 2, 5, 1, 4, 7, 2, 6, 12, 3, 9],
    'Education': ['Бакалавър', 'Бакалавър', 'Магистър', 'Магистър', 'Магистър', 'Магистър', 'Магистър', 'Средно', 'Бакалавър', 'Магистър', 'Средно', 'Бакалавър', 'Магистър', 'Средно', 'Бакалавър', 'Магистър', 'Бакалавър', 'Магистър'],
    'Role': [
        'Junior Developer', 'Junior Developer', 'Software Developer', 'Software Developer', 'Senior Developer', 'Architect', 'Architect',
        'Customer Service', 'Sales Expert', 'Sales Expert', 'Logistics Specialist', 'Logistics Specialist', 'Logistics Specialist',
        'Customer Service', 'Software Developer', 'AI Engineer', 'QA Engineer', 'AI Engineer'
    ],
    # Брутни суми в BGN, съобразени с българския пазар
    'Salary_Gross': [
        2500, 3200, 4500, 5200, 7500, 11000, 14000, 
        1600, 2800, 5500, 1800, 3200, 4800, 
        2100, 6200, 15000, 4000, 13000
    ]
}

df = pd.DataFrame(data)
X = df[['Experience', 'Education', 'Role']]
y = df['Salary_Gross']

# 2. Настройка на Pipeline (същата архитектура)
preprocessor = ColumnTransformer(
    transformers=[
        ('edu', OrdinalEncoder(categories=[['Средно', 'Бакалавър', 'Магистър']]), ['Education']),
        ('role', OneHotEncoder(handle_unknown='ignore'), ['Role'])
    ],
    remainder='passthrough'
)

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

model_pipeline.fit(X, y)
joblib.dump(model_pipeline, 'salary_model_bg_gross.pkl')