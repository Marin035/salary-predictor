import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Създаваме нашия dataset (сурови данни)
data = {
    'Experience': [1, 3, 5, 2, 8, 10, 4, 6, 2, 7, 12, 15],
    'Education': ['Средно', 'Бакалавър', 'Магистър', 'Средно', 'Магистър', 'Бакалавър', 'Бакалавър', 'Магистър', 'Бакалавър', 'Магистър', 'Бакалавър', 'Магистър'],
    'Role': ['QA Engineer', 'Software Developer', 'Software Developer', 'QA Engineer', 'AI Engineer', 'AI Engineer', 'QA Engineer', 'Data Scientist', 'Data Scientist', 'Software Developer', 'AI Engineer', 'Data Scientist'],
    'Salary': [1500, 2500, 3800, 1700, 6000, 6500, 2800, 4500, 3200, 4800, 7000, 8500]
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