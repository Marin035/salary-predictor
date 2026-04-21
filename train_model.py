import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

X = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])

y = np.array([1500, 2000, 2800, 3500, 4200, 4800, 5500, 6100, 6900, 7500])

model = LinearRegression()
model.fit(X,y)

joblib.dump(model, 'salary_model.pkl')

print("Моделът е успешно трениран и запазен като 'salary_model.pkl'!")