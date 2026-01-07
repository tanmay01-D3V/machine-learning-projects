
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

print('Data Set Loaded Successfully')
diabetes = load_diabetes()

df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
target = pd.Series(diabetes.target, name='progression')

print('--- HEAD ---')
print(df.head())

X_train, X_test, y_train, y_test = train_test_split(
    df, target, test_size=0.2, random_state=42
)

print(f'\nTraining set: {X_train.shape[0]} rows')
print(f'Testing set: {X_test.shape[0]} rows')

print('\n---Feature Engineering: Polynomial Features---')
degree = 2
print(f'Applying PolynomialFeatures with degree={degree}')
poly = PolynomialFeatures(degree=degree)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

print(f'Original fitures: {X_train.shape[1]}')
print(f'Polynomial fitures: {X_train_poly.shape[1]}')


print('\n---Training the Model---')
model = LinearRegression()
model.fit(X_train_poly, y_train)

print('Model trained successfully.')
predictions = model.predict(X_test_poly)

print('\nFirst 5 Predictions:', predictions[:5])
print('First 5 Actual Values:', y_test.values[:5])

print('\n---Evaluating the Model---')
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print('Mean Squared Error (MSE):', mse)
print('Root Mean Squared Error (RMSE):', rmse)
print('Mean Absolute Error (MAE):', mae)
print('R-Squared Score:', r2)

print('\n--- Model Performance ---')
print(f'Mean Squared Error: {mse:.2f}')
print(f'R-Squared Score:   {r2:.2f}')
