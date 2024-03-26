from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

import numpy as np
import pandas as pd

df = pd.read_csv('data_C02_emission.csv')

df = df.drop(['Make', 'Model'], axis=1)

input_variables = ['Fuel Consumption City (L/100km)', 
                   'Fuel Consumption Hwy (L/100km)', 
                   'Fuel Consumption Comb (L/100km)', 
                   'Fuel Consumption Comb (mpg)',
                   'Engine Size (L)',
                   'Cylinders']

output_variable = ['CO2 Emissions (g/km)']

X = df[input_variables].to_numpy()
y = df[output_variable].to_numpy()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 1)

plt.scatter(X_train[:, 0], y_train, color='blue', alpha=0.5)
plt.scatter(X_test[:, 0], y_test, color='red', alpha=0.5)
plt.legend(['Train', 'Test'])
plt.show()

scaler = MinMaxScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

plt.hist(X_train[:, 3], bins=20, color='blue', alpha=0.5)
plt.hist(X_train_normalized[:, 3], bins=20, color='red', alpha=0.5)
plt.legend(['Original', 'Normalized'])
plt.show()

model = lm.LinearRegression()
model.fit(X_train_normalized, y_train)

print(model.coef_)

y_test_predicted = model.predict(X_test_normalized)
plt.scatter(y_test, y_test_predicted, color='blue', alpha=0.5)
plt.show()

print(f"MSE: {metrics.mean_squared_error(y_test, y_test_predicted)}")
print(f"RMSE: {metrics.root_mean_squared_error(y_test, y_test_predicted)}")
print(f"MAE: {metrics.mean_absolute_error(y_test, y_test_predicted)}")
print(f"MAPE: {metrics.mean_absolute_percentage_error(y_test, y_test_predicted) * 100} %")
print(f"R2: {metrics.r2_score(y_test, y_test_predicted)}")