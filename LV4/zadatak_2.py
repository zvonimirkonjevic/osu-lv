from sklearn.model_selection import train_test_split
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.preprocessing import OneHotEncoder

import numpy as np
import pandas as pd

df = pd.read_csv('data_C02_emission.csv')

input_variables = ['Fuel Consumption City (L/100km)', 
                   'Fuel Consumption Hwy (L/100km)', 
                   'Fuel Consumption Comb (L/100km)', 
                   'Fuel Consumption Comb (mpg)',
                   'Engine Size (L)',
                   'Cylinders',
                   'Fuel Type']

output_variable = ['CO2 Emissions (g/km)']

X = df[input_variables].to_numpy()
y = df[output_variable].to_numpy()

encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X[:, 6].reshape(-1, 1)).toarray()
X = np.hstack((X[:, :-1], X_encoded))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = lm.LinearRegression()
model.fit(X_train, y_train)
print(model.coef_)

y_test_predicted = model.predict(X_test)
plt.scatter(y_test, y_test_predicted, color='blue', alpha=0.5)
plt.show()

print(f"MSE: {metrics.mean_squared_error(y_test, y_test_predicted)}")
print(f"RMSE: {metrics.root_mean_squared_error(y_test, y_test_predicted)}")
print(f"MAE: {metrics.mean_absolute_error(y_test, y_test_predicted)}")
print(f"MAPE: {metrics.mean_absolute_percentage_error(y_test, y_test_predicted) * 100} %")
print(f"R2: {metrics.r2_score(y_test, y_test_predicted)}")

max_error = np.max(np.abs(y_test_predicted - y_test)) 

max_error_index = np.argmax(np.abs(y_test_predicted - y_test))

car_model = df.iloc[max_error_index]['Model']

print(f"Car model with maximum error: {car_model}")