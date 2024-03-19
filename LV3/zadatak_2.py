import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

df = pd.read_csv('data_C02_emission.csv')

# a)
plt.figure()
df['CO2 Emissions (g/km)'].plot(kind="hist", bins = 30)
plt.show()

# b)
fuel_colors = {'X': 'yellow', 'Z': 'chartreuse', 'D': 'lightseagreen', 'E': 'steelblue', 'N': 'purple'}

plt.figure()
plt.scatter(df['CO2 Emissions (g/km)'], df['Fuel Consumption City (L/100km)'], c=df['Fuel Type'].map(fuel_colors))
plt.show()
