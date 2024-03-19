import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data_C02_emission.csv')

print(df.head(15))

# a)
print(len(df))

print(df.dtypes)

print(df.isnull())

df.dropna(axis=0)

df.drop_duplicates()

for col in df:
  if type(df[col][0]) == type('abc'):
    df[col] = df[col].astype('category')

print(df.dtypes)

# b)
max_fuel_consumption = df.sort_values(by=['Fuel Consumption City (L/100km)'], ascending=False)
print(max_fuel_consumption.loc[:, ['Make', 'Model', 'Fuel Consumption City (L/100km)']].head(3))

min_fuel_consumption = df.sort_values(by=['Fuel Consumption City (L/100km)'], ascending=True)
print(min_fuel_consumption.loc[:, ['Make', 'Model', 'Fuel Consumption City (L/100km)']].head(3))

# c)
print(len(df[(df['Engine Size (L)'] > 2.5 ) & ( df['Engine Size (L)'] < 3.5 )]))

engine_size_emmision = df[(df['Engine Size (L)'] > 2.5 ) & ( df['Engine Size (L)'] < 3.5 )]['CO2 Emissions (g/km)']
print(engine_size_emmision.mean())

# d)
print(len(df[(df['Make'] == 'Audi')]))

audi_with_4_cylinders = df[(df['Make'] == 'Audi' ) & (df['Cylinders'] == 4)]

print(audi_with_4_cylinders['CO2 Emissions (g/km)'].mean())

# e)
unique_cylinder_values = df.Cylinders.unique()

for cylinder in unique_cylinder_values:
  print(df[(df['Cylinders'] == cylinder)]['CO2 Emissions (g/km)'].mean())

# f)
print(df[(df['Fuel Type'] == 'D')]['Fuel Consumption City (L/100km)'].mean())

print(df[(df['Fuel Type'] == 'X')]['Fuel Consumption City (L/100km)'].mean())

print(df[(df['Fuel Type'] == 'D')]['Fuel Consumption City (L/100km)'].median())

print(df[(df['Fuel Type'] == 'X')]['Fuel Consumption City (L/100km)'].median())

# g)
max_city_fuel_consumption = df[(df['Fuel Type'] == 'D') & (df['Cylinders']==4)]['Fuel Consumption City (L/100km)'].max()

print(df[(df['Fuel Type'] == 'D') & (df['Cylinders']==4) & (df['Fuel Consumption City (L/100km)'] == max_city_fuel_consumption)])

# h)
print(len(df[(df['Transmission'].str.startswith('M'))]))

# i)
print(df.corr(numeric_only=True))