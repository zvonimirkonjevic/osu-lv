import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('data.csv', delimiter=',')
data = np.delete(data, 0, 0)

# Na temelju veličine numpy polja data, na koliko osoba su izvršena mjerenja?
print(len(data))

# Prikažite odnos visine i mase osobe pomoću naredbe matplotlib.pyplot.scatter.
plt.figure()
plt.scatter(data[:, 1], data[:, 2])
plt.xlabel('Visina')
plt.ylabel('Masa')
plt.title('Odnos visine i mase osobe')
plt.show()

# Ponovite prethodni zadatak, ali prikažite mjerenja za svaku pedesetu osobu na slici
plt.figure()
plt.scatter(data[::50, 1], data[::50, 2])
plt.xlabel('Visina')
plt.ylabel('Masa')
plt.title('Odnos visine i mase osobe')
plt.show()

# Izračunajte i ispišite u terminal minimalnu, maksimalnu i srednju vrijednost visine u ovom podatkovnom skupu
height = data[:, 1]
print(f"Min: {height.min()}, Max: {height.max()}, Mean: {height.mean()}")

# Ponovite zadatak pod d), ali samo za muškarce, odnosno žene
male_height = data[data[:, 0] == 1, 1]
print(f"Min: {male_height.min()}, Max: {male_height.max()}, Mean: {male_height.mean()}")

female_height = data[data[:, 0] == 0, 1]
print(f"Min: {female_height.min()}, Max: {female_height.max()}, Mean: {female_height.mean()}")