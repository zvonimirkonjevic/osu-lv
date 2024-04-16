import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
img = Image.imread("/home/zvonimir/repos/osu-lv/LV7/imgs/test_1.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()

unique_colors = len(np.unique(img_array_aprox, axis=0))
print("Number of unique colors in the image:", unique_colors)

km = KMeans(n_clusters=5)
km.fit(img_array_aprox)
labels = km.predict(img_array_aprox)

for i in range(5):
    img_array_aprox[labels == i] = km.cluster_centers_[i]

img_aprox = np.reshape(img_array_aprox, (w,h,d))

plt.figure()
plt.title("Rezultat grupiranja")
plt.imshow(img_aprox)
plt.tight_layout()
plt.show()

# povecavanjem broja clustera povecavamo broj boja koje ce biti prisutne na rezultantnoj slici
K_values = range(1, 11) 
J_values = []

for K in K_values:
    km = KMeans(n_clusters=K)
    km.fit(img_array_aprox)
    J_values.append(km.inertia_)

plt.figure()
plt.plot(K_values, J_values, marker='o')
plt.title("Dependency of J on number of clusters K")
plt.xlabel("Number of clusters (K)")
plt.ylabel("J value")
plt.tight_layout()
plt.show()

# možemo uočiti da je optimalan lakat kod K=4, nakon toga se J vrijednost ne smanjuje značajno