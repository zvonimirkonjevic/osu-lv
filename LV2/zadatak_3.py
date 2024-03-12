import numpy as np
import matplotlib.pyplot as plt

img = plt.imread('road.jpg')
plt.figure()
plt.imshow(img)
plt.show()

# posvijetliti sliku
img = plt.imread('road.jpg')
value = 100
mask = (255 - img) < value
brighter_img = np.where(mask, 255, img + value)
plt.figure()
plt.imshow(brighter_img)
plt.show()

# prikazati samo drugu četvrtinu slike po širini
img = plt.imread('road.jpg')
cropped_img = img[:, img.shape[1]//4:img.shape[1]//2]
plt.figure()
plt.imshow(cropped_img)
plt.show()


# zarotirati sliku za 90 stupnjeva u smjeru kazaljke na satu
img = plt.imread('road.jpg')
rotated_img = np.rot90(img)
plt.figure()
plt.imshow(rotated_img)
plt.show()

# zrcaliti sliku
img = plt.imread('road.jpg')
mirrored_img = np.fliplr(img)
plt.figure()
plt.imshow(mirrored_img)
plt.show()