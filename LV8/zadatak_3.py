import numpy as np
from tensorflow import keras
from keras import layers
from matplotlib import pyplot as plt
from keras import models
import cv2

model = models.load_model('model.h5')

img = cv2.imread('/home/zvonimir/repos/osu-lv/LV8/test3.png', cv2.IMREAD_GRAYSCALE)

img = img.astype("float32") / 255

img = np.reshape(img, (1, 784))

prediction = model.predict(img)

plt.figure()
plt.imshow(img.reshape(28, 28), cmap='gray')
plt.title(f'Prediction: {np.argmax(prediction)}')
plt.show()