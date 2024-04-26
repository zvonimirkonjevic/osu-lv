import numpy as np
from tensorflow import keras
from keras import layers
from matplotlib import pyplot as plt
from keras import models

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

x_train_s = np.expand_dims(x_train_s, -1)
x_test_s = np.expand_dims(x_test_s, -1)

y_train_s = keras.utils.to_categorical(y_train, 10)
y_test_s = keras.utils.to_categorical(y_test, 10)
x_train_s = np.reshape(x_train_s,(60000, 784))
x_test_s = np.reshape(x_test_s,(10000, 784))

model = models.load_model('model.h5')
model.summary()

predictions = model.predict(x_test_s)

y_pred = np.argmax(predictions, axis=1)
y_true = np.argmax(y_test_s, axis=1)

missclassified = np.where(y_pred != y_true)[0]
plt.subplots(3, 3, figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_test[missclassified[i]], cmap='gray')
    plt.title(f"True: {y_true[missclassified[i]]}, Pred: {y_pred[missclassified[i]]}")
    plt.axis('off')
plt.show()