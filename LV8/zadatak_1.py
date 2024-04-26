import numpy as np
from tensorflow import keras
from keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# train i test podaci
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# prikaz karakteristika train i test podataka
print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))

# TODO: prikazi nekoliko slika iz train skupa
plt.subplots(3, 3, figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(y_train[i])
    plt.axis('off')
plt.show()


# skaliranje slike na raspon [0,1]
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

# slike trebaju biti (28, 28, 1)
x_train_s = np.expand_dims(x_train_s, -1)
x_test_s = np.expand_dims(x_test_s, -1)

print("x_train shape:", x_train_s.shape)
print(x_train_s.shape[0], "train samples")
print(x_test_s.shape[0], "test samples")


# pretvori labele
y_train_s = keras.utils.to_categorical(y_train, num_classes)
y_test_s = keras.utils.to_categorical(y_test, num_classes)
x_train_s = np.reshape(x_train_s,(60000, 784))
x_test_s = np.reshape(x_test_s,(10000, 784))


# TODO: kreiraj model pomocu keras.Sequential(); prikazi njegovu strukturu
model = keras.Sequential([
    layers.Dense(100,activation='leaky_relu', input_shape=(784,)),
    layers.Dense(50, activation='leaky_relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.summary()

# TODO: definiraj karakteristike procesa ucenja pomocu .compile()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 16
epochs = 30

# TODO: provedi ucenje mreze
history = model.fit(x_train_s, y_train_s, batch_size=batch_size, epochs=epochs, validation_split=0.2)

predictions = model.predict(x_test_s)

# TODO: Prikazi test accuracy i matricu zabune
score = model.evaluate(x_test_s, y_test_s, verbose=0)
print(score)

y_pred = np.argmax(predictions, axis=1)
y_test = np.argmax(y_test_s, axis=1)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
disp.plot()
plt.show()
# TODO: spremi model

model.save('model.h5')
del model