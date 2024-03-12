import numpy as np
import matplotlib.pyplot as plt

top_black = np.zeros((50,50))
top_white = np.full((50,50), 255)
upper_array = np.hstack((top_black, top_white))
bottom_black = np.zeros((50,50))
bottom_white = np.full((50,50), 255)
bottom_array = np.hstack((bottom_white, bottom_black))
checkerboard = np.vstack((upper_array, bottom_array))

plt.figure()
plt.imshow(checkerboard, cmap='gray')
plt.show()