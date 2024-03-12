import numpy as np
import matplotlib.pyplot as plt

x = np.array([1,2,3,3,1])
y = np.array([1,2,2,1,1])

plt.plot(x, y, linewidth=2, marker='x', markersize=10, markerfacecolor='cornflowerblue', markeredgecolor='white')
plt.axis([0.0, 4.0, 0.0, 4.0])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Graf')   
plt.show()