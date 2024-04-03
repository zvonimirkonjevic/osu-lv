import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

plt.figure()
plt.scatter(X_train[:, 0], X_train[:, 1], marker='o', cmap = 'viridis', c=y_train)
plt.scatter(X_test[:, 0], X_test[:, 1], marker='x', cmap='plasma', c=y_test)
plt.title('Binary classification problem')
plt.show()

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

parameters = model.coef_
intercept = model.intercept_

cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()
plt.show()

print(classification_report(y_test, y_pred))

plt.figure()
plt.scatter(X_test[y_test == y_pred][:, 0], X_test[y_test == y_pred][:, 1], marker='o', c='g')
plt.scatter(X_test[y_test != y_pred][:, 0], X_test[y_test != y_pred][:, 1], marker='o', c='r')
plt.title('Correct and incorrect predictions')
plt.show()