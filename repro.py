import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X = pd.DataFrame(np.array([[1, 2], [3, 4], [5, 6], [7, 8]]), columns=['a', 'b'])
y = np.array([1, 2, 3, 4])

# create model
knn = KNeighborsClassifier()

# fit the model
knn.fit(X, y)

# make predictions
knn.predict(X)