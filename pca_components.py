import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.scorer import accuracy_scorer
from sklearn.neighbors import KNeighborsClassifier

import util_00 as util

x_train, x_test, y_train, y_test = util.get_data()
print(x_train.shape)
print(y_train.shape)
x_train = [x.reshape(1, -1)[0] for x in x_train]
x_test = [x.reshape(1, -1)[0] for x in x_test]

pca = PCA().fit(x_train)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')