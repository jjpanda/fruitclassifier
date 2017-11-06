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

pca_components = [3, 6, 9 , 12, 15, 18, 21, 24, 50, 100, 150, 200]

for n in pca_components:
    pca = PCA(n_components=n).fit(x_train)
    print(pca.explained_variance_ratio_)
    print(np.cumsum(pca.explained_variance_ratio_))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.savefig('pca_'+ str(n)+'.png')