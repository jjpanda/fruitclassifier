import pickle

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

import util_00 as util

x_train, x_test, y_train, y_test = util.get_pca_data(6)

model = KNeighborsClassifier(n_neighbors = 10, weights = 'distance')
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

knn_pkl_filename = 'knn_classifier_20171105.pkl'
knn_model_pkl = open(knn_pkl_filename, 'wb')
pickle.dump(model, knn_model_pkl)
knn_model_pkl.close()

print('Results for knn')
util.print_results(y_test, y_pred)
util.generate_confusion_matrix(y_test, y_pred, 'knn_confusion.png')