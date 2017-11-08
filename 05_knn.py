import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection

import util_00 as util

x_train, x_test, y_train, y_test = util.get_pca_data(6)

parameters = [
    {'n_neighbors': [i for i in range(3,50)], 'weights': ['distance', 'uniform']}]

model = model_selection.GridSearchCV(KNeighborsClassifier(), parameters, cv = model_selection.StratifiedKFold(n_splits = 5, shuffle = True, random_state = 2017))
model.fit(x_train, y_train)
y_pred = np.array(model.predict(x_test))
print('best params: ', model.best_params_)

util.save_pkl(model, 'knn_classifier')

print('Results for knn')
util.print_results(y_test, y_pred)
util.generate_confusion_matrix(y_test, y_pred, 'knn_confusion.png')