import numpy as np
from sklearn import tree, svm, ensemble, model_selection, metrics

import util_00 as util

x_train, x_test, y_train, y_test = util.get_pca_data(6)

def adaboost_():
    return ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth = 2), n_estimators = 10, algorithm ='SAMME', random_state = 2017)

def gboost_():
    return ensemble.GradientBoostingClassifier(n_estimators = 10, random_state = 2017)

models = [_adaboost, _gboost]

for i in models:
    print(i)
    model = i()
    model.fit(x_train, y_train)
    y_pred = np.array(model.predict(x_test))
    
    print('Results for', i.__name__)
    util.print_results(y_test, y_pred)
    util.generate_confusion_matrix(y_test, y_pred, i.__name__+'confusion.png')
    