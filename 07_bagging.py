import numpy as np
from sklearn import tree, svm, ensemble, model_selection, metrics

import util_00 as util

x_train, x_test, y_train, y_test = util.get_pca_data(6)

#default base estimator = decision three
def _bagging():
    return ensemble.BaggingClassifier(tree.DecisionTreeClassifier(), oob_score = True, random_state = 2017)

def _rforest():
    return ensemble.RandomForestClassifier(max_features = 1, oob_score = True, random_state = 2017)

def _extree():
    return ensemble.ExtraTreesClassifier(max_features = 1, bootstrap = True, oob_score = True, random_state = 2017)

models = [_bagging, _rforest, _extree]

for i in models:
    print(i)
    model = i()
    model.fit(x_train, y_train)
    y_pred = np.array(model.predict(x_test))
    
    print('Results for', i.__name__)
    util.print_results(y_test, y_pred)
    util.generate_confusion_matrix(y_test, y_pred, i.__name__+'_confusion.png')