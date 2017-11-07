import numpy as np
from sklearn import tree, svm, ensemble, model_selection, metrics

import util_00 as util

x_train, x_test, y_train, y_test = util.get_pca_data(6)

#default base estimator = decision three
def bagging_():
    return ensemble.BaggingClassifier(tree.DecisionTreeClassifier(), oob_score = True, random_state = 2017)

def rforest_():
    return ensemble.RandomForestClassifier(max_features = 1, oob_score = True, random_state = 2017)

def extree_():
    return ensemble.ExtraTreesClassifier(max_features = 1, bootstrap = True, oob_score = True, random_state = 2017)

models = [bagging_, rforest_, extree_]

for i in models:
    print(i)
    model = i()
    model.fit(x_train, y_train)
    y_pred = np.array(model.predict(x_test))
    
    print('Results for', i.__name__)
    util.print_results(y_test, y_pred)
    util.generate_confusion_matrix(y_test, y_pred, i.__name__+'confusion.png')