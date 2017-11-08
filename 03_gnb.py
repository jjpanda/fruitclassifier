import numpy as np
from sklearn.naive_bayes import GaussianNB

import util_00 as util

x_train, x_test, y_train, y_test = util.get_pca_data(6)

model = GaussianNB()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

util.save_pkl(model, 'gnb_classifier')

print('Results for gnb')
util.print_results(y_test, y_pred)
util.generate_confusion_matrix(y_test, y_pred, 'gnb_confusion.png')