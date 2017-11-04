import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics.scorer import accuracy_score
from sklearn.naive_bayes import GaussianNB

import util_00 as util

n_component = [3, 6, 8 , 10, 15]
x_train, x_test, y_train, y_test = util.get_data()
print(x_train.shape)
print(y_train.shape)
x_train = [x.reshape(1, -1)[0] for x in x_train]
x_test = [x.reshape(1, -1)[0] for x in x_test]

#pre-processing to extract more meaningful features
pca = PCA(svd_solver='randomized', n_components=6, random_state=2017) 
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)       

model = GaussianNB()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

gnb_pkl_filename = 'gnb_classifier_20171105.pkl'
gnb_model_pkl = open(gnb_pkl_filename, 'wb')
pickle.dump(model, gnb_model_pkl)
gnb_model_pkl.close()

# Loading the saved decision tree model pickle
#gnb_model_pkl = open(gnb_pkl_filename, 'rb')
#model = pickle.load(gnb_model_pkl)

print(classification_report(y_test, y_pred))
print('accuracy_score:', accuracy_score(y_test, y_pred))

mat = confusion_matrix(y_test, y_pred)
sns_plot = sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.savefig('gnb_confusion.png')
plt.show()
