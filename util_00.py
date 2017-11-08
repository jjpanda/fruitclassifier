import os
import pickle
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score, classification_report
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import time

#check directory exist
def checkdir(absolute_path):
    print('checking directory:', absolute_path)
    if not os.path.exists(absolute_path):
        print('Creating ''%s''' % absolute_path)
        os.makedirs(absolute_path)


train_data_dir = "data/train"
test_data_dir = "data/test"

fruits = {'apple':0, 'orange':1, 'peach':2, 'pineapple':3}

def get_tick_labels():
    fruit_label = []
    for name, num in fruits.items():
        fruit_label.append(str(name))
    print(fruit_label)
    return fruit_label

def _load_images(path, fruits):
    data = []
    labels = []
    dirs = os.listdir(path)
    for subpath in dirs:
        label = fruits[subpath]
        path1sub = path + '/' + str(subpath) + '/'
        dirs = os.listdir(path1sub)
        for item in dirs:
            if os.path.isfile(path1sub+item):
                im = Image.open(path1sub+item)
                data.append(np.array(im))
                labels.append(label)
    return np.array(data), np.array(labels)

def get_data():
    x_train, y_train = get_train_data()
    x_test, y_test = get_test_data()
    return x_train, x_test, y_train, y_test

def get_train_data():
    return _load_images(train_data_dir, fruits)

def get_test_data():
    return _load_images(test_data_dir, fruits)
	
def get_resize_dimension():
    resize_width = 128
    resize_height = 128
    return resize_width, resize_height
    
def print_results(y_test, y_pred):
    print(classification_report(y_test, y_pred))
    print('Accuracy score: ', accuracy_score(y_test, y_pred))
    print('Precision score: ', precision_score(y_test, y_pred, average='weighted'))
    print('Recall score: ', recall_score(y_test, y_pred, average='weighted'))
    print('F1 score: ', f1_score(y_test, y_pred, average='weighted'))

def generate_confusion_matrix(y_test, y_pred, image_name_w_ext):
    mat = confusion_matrix(y_test, y_pred)
    labels = get_tick_labels()
    print(labels)
    sns_plot = sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,  xticklabels=labels, yticklabels=labels)
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.savefig(image_name_w_ext)

def get_pca_data(pca_component):
    x_train, x_test, y_train, y_test = get_data()
    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)
    x_train = [x.reshape(1, -1)[0] for x in x_train]
    x_test = [x.reshape(1, -1)[0] for x in x_test]

    #pre-processing to extract more meaningful features
    pca = PCA(svd_solver='randomized', n_components=pca_component, random_state=2017) 
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)       
    return x_train, x_test, y_train, y_test

def save_pkl(model, filename_only):
    _model_pkl = open(str(filename_only) + '_' + time.strftime("%d%m%Y") + '.pkl', 'wb')
    pickle.dump(model, _model_pkl)
    _model_pkl.close()