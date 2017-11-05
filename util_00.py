import os

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

#check directory exist
def checkdir(absolute_path):
    print('checking directory:', absolute_path)
    if not os.path.exists(absolute_path):
        print('Creating ''%s''' % absolute_path)
        os.makedirs(absolute_path)


train_data_dir = "data/train"
test_data_dir = "data/test"

def _get_fruit_labels(path):
    fruits = {}
    count = 0
    for fruit in os.listdir(path):
        fruits[str(fruit)] = count
        print(fruit)
        count += 1
    return fruits

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
    fruits = _get_fruit_labels(train_data_dir)
    return _load_images(train_data_dir, fruits)

def get_test_data():
    fruits = _get_fruit_labels(test_data_dir)
    return _load_images(test_data_dir, fruits)