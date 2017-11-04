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


train_data_dir = "data/augmented"
validation_data_dir = "data/augmented"

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
    fruits = _get_fruit_labels(train_data_dir)
    x_data, y_data = _load_images(train_data_dir, fruits)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.33, random_state=2017)
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)
