# Image classification of fruits

To build an image classification system to classify 4 kinds of fruits: 
~~~
1. apple
2. orange
3. peach
4. pineapple
~~~

# Install
~~~
pip3 install theano tensorflow keras
pip3 install h5py
~~~

# Data
1. Download and export URLs of images of each fruit type from ImageNet as [text files](data/download/url/)
2. Run [00_dl_images.py](00_dl_images.py) to download the images
3. Run [01_trans_images.py](01_trans_images.py) 
     * x_train = Resize (128 x 128), Flip and/or Rotate
     * x_test = Resize (128 x 128)
4. Perform analysis on various models

# Results
* [Click here](results/README.md)

# Setup Web Server
* [Click here](django/README.md)

# References
* [Ensemble ML](https://machinelearningmastery.com/ensemble-machine-learning-algorithms-python-scikit-learn/)
* [ML on AWS](https://machinelearningmastery.com/develop-evaluate-large-deep-learning-models-keras-amazon-web-services/)

