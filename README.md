# Image classification of fruits

### To build an image classification system to classify 4 kinds of fruits: 
~~~
1. apple
2. orange
3. peach
4. pineapple
~~~

#### Install
~~~
pip3 install theano tensorflow keras
~~~

#### Data
~~~
1. Download and export URLs of images of each fruit type from ImageNet as txt
2. Run 01_dl_images.py to download the images
3. Run 02_trans_images.py to resize (128 x 128) and augment the image 
     3.1 x_train = Resize, Flip and/or Rotate
     3.2 x_test = Resize
4. Perform analysis on various models
~~~

#### Results

##### Principal component analysis 
A vital part of using PCA in practice is the ability to estimate how many components are needed to describe the data. We can determine this by looking at the cumulative explained variance ratio as a function of the number of components.
![pca](pca.png)

##### k-Nearest Neighbors
~~~
             precision    recall  f1-score   support

          0       0.47      0.43      0.45       576
          1       0.65      0.57      0.61       829
          2       0.44      0.40      0.42       683
          3       0.50      0.67      0.58       646

avg / total       0.52      0.52      0.52      2734

accuracy_score: 0.522311631309
~~~
![Confusion Matrix](knn_confusion.png)

##### Gaussian Naive Bayes
~~~
             precision    recall  f1-score   support

          0       0.34      0.38      0.36       576
          1       0.60      0.49      0.54       829
          2       0.36      0.21      0.26       683
          3       0.40      0.63      0.49       646

avg / total       0.44      0.43      0.42      2734

accuracy_score: 0.430504754938
~~~
![Confusion Matrix](gnb_confusion.png)
