# Results

## 1. Principal component analysis 
* [02_pca.py](../02_pca.py)

A vital part of using PCA in practice is the ability to estimate how many components are needed to describe the data. We can determine this by looking at the cumulative explained variance ratio as a function of the number of components.
![pca](pca/pca.png)

## 2. Gaussian Naive Bayes
* [03_gnb.py](../03_gnb.py)
* Saved Model: [gnb_classifier_09112017.pkl](gnb/gnb_classifier_09112017.pkl)
~~~
             precision    recall  f1-score   support

          0       0.34      0.38      0.36       576
          1       0.60      0.49      0.54       829
          2       0.36      0.21      0.26       683
          3       0.40      0.63      0.49       646

avg / total       0.44      0.43      0.42      2734

Accuracy score:  0.430504754938
Precision score:  0.438619628064
Recall score:  0.430504754938
F1 score:  0.421662646121
~~~
### Confusion Matrix
![Confusion Matrix](gnb/gnb_confusion.png)

## 3. Convolutional Neural Netwnork (CNN) using Keras
* [04_cnn.py](04_cnn.py)
* Saved Model: [04_CNN.h5](cnn/04_CNN.h5)
### Model Summary
~~~
24606 train samples
2734 test samples
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 126, 126, 32)      896
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 63, 63, 32)        0
_________________________________________________________________
dropout_1 (Dropout)          (None, 63, 63, 32)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 61, 61, 32)        9248
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 30, 30, 32)        0
_________________________________________________________________
dropout_2 (Dropout)          (None, 30, 30, 32)        0
_________________________________________________________________
flatten_1 (Flatten)          (None, 28800)             0
_________________________________________________________________
dense_1 (Dense)              (None, 64)                1843264
_________________________________________________________________
dropout_3 (Dropout)          (None, 64)                0
_________________________________________________________________
dense_2 (Dense)              (None, 4)                 260
=================================================================
Total params: 1,853,668
Trainable params: 1,853,668
Non-trainable params: 0
~~~

~~~
>>> print(y_test)
[0 0 0 ..., 3 3 3]
>>> print(y_pred)
[0 2 1 ..., 3 3 3]

             precision    recall  f1-score   support

          0       0.54      0.71      0.61       576
          1       0.82      0.81      0.81       829
          2       0.78      0.39      0.51       683
          3       0.75      0.94      0.83       646

avg / total       0.73      0.71      0.70      2734

Accuracy score:  0.712874908559
Precision score:  0.731028282706
Recall score:  0.712874908559
F1 score:  0.700299616223
~~~
### Confusion Matrix
![Confusion Matrix](cnn/cnn_confusion.png)

## 4. k-Nearest Neighbors
* [05_knn.py](../05_knn.py)
* Saved Model: [knn_classifier_09112017.pkl](knn/knn_classifier_09112017.pkl)
~~~
best params:  {'n_neighbors': 6, 'weights': 'distance'}
Results for knn
             precision    recall  f1-score   support

          0       0.45      0.44      0.44       576
          1       0.65      0.57      0.61       829
          2       0.45      0.42      0.44       683
          3       0.52      0.65      0.58       646

avg / total       0.53      0.53      0.52      2734

Accuracy score:  0.525237746891
Precision score:  0.528060114425
Recall score:  0.525237746891
F1 score:  0.52432058499
~~~
### Confusion Matrix
![Confusion Matrix](knn/knn_confusion.png)

## 5. Boosting Algorithms
* [06_boosting.py](06_boosting.py)

### AdaBoost
* Saved Model: [adaboost_classifier_09112017.pkl](boosting/adaboost_classifier_09112017.pkl)
~~~
             precision    recall  f1-score   support

          0       0.35      0.25      0.29       576
          1       0.66      0.47      0.55       829
          2       0.33      0.38      0.36       683
          3       0.37      0.54      0.44       646

avg / total       0.44      0.42      0.42      2734

Accuracy score:  0.416971470373
Precision score:  0.443021826763
Recall score:  0.416971470373
F1 score:  0.418820044314
~~~
![Confusion Matrix](boosting/adaboost_confusion.png)

### Stochastic Gradient Boosting
* Saved Model: [gboost_classifier_09112017.pkl](boosting/gboost_classifier_09112017.pkl)
~~~
             precision    recall  f1-score   support

          0       0.42      0.21      0.28       576
          1       0.62      0.53      0.57       829
          2       0.36      0.32      0.34       683
          3       0.41      0.71      0.52       646

avg / total       0.46      0.45      0.44      2734

Accuracy score:  0.452084857352
Precision score:  0.46133604222
Recall score:  0.452084857352
F1 score:  0.438013617315
~~~
![Confusion Matrix](boosting/gboost_confusion.png)

## 6. Bagging Algorithms
* [07_bagging.py](07_bagging.py)

### Bagged Decision Trees
* Saved Model: [bagging_classifier_10112017.pkl](bagging/bagging_classifier_10112017.pkl)
~~~
             precision    recall  f1-score   support

          0       0.45      0.54      0.49       576
          1       0.62      0.62      0.62       829
          2       0.47      0.39      0.43       683
          3       0.54      0.54      0.54       646

avg / total       0.53      0.53      0.52      2734

Accuracy score:  0.525969275786
Precision score:  0.526831611224
Recall score:  0.525969275786
F1 score:  0.524386794788

Confusion matrix:  
[[309  84  87  96]
 [132 515  98  84]
 [125 173 264 121]
 [127  62 107 350]]
~~~
### Confusion Matrix
![Confusion Matrix](bagging/bagging_confusion.png)

### Random Forest
* Saved Model: [rforest_classifier_10112017.pkl](bagging/rforest_classifier_10112017.pkl)
~~~
             precision    recall  f1-score   support

          0       0.55      0.60      0.57       576
          1       0.72      0.67      0.69       829
          2       0.58      0.51      0.54       683
          3       0.61      0.69      0.65       646

avg / total       0.62      0.62      0.62      2734

Accuracy score:  0.618873445501
Precision score:  0.621300781095
Recall score:  0.618873445501
F1 score:  0.618506438507

Confusion matrix:  
[[344  74  61  97]
 [ 99 553 107  70]
 [104 112 349 118]
 [ 80  33  87 446]]
~~~
### Confusion Matrix
![Confusion Matrix](bagging/rforest_confusion.png)

### Extra Trees
* Saved Model: [extree_classifier_10112017.pkl](bagging/extree_classifier_10112017.pkl)
~~~
             precision    recall  f1-score   support

          0       0.52      0.57      0.54       576
          1       0.71      0.65      0.68       829
          2       0.59      0.51      0.55       683
          3       0.58      0.68      0.63       646

avg / total       0.61      0.60      0.60      2734

Accuracy score:  0.604608632041
Precision score:  0.609209335073
Recall score:  0.604608632041
F1 score:  0.604660692954

Confusion matrix:  
[[328  73  61 114]
 [110 540 101  78]
 [105 109 347 122]
 [ 89  42  77 438]]
~~~
![Confusion Matrix](bagging/extree_confusion.png)
