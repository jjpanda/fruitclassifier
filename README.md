# Image classification of fruits

## To build an image classification system to classify 4 kinds of fruits: 
~~~
1. apple
2. orange
3. peach
4. pineapple
~~~

# Install
~~~
pip3 install theano tensorflow keras
~~~

# Data
~~~
1. Download and export URLs of images of each fruit type from ImageNet as txt
2. Run 01_dl_images.py to download the images
3. Run 02_trans_images.py to resize (128 x 128) and augment the image 
     3.1 x_train = Resize, Flip and/or Rotate
     3.2 x_test = Resize
4. Perform analysis on various models
~~~

# Results

## Principal component analysis 
* [pca_components.py](pca_components.py)

A vital part of using PCA in practice is the ability to estimate how many components are needed to describe the data. We can determine this by looking at the cumulative explained variance ratio as a function of the number of components.
![pca](results/pca/pca.png)

## k-Nearest Neighbors
* [k-nn.py](k-nn.py)
~~~
             precision    recall  f1-score   support

          0       0.47      0.43      0.45       576
          1       0.65      0.57      0.61       829
          2       0.44      0.40      0.42       683
          3       0.50      0.67      0.58       646

avg / total       0.52      0.52      0.52      2734

accuracy_score: 0.522311631309
~~~
![Confusion Matrix](results/knn/knn_confusion.png)

## Gaussian Naive Bayes
* [gnb.py](gnb.py)
~~~
             precision    recall  f1-score   support

          0       0.34      0.38      0.36       576
          1       0.60      0.49      0.54       829
          2       0.36      0.21      0.26       683
          3       0.40      0.63      0.49       646

avg / total       0.44      0.43      0.42      2734

accuracy_score: 0.430504754938
~~~
![Confusion Matrix](results/gnb/gnb_confusion.png)

## Convolutional Neural Netwnork (CNN) using Keras
* [cnn.py](cnn.py)

### Model Summary
~~~
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_3 (Conv2D)            (None, 124, 124, 32)      2432
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 62, 62, 32)        0
_________________________________________________________________
dropout_4 (Dropout)          (None, 62, 62, 32)        0
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 58, 58, 64)        51264
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 29, 29, 64)        0
_________________________________________________________________
dropout_5 (Dropout)          (None, 29, 29, 64)        0
_________________________________________________________________
flatten_2 (Flatten)          (None, 53824)             0
_________________________________________________________________
dense_3 (Dense)              (None, 128)               6889600
_________________________________________________________________
dropout_6 (Dropout)          (None, 128)               0
_________________________________________________________________
dense_4 (Dense)              (None, 4)                 516
=================================================================
Total params: 6,943,812
Trainable params: 6,943,812
Non-trainable params: 0
~~~

~~~
>>> #Use categorical_crossentropy loss model to train the model
... #SGD - Stochastic gradient descent optimizer.
... model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(), metrics=['accuracy'])
>>>
>>> #epochs - how many times you go through the training set
... run = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
Traceback (most recent call last):
  File "<stdin>", line 2, in <module>
  File "C:\python36\lib\site-packages\keras\models.py", line 867, in fit
    initial_epoch=initial_epoch)
  File "C:\python36\lib\site-packages\keras\engine\training.py", line 1522, in fit
    batch_size=batch_size)
  File "C:\python36\lib\site-packages\keras\engine\training.py", line 1390, in _standardize_user_data
    _check_array_lengths(x, y, sample_weights)
  File "C:\python36\lib\site-packages\keras\engine\training.py", line 241, in _check_array_lengths
    'and ' + str(list(set_y)[0]) + ' target samples.')
ValueError: Input arrays should have the same number of samples as target arrays. Found 24606 input samples and 98424 target samples.
>>> score = model.evaluate(x_test, y_test, verbose=0)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "C:\python36\lib\site-packages\keras\models.py", line 896, in evaluate
    sample_weight=sample_weight)
  File "C:\python36\lib\site-packages\keras\engine\training.py", line 1646, in evaluate
    batch_size=batch_size)
  File "C:\python36\lib\site-packages\keras\engine\training.py", line 1390, in _standardize_user_data
    _check_array_lengths(x, y, sample_weights)
  File "C:\python36\lib\site-packages\keras\engine\training.py", line 241, in _check_array_lengths
    'and ' + str(list(set_y)[0]) + ' target samples.')
ValueError: Input arrays should have the same number of samples as target arrays. Found 2734 input samples and 10936 target samples.
>>> model.save('model_k_seq.h5')
>>>
>>> print('Test loss:', score[0])
Test loss: 0.506276124706
>>> print('Test accuracy:', score[1])
Test accuracy: 0.850402340892
>>>
>>> y_pred = model.predict(x_test)
>>>
>>> print(classification_report(y_test, y_pred))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'classification_report' is not defined
>>> print('accuracy_score:', accuracy_score(y_test, y_pred))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'accuracy_score' is not defined
>>>
>>> mat = confusion_matrix(y_test, y_pred)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'confusion_matrix' is not defined
>>> sns_plot = sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'sns' is not defined
>>> plt.xlabel('true label')
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'plt' is not defined
>>> plt.ylabel('predicted label')
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'plt' is not defined
>>> plt.savefig('cnn_confusion.png')
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'plt' is not defined
>>> plt.show()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'plt' is not defined
>>>
~~~
