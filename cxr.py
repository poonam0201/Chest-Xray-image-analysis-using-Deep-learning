# -*- coding: utf-8 -*-
#importing all libraries
import numpy as np #provides a high-performance multidimensional array object, and tools for working with these arrays.
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os#The os module in python provides a way of usin operating system depend functionality
from cv2 import imread, createCLAHE # read and equalize images
from glob import glob
from tqdm import tqdm
from skimage.transform import resize
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, BatchNormalization, AveragePooling2D

#preprocessing the data
sample_labels = pd.read_csv('sample_labels.csv')

#getting the path of all images
path_df = {os.path.basename(x): x for x in glob(os.path.join('images', '*.png'))}

#creating a new column of paths in sample_labels dataframe
sample_labels['path'] = sample_labels['Image Index'].map(path_df.get)

#replace no findings with blank
sample_labels['Finding Labels'] = sample_labels['Finding Labels'].map(lambda x: x.replace('No Finding', ''))

#appending all disease column vectors to sample_labels dataframe
from itertools import chain
classes = np.unique(list(chain(*sample_labels['Finding Labels'].map(lambda x: x.split('|')).tolist())))
for i in classes:
    if len(i)>1: # leave out empty labels
        sample_labels[i] = sample_labels['Finding Labels'].map(lambda finding: 1.0 if i in finding else 0)

#creating a disease vector dataframe
attributes=sample_labels.columns.values
desease_vector=sample_labels.loc[:,attributes[12:len(attributes)]]

#resizing the images       
outputdim=(256,256)
def imread_and_normalize(im_path):
    clahe_tool = createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_data = np.mean(imread(im_path), 2).astype(np.uint8)
    img_data = clahe_tool.apply(img_data)
    n_img = (255*resize(img_data, outputdim, mode = 'constant')).clip(0,255).astype(np.uint8)
    return np.expand_dims(n_img, -1)

resized_img_arr = np.zeros((sample_labels.shape[0],)+outputdim+(1,), dtype=np.uint8)
for i, c_path in enumerate(tqdm(sample_labels['path'].values)):
    resized_img_arr[i] = imread_and_normalize(c_path)

#creating training and validation data 
from sklearn.model_selection import train_test_split
(x_train,x_test,y_train,y_test)=train_test_split(resized_img_arr,desease_vector,test_size=0.2)


from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
#creating the training model
model = Sequential()


#model.add(ZeroPadding2D((1, 1), input_shape=(256, 256,1), dim_ordering='tf'))
model.add(Convolution2D(4, 1, 1, activation='softmax',input_shape=(256, 256,1)))
	#model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
	model.add(Convolution2D(4, 1, 1, activation='softmax'))
#odel.add(Conv2D(32, (3, 3), activation='sigmoid')
model.add(MaxPooling2D((2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))




model.add(Convolution2D(4, 1, 1, activation='softmax'))
	#model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
	model.add(Convolution2D(4, 1, 1, activation='softmax'))
#odel.add(Conv2D(32, (3, 3), activation='sigmoid')
model.add(MaxPooling2D((2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Flatten())

#model.add(Dense(64))
model.add(Dense(desease_vector.shape[1], activation = 'softmax'))
model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['acc'])
model.summary()

#fit the training and validation data to the model
result=model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=5,verbose=True,shuffle='batch')
print(result.history.keys())
print(result.history['val_loss'])



predictions = model.predict(x_test, verbose=2, batch_size=100)
print(predictions)



predictions[predictions >=0.9] = 1
predictions[predictions <0.9] = 0


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

Accuracy_Score=accuracy_score(y_test,predictions)
print('Average Accuracy:%0.2f +/- (%0.1f) %%' % (Accuracy_Score.mean()*100, Accuracy_Score.std()*100))

import matplotlib.pyplot as plt
plt.plot(result.history['acc'])
plt.plot(result.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(result.history['loss'])
plt.plot(result.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



