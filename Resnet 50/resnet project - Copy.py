# %% [markdown]
# # Indian Landmark Detection Using VGG-19

# %% [markdown]
# # Importing libraries

# %%
! pip install pandas
! pip install numpy
! pip install opencv-python
! pip install seaborn
! pip install tensorflow
! pip install scikit-learn
! pip install pickle
! pip install os

# %%
import os
import cv2
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import keras
import tensorflow

from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG19
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Dropout, BatchNormalization, Activation

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, precision_score, f1_score 

# %% [markdown]
# # Defining data paths

# %%
train_path = r'archive - Copy\Indian-monuments\images\train'
test_path = r'archive - Copy\Indian-monuments\images\test'

# %% [markdown]
# # Converting image to pixels

# %%
for folder in os.listdir(train_path):
    sub_path = train_path + "/" + folder
    
    print(folder)
    for i in range(2):
        temp_path = os.listdir(sub_path)[i]
        temp_path = sub_path + "/" + temp_path
        img = mpimg.imread(temp_path)
        imgplot = plt.imshow(img)
        plt.show()

# %%
def imagearray(path, size):
    data = []
    for folder in os.listdir(path):
        sub_path=path+"/"+folder

        for img in os.listdir(sub_path):
            image_path=sub_path+"/"+img
            img_arr=cv2.imread(image_path)
            img_arr=cv2.resize(img_arr,size)
            
            data.append(img_arr)

    return data

# %%
size = (250,250)
train = imagearray(train_path, size)


# %%
test = imagearray(test_path, size)

# %% [markdown]
# # Normalization

# %%
x_train = np.array(train)
x_test = np.array(test)

x_train.shape,x_test.shape

# %%
x_train = x_train/255.0
x_test = x_test/255.0


# %% [markdown]
# # Defining target variables

# %%
def data_class(data_path, size, class_mode):
    datagen = ImageDataGenerator(rescale = 1./255)
    classes = datagen.flow_from_directory(data_path,
                                          target_size = size,
                                          batch_size = 32,
                                          class_mode = class_mode)
    return classes

# %%
train_class = data_class(train_path, size, 'sparse')
test_class = data_class(test_path, size, 'sparse')

# %%
y_train = train_class.classes
y_test = test_class.classes
print(train_class)

# %%
train_class.class_indices

# %%
y_train.shape,y_test.shape,x_train.shape,x_test.shape
# y_val.shape

# %% [markdown]
# # VGG19 Model

# %%
# vgg = VGG19(input_shape = (250, 250, 3), weights = 'imagenet', include_top = False)

# %%
# for layer in vgg.layers:
#     layer.trainable = False

# x = Flatten()(vgg.output)
# prediction = Dense(24, activation='softmax')(x)

# model = Model(inputs=vgg.input, outputs=prediction)
# model.summary()
# model.compile(
#   loss='sparse_categorical_crossentropy',
#   optimizer="adam",
#   metrics=['accuracy']
# )

# %%
import tensorflow as tf
model = Sequential()

pretrained_model= tf.keras.applications.ResNet152(include_top=False,
                   input_shape=(250,250,3),
                   pooling='avg',classes=24,
                   weights='imagenet')
for layer in pretrained_model.layers:
        layer.trainable=False

model.add(pretrained_model)
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(24, activation='softmax'))

# %%
early_stop = EarlyStopping(monitor = 'val_loss', mode='min', verbose = 1, patience = 5)

# %%
history = model.fit(x_train,y_train, validation_data = (x_test,y_test), epochs = 10, callbacks=[early_stop], batch_size = 30,
                    shuffle=True)

# %%
model.save("resnet.h5")


# %% [markdown]
# # Visualization

# %%
plt.figure(figsize=(10, 8))
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.title('Accuracy')
plt.show()

# %%
plt.figure(figsize=(10, 8))
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.title('Loss')
plt.show()

# %% [markdown]
# # Model Evaluation

# %%
model.evaluate(x_test, y_test, batch_size=32)

# %%
y_pred = model.predict(x_test)

# %%
y_pred=np.argmax(y_pred,axis=1)

# %%
print(classification_report(y_pred,y_test))


# %% [markdown]
# # Confusion Matrix

# %%
cm = confusion_matrix(y_pred,y_test)

plt.figure(figsize=(10, 8))
ax = plt.subplot()
sns.set(font_scale=2.0)
sns.heatmap(cm, annot=True, fmt='g', cmap="Blues", ax=ax); 

# labels, title and ticks
ax.set_xlabel('Predicted labels', fontsize=20);
ax.set_ylabel('True labels', fontsize=20); 
ax.set_title('Confusion Matrix', fontsize=20); 


# %%
f1_score(y_test, y_pred, average='micro')

# %%
recall_score(y_test, y_pred, average='weighted')

# %%
precision_score(y_test, y_pred, average='micro')

# %% [markdown]
# # Saving Model

# %%
model.save("resent152(1).h5")


