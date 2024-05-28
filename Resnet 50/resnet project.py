# %% [markdown]
# # Indian Landmark Detection Using VGG-19

# %% [markdown]
# # Importing libraries

# %%
import os
# import cv2
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
    for i in range(1):
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
# train = imagearray(train_path, size)


# %%
# test = imagearray(test_path, size)

# %%


# %% [markdown]
# # Normalization

# %%
# x_train = np.array(train)
# x_test = np.array(test)

# x_train.shape,x_test.shape

# %%
# x_train = x_train/255.0
# x_test = x_test/255.0


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
size = (250,250)

train_class = data_class(train_path, size, 'sparse')
test_class = data_class(test_path, size, 'sparse')

# %%
y_train = train_class.classes
y_test = test_class.classes
train_d=train_class
test_d=test_class
print(train_class)

# %%
train_class.class_indices

# %%
# y_train.shape,y_test.shape

# %%
import tensorflow as tf
img_height,img_width=180,180
batch_size=32
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  train_path,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# %%
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  train_path,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

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

model = Sequential([tf.keras.layers.BatchNormalization()])

pretrained_model= tf.keras.applications.ResNet50V2(include_top=False,
                   input_shape=(180,180,3),
                   pooling='avg',classes=24,
                   weights='imagenet')
for layer in pretrained_model.layers:
        layer.trainable=False
# model.add(tf.keras.layers.Conv2D(filters=180,kernel_size=3,activation="relu",input_shape=[180,180,3]))
model.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,activation="relu",input_shape=[178,178,3]))
# model.add(tf.keras.layers.MaxPool2D(strides=2,pool_size=2))
model.add(pretrained_model)
model.add(Flatten())
model.add(Dense(units=512, activation='relu'))
model.add(Dense(units=24, activation='softmax'))
# model.add(Flatten())
     

# %%
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])


# %%
# early_stop = EarlyStopping(monitor = 'val_loss', mode='min', verbose = 1, patience = 5)

# %%
# ! pip install tensorflow_hub

# %%
# import tensorflow as tf
# import tensorflow_hub as hub
# from tensorflow.keras import layers

# %%

# def create_model(model_url, num_classes=10):
#   """Takes a TensorFlow Hub URL and creates a Keras Sequential model with it.
  
#   Args:
#     model_url (str): A TensorFlow Hub feature extraction URL.
#     num_classes (int): Number of output neurons in output layer,
#       should be equal to number of target classes, default 10.

#   Returns:
#     An uncompiled Keras Sequential model with model_url as feature
#     extractor layer and Dense output layer with num_classes outputs.
#   """
#   # Download the pretrained model and save it as a Keras layer
#   feature_extractor_layer = hub.KerasLayer(model_url,
#                                            trainable=False, # freeze the underlying patterns
#                                            name='feature_extraction_layer',
#                                            input_shape=IMAGE_SHAPE+(3,)) # define the input image shape
  
#   # Create our own model
#   model = tf.keras.Sequential([
#     feature_extractor_layer, # use the feature extraction layer as the base
#     Dense(num_classes, activation='softmax', name='output_layer') # create our own output layer      
#   ])

#   return model

# %%
# resnet_url = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4"

# %%
# ! pip install tensorflow

# %%
# IMAGE_SHAPE = (224, 224)
# BATCH_SIZE = 32
# resnet_model = create_model(resnet_url, num_classes=24)

# # Compile
# resnet_model.compile(loss='categorical_crossentropy',
#                      optimizer=tf.keras.optimizers.Adam(),
#                      metrics=['accuracy'])

# %% [markdown]
# history = model.fit(x_train,y_train, validation_data = (x_test,y_test), epochs = 10, callbacks=[early_stop], batch_size = 3                    shuffle=True)

# %%
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model

# %%
import tensorflow as tf
base_model = ResNet50(include_top=False, weights='imagenet')
x= base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x=tf.keras.layers.Dense(1024, activation= "relu" )(x)
predictions = tf.keras.layers.Dense (24, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers:
    layer.trainable = False
model. compile(optimizer='adam', loss='categorical_crossentropy', metrics = ["accuracy"])

# print(target)

# %%
epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)
# history = model.fit(
#   y_train,
#   validation_data=y_test,
#   epochs=10
# )
   

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


