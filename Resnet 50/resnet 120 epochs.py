# %%
# ! pip install pytorch

# %%
# !pip install split-folders matplotlib opencv-python spicy

# %%
# Import Keras modules and its important APIs
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
# from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
import numpy as np
import os
import torch
import torch.nn as nn
import numpy as np
import cv2 as cv
import os
import splitfolders
import matplotlib.pyplot as plt

# tensorflow
import tensorflow.keras as keras
import tensorflow as tf

# image processing
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

# model / neural network
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input


# %%
train_path="archive - Copy/Indian-monuments/images/train"

# %%
# splitfolders.ratio(train_path, output="archive - Copy/data-split", seed=1337, ratio=(0.7, 0.2, 0.1), group_prefix=None, move=False)


# %%
dicti_values={0:'Ajanta Caves',
 1:'Charar E Sharif',
 2:'Chhota Imambara',
 3:'Ellora Caves',
 4:'Fatehpur Sikri',
 5:'Gateway of India',
 6:'Humayun_s Tomb',
 7:'India gate pics',
 8:'Khajuraho',
 9:'Sun Temple Konark',
 10:'alai darwaza',
 11:'alai minar',
 12:'basilica of bom jesus',
 13:'charminar',
 14:'golden temple',
 15:'hawa mahal pics',
 16:'iron pillar',
 17:'jamali kamali_tomb',
 18:'lotus temple',
 19:'mysore palace',
 20:'qutub minar',
 21:'tajmahal',
 22:'tanjavur temple',
 23:'victoria memorial'}
l=[]
for i in range(23):
    l.append(dicti_values[i])
class_names=l
print(class_names)

# %%
datagen = ImageDataGenerator()

# %%
train_generator = datagen.flow_from_directory( 
    directory="archive - Copy/data-split/train", 
    classes = class_names,
    target_size=(224, 224),  
    batch_size=32, 
    class_mode="binary", 
)
valid_generator = datagen.flow_from_directory( 
    directory="archive - Copy/data-split/val", 
    classes = class_names,
    target_size=(224, 224), 
    batch_size=32, 
    class_mode="binary", 
)

# test data
test_generator = datagen.flow_from_directory( 
    directory="archive - Copy\data-split/test", 
    classes = class_names,
    target_size=(224, 224), 
    batch_size=32, 
    class_mode="binary", 
)

# %%
resnet_50 = ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3))
for layer in resnet_50.layers:
    layer.trainable = False

# %%
x = resnet_50.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512, activation='relu')(x) 
x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation='relu')(x) 
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation='relu')(x) 
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation='relu')(x) 
x = layers.Dropout(0.5)(x)
predictions = layers.Dense(24, activation='softmax')(x)
model = Model(inputs = resnet_50.input, outputs = predictions)

# %%
def trainModel(model, epochs, optimizer):
    batch_size = 32
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model.fit(train_generator, validation_data=valid_generator, epochs=epochs, batch_size=batch_size)

# %%
model_history = trainModel(model = model, epochs = 120, optimizer = "Adam")

# %%
model.save('resnet_model6.h5')

# %%
loss_train_curve = model_history.history["loss"]
loss_val_curve = model_history.history["val_loss"]
plt.plot(loss_train_curve, label = "Train")
plt.plot(loss_val_curve, label = "Validation")
plt.legend(loc = 'upper right')
plt.title("Loss")
plt.show()

# %%
acc_train_curve = model_history.history["accuracy"]
acc_val_curve = model_history.history["val_accuracy"]
plt.plot(acc_train_curve, label = "Train")
plt.plot(acc_val_curve, label = "Validation")
plt.legend(loc = 'lower right')
plt.title("Accuracy")
plt.show()

# %%
test_loss, test_acc = model.evaluate(test_generator)
print("The test loss is: ", test_loss)
print("The best accuracy is: ", test_acc*100)

# %%
model.save('resnet_model7.h5')

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
import cv2
size = (250,250)
train = imagearray(train_path, size)
test = imagearray(test_path, size)

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
y_train.shape,y_test.shape

# %%
y_pred = model.predict(x_test)

# %%
y_pred=np.argmax(y_pred,axis=1)


# %%
print(classification_report(y_pred,y_test))


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



