# %%
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# %% [markdown]
# # Preparing The Data

# %%


# %%
# print(data_dir)

# %%
# roses = list(data_dir.glob('roses/*'))
# print(roses[0])
# PIL.Image.open(str(roses[0]))

# %%
train_path = r'archive - Copy\Indian-monuments\images\train'
test_path = r'archive - Copy\Indian-monuments\images\test'

# %%
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

# %%
class_names = train_ds.class_names
print(class_names)

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(24, 24))
for images, labels in train_ds.take(1):
  for i in range(6):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

# %%


# %% [markdown]
# # Training The Model
# 

# %%
resnet_model = Sequential()
resnet_model.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,activation="relu"))          


pretrained_model= tf.keras.applications.ResNet50(include_top=False,
                   input_shape=(180,180,3),
                   pooling='avg',classes=5,
                   weights='imagenet')
for layer in pretrained_model.layers:
        layer.trainable=False

resnet_model.add(pretrained_model)
resnet_model.add(tf.keras.layers.Flatten())
resnet_model.add(tf.keras.layers.Dense(512, activation='relu'))
resnet_model.add(tf.keras.layers.Dense(5, activation='softmax'))

# %%
resnet_model.summary()

# %%
resnet_model.compile(optimizer="Adam",loss='categorical_crossentropy',metrics=['accuracy'])

# %%
epochs=10
history = resnet_model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# %% [markdown]
# # Evaluating The Model

# %%
fig1 = plt.gcf()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.axis(ymin=0.4,ymax=1)
plt.grid()
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'])
plt.show()

# %%
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.grid()
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'])
plt.show()

# %% [markdown]
# # Making Predictions

# %%
import cv2
image=cv2.imread(str("archive - Copy\Indian-monuments\images/test/alai_darwaza\img180.jpg"))
image_resized= cv2.resize(image, (img_height,img_width))
image=np.expand_dims(image_resized,axis=0)
print(image.shape)


# %%
pred=resnet_model.predict(image)
print(pred)

# %%
output_class=class_names[np.argmax(pred)]
print("The predicted class is", output_class)

# %%
resnet_model.save("resnet.h5")

# %%



