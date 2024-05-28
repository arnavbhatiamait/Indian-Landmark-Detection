# %% [markdown]
# # VGG-19 Model Predictions

# %% [markdown]
# importing the libraries

# %%
import os
import cv2
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pywhatkit as pwt
import pyttsx3
import keras
import tensorflow 
import tensorflow as tf
import wikipedia
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG19
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Dropout, BatchNormalization, Activation
from googlesearch import search
import webbrowser
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, precision_score, f1_score 

# %% [markdown]
# Making Predictions

# %%
new_model = tf.keras.models.load_model('Classification_model_vgg19(21).h5')

# %%
def imagearray(path, size):
    img = mpimg.imread(path)
    imgplot = plt.imshow(img)
    plt.show()
    data = []
    img_arr=cv2.imread(path)
    print(path)
    img_arr=cv2.resize(img_arr,size)        
    data.append(img_arr)
    
    # print(data)
    return data

# %%
size=(250,250)
img_load=imagearray("archive - Copy\Indian-monuments\images/train/victoria memorial/1.jpeg",size)
pred_img=np.array(img_load)

# %%
pred=new_model.predict(pred_img)
pred=np.argmax(pred,axis=1)
print(pred)

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
print(len(dicti_values))
for m in range (24):
    if (pred==m):
        print("The predicted value of the input picture is ",dicti_values[m])

# %%
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voices', voices[0].id)

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

# %% [markdown]
# Making Searches

# %%
def linksearch(querry):
    speak("here related links...")
    for i in search(querry):
        print(i)

# %%
linksearch(dicti_values[int(pred)])

# %% [markdown]
# play video on YouTubeVideo

# %%
def ytsearch(querry):
    speak("here is your video")
    pwt.playonyt(querry)

# %%
ytsearch(dicti_values[int(pred)])

# %%
def opengoogle(querry):
    speak("here are your search results")
    pwt.search(querry)

# %%
opengoogle(dicti_values[int(pred)])


