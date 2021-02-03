#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau

import matplotlib.pyplot as plt


# In[2]:


tf.__version__


# In[3]:


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the fourth GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


# In[4]:


base_dir = 'cats_and_dogs_small'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')

IMG_WIDTH = 180
IMG_HEIGHT = 180
BATCH_SIZE = 64
EPOCHS= 100


# In[5]:


print('total training cat images:', len(os.listdir(train_cats_dir)))
print('total training dog images:', len(os.listdir(train_dogs_dir)))


# In[6]:


#generate training data with augmentation
train_datagen = ImageDataGenerator( rotation_range=15,
                                    rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    #vertical_flip=True,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    #channel_shift_range=0.1,
                                    #brightness_range=[0.1, 10]
                                  )

valid_datagen = ImageDataGenerator(rescale=1./255)


# In[7]:


train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='binary')

validation_generator = valid_datagen.flow_from_directory(
        validation_dir,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='binary')


# In[8]:


train_generator.__len__()


# In[9]:


def build_model(name):
    inputs = keras.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    x = layers.Conv2D(64, (5, 5), padding='same', activation='relu')(inputs)
    x = layers.MaxPool2D(pool_size=3,strides=2,padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(64, (5, 5), padding='same', activation='relu')(x)
    x = layers.MaxPool2D(pool_size=3,strides=2,padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(64, (5, 5), padding='same', activation='relu')(x)
    x = layers.MaxPool2D(pool_size=3,strides=2,padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(384, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(192, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs, outputs, name=name)
    model.summary()
    
    return model


# In[10]:


# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_binary_accuracy', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)


# In[11]:


optimizer = RMSprop(lr=0.001, rho=0.95, epsilon=1e-08, decay=0.0)
#optimizer = keras.optimizers.Adam()


# In[12]:


model_dir = 'lab_logs_datagen/models0528/'
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)


# In[13]:


model_mckp = keras.callbacks.ModelCheckpoint(model_dir + '/checkpoint.h5',
                                            monitor='val_binary_accuracy',
                                            save_best_only=True,
                                            mode='max')


# In[14]:


model = build_model('model')
model.compile( optimizer = optimizer,
               loss=keras.losses.BinaryCrossentropy(),
               metrics=[keras.metrics.BinaryAccuracy()]
             )


# In[15]:


history = model.fit_generator(
                              train_generator,
                              steps_per_epoch=BATCH_SIZE,
                              callbacks=[learning_rate_reduction, model_mckp],
                              epochs=EPOCHS,
                              validation_data=validation_generator,
                              validation_steps=BATCH_SIZE)


# In[16]:


plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')
plt.show()


# In[17]:


plt.figure(figsize=(8,4))
plt.plot(history.history['binary_accuracy'], label='train')
plt.plot(history.history['val_binary_accuracy'], label='validation')
plt.ylabel('metrics')
plt.xlabel('epochs')
plt.legend(loc='lower right')
plt.show()


# In[ ]:




