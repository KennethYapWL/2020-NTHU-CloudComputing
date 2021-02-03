#!/usr/bin/env python
# coding: utf-8

# #### Imports

# In[13]:


import numpy as np
import os
import zipfile
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow_model_optimization.sparsity import keras as sparsity
from tensorflow_model_optimization.python.core.sparsity.keras import prune
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau


# In[14]:


try:
    import tensorflow.compat.v2 as tf
except Exception:
    pass
tf.enable_v2_behavior()


# #### Call GPU

# In[15]:


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


# #### Initial Settings

# In[16]:


base_dir = 'cats_and_dogs_small'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

EPOCHS = 15
WIDTH = 150
HEIGHT = 150
OPTIMIZERS = keras.optimizers.SGD(lr = 0.01)


# #### Image Data Generator

# In[17]:


train_datagen = ImageDataGenerator(
                    rescale=1./255,)
test_datagen = ImageDataGenerator(rescale=1./255)


# In[18]:


train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(WIDTH, HEIGHT),
        batch_size=8,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(WIDTH, HEIGHT),
        batch_size=8,
        class_mode='binary')


# #### Build VGG16 model

# In[19]:


# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)


# In[20]:


def create_model():
    base =  tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(WIDTH, HEIGHT, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    output =  tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=base.input, outputs = output)
    model.compile(loss='binary_crossentropy', optimizer=OPTIMIZERS, metrics=['acc'])
    model.summary()
    return model


# In[9]:


# Create model
model = create_model()
print('[INFO] Start training process...')

model.fit(
      train_generator,
      steps_per_epoch=train_generator.__len__(),
      epochs=EPOCHS,
      validation_data=validation_generator,
      validation_steps=validation_generator.__len__(),
      callbacks=[learning_rate_reduction]
)

model_path = './models/VGG16_model.h5'

print('[INFO] Save model to {}'.format(model_path))
tf.keras.models.save_model(model, model_path, include_optimizer=False)


# ---
# ### Pruning
# #### Load the model (unnecessary, but for safe) and do pruning

# In[21]:


# Load model
model_path = './models/VGG16_model.h5'
print('[INFO] Load model from {}'.format(model_path))
try:
    model = tf.keras.models.load_model(model_path)
except:
    raise ValueError('Model not found. Please run vgg16_train.py')
    
# Create model
model.compile(loss='binary_crossentropy', optimizer=OPTIMIZERS, metrics=['acc'])


# In[22]:


# Pruning setup
# Hint: modify the epochs to let model recover
epochs = EPOCHS
end_step = train_generator.__len__() * epochs


# In[23]:


# Define pruning paramaters
# Hint1: pruned model needs steps to recover
# Hint2: initial sparsity too large will lead to low acc
# TODO Compare result with final sparsity 0.25, 0.5, 0.75
pruning_params = {
        'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.10,
                                                   final_sparsity=0.90,
                                                   begin_step=0,
                                                   end_step=end_step,
                                                   frequency=200)
        }


# In[24]:


# Assign pruning paramaters
pruned_model = sparsity.prune_low_magnitude(model, **pruning_params)

# Print the converted model
pruned_model.summary()

pruned_model.compile(loss='binary_crossentropy', optimizer=OPTIMIZERS, metrics=['acc'])

callbacks = [
    sparsity.UpdatePruningStep(),
    sparsity.PruningSummaries(log_dir='./', profile_batch=0),
    learning_rate_reduction
]

print('[INFO] Start pruning process...')

pruned_model.fit(
      train_generator,
      steps_per_epoch=train_generator.__len__(),
      callbacks=callbacks,
      epochs=epochs,
      validation_data=validation_generator,
      validation_steps=validation_generator.__len__()
)

pruned_model_path = './models/pruned_VGG16.h5'
# convert pruned model to original
final_model = sparsity.strip_pruning(pruned_model)
tf.keras.models.save_model(final_model, pruned_model_path, include_optimizer=False)


# #### Zip file

# In[25]:


zip_path = './models/VGG16_model.zip'
with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(model_path)

# Zip file
pruned_zip_path = './models/pruned_VGG16.zip'
with zipfile.ZipFile(pruned_zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(pruned_model_path)


# #### Examine Pruning Result

# In[26]:


# Print file size
print("Size of the model before compression: %.2f MB"
        % (os.path.getsize(model_path) / float(2**20)))

print("Size of the model after compression: %.2f MB"
        % (os.path.getsize(zip_path) / float(2**20)))

print("Size of the pruned model before compression: %.2f MB"
        % (os.path.getsize(pruned_model_path) / float(2**20)))

print("Size of the pruned model after compression: %.2f MB"
        % (os.path.getsize(pruned_zip_path) / float(2**20)))

# Evaluate valid acc
model_out = model.evaluate(validation_generator, steps=validation_generator.__len__(), verbose=0)
pruned_out = pruned_model.evaluate(validation_generator, steps=validation_generator.__len__(), verbose=0)
print('[INFO] model val_acc: {}'.format(model_out[1]))
print('[INFO] pruend model val_acc: {}'.format(pruned_out[1]))


# #### Convert model to tflite

# In[41]:


# Convert model
tflite_models_dir = './models'
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
tflite_model_file = os.path.join(tflite_models_dir, 'vgg16_model.tflite')
with open(tflite_model_file, 'wb') as f:
    f.write(tflite_model)


# ---
# ### Quantization

# In[42]:


# Set quantization options
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# TODO: Uncomment to use fp16 precision, default is dynamic range precision
#converter.target_spec.supported_types = [tf.float16]


# #### declare the filename of quantized model tflite (here is set to advance.tflite)

# In[43]:


tflite_quant_model = converter.convert()
tflite_model_quant_file = os.path.join(tflite_models_dir, "advance.tflite")
with open(tflite_model_quant_file, 'wb') as f:
    f.write(tflite_quant_model)


# In[44]:


# Load the model into tflite interpreter
interpreter = tf.lite.Interpreter(model_path=tflite_model_file)
interpreter.allocate_tensors()

interpreter_quant = tf.lite.Interpreter(model_path=tflite_model_quant_file)
interpreter_quant.allocate_tensors()


# In[45]:


# A helper function to evaluate the TF Lite model.
def evaluate_model(interpreter):
  input_index = interpreter.get_input_details()[0]["index"]
  output_index = interpreter.get_output_details()[0]["index"]

  # Run predictions on every image in the "test" dataset.
  prediction_digits = []
  accurate_count = 0
  data_count = 0
  total_valid_images = len(validation_generator.filenames)
  start = time.time()
  for (x, y) in test_datagen.flow_from_directory(validation_dir, batch_size=1, target_size=(WIDTH, HEIGHT)):
    data_count += 1
    
    interpreter.set_tensor(input_index, x)

    # Run inference.
    interpreter.invoke()

    output = interpreter.tensor(output_index)
    category = 1 if output()[0] >= 0.5 else 0
    if y[0][category] == 1:
        accurate_count += 1
    
    if data_count == total_valid_images:
        break

  accuracy = accurate_count * 1.0 / total_valid_images
  end = time.time()
  elapsed = end - start
  print('Performance: {:0.1f} ms/image'.format((elapsed/total_valid_images) * 1000))
  return accuracy


# In[46]:


# Print quantized tflite size
print("Size of the model before quantization: %.2f Mb"
        % (os.path.getsize(tflite_model_file) / float(2**20)))

print("Size of the model after quantization: %.2f Mb"
        % (os.path.getsize(tflite_model_quant_file) / float(2**20)))


# In[ ]:


# Evaluate valid acc
print('[INFO] Start inference process...')
print('Original model acc: {:f}\n'.format(evaluate_model(interpreter)))
print('Quantized model acc: {:f}\n'.format(evaluate_model(interpreter_quant)))


# In[ ]:




