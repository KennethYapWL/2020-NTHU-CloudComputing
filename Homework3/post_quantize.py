try:
    import tensorflow.compat.v2 as tf
except Exception:
    pass
tf.enable_v2_behavior()

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import time

base_dir = 'cats_and_dogs_small'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

train_datagen = ImageDataGenerator(
                    rescale=1./255,)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(112, 112),
        batch_size=8,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(112, 112),
        batch_size=8,
        class_mode='binary')

# Load model
model_path = './models/VGG16_model.h5'
print('[INFO] Load model from {}'.format(model_path))
try:
    model = tf.keras.models.load_model(model_path)
except:
    raise ValueError('Model not found. Please run vgg16_train.py')
model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.SGD(lr=0.001), metrics=['acc'])


# Convert model
tflite_models_dir = './models'
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
tflite_model_file = os.path.join(tflite_models_dir, 'vgg16_model.tflite')
with open(tflite_model_file, 'wb') as f:
    f.write(tflite_model)

print('[INFO] Start converting quantized model  ')

# Set quantization options
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# TODO: Uncomment to use fp16 precision, default is dynamic range precision
converter.target_spec.supported_types = [tf.float16]

tflite_quant_model = converter.convert()
tflite_model_quant_file = os.path.join(tflite_models_dir, "quantized_vgg16_model.tflite")
with open(tflite_model_quant_file, 'wb') as f:
    f.write(tflite_quant_model)

# Load the model into tflite interpreter
interpreter = tf.lite.Interpreter(model_path=tflite_model_file)
interpreter.allocate_tensors()

interpreter_quant = tf.lite.Interpreter(model_path=tflite_model_quant_file)
interpreter_quant.allocate_tensors()

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
  for (x, y) in test_datagen.flow_from_directory(validation_dir, batch_size=1, target_size=(112, 112)):
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

# Print quantized tflite size
print("Size of the model before quantization: %.2f Mb"
        % (os.path.getsize(tflite_model_file) / float(2**20)))

print("Size of the model after quantization: %.2f Mb"
        % (os.path.getsize(tflite_model_quant_file) / float(2**20)))

# Evaluate valid acc
print('[INFO] Start inference process...')
print('Original model acc: {:f}\n'.format(evaluate_model(interpreter)))
print('Quantized model acc: {:f}\n'.format(evaluate_model(interpreter_quant)))
