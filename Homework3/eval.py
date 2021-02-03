import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import time

base_dir = 'cats_and_dogs_small'
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')

test_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = test_datagen.flow_from_directory(
                            validation_dir,
                            target_size=(112, 112),
                            batch_size=8,
                            class_mode='binary')
'''
test_generator = test_datagen.flow_from_directory(
                            test_dir,
                            target_size=(112, 112),
                            batch_size=8,
                            class_mode='binary')
'''
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


# Load tflite file
tflite_model_file = 'models/vgg16_model.tflite'
interpreter = tf.lite.Interpreter(model_path=tflite_model_file)
interpreter.allocate_tensors()

print('[INFO] Start inference process...')
print('model acc: {}\n'.format(evaluate_model(interpreter)))

