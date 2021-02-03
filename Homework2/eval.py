import os
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

checkpoint = 'VGG16_model.h5'

base_dir = 'cats_and_dogs_small' # Would be absolute path to dataset when scoring.

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')
realtest_dir = os.path.join(base_dir, 'realtest')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')

realtest_cats_dir = os.path.join(realtest_dir, 'cats')
realtest_dogs_dir = os.path.join(realtest_dir, 'dogs')

model = models.load_model(checkpoint)
print(model.summary())

input_shape = (model.input.shape[1], model.input.shape[2])
print('input shape: ' + str(input_shape))
output_shape = model.output.shape[1]
print('output shape: %d' % output_shape)

class_mode = 'binary' if output_shape == 1 else 'categorical'

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
            #realtest_dir, # Would be test set.
            #test_dir,
            validation_dir,
            target_size=input_shape,
            batch_size=32,
            class_mode=class_mode)

score = model.evaluate(test_generator)

print('test loss: %f, test acc: %f' % (score[0], score[1]))










