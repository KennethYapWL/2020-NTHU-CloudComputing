import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

base_dir = 'cats_and_dogs_small'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

def create_model():

    base =  tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(112, 112, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    output =  tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=base.input, outputs = output)
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.SGD(lr=0.001), metrics=['acc'])
    return model

train_datagen = ImageDataGenerator(rescale=1./255)
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

# Create model
model = create_model()


print('[INFO] Start training process...')

model.fit(
      train_generator,
      steps_per_epoch=train_generator.__len__(),
      epochs=5,
      validation_data=validation_generator,
      validation_steps=validation_generator.__len__()
)

model_path = './models/VGG16_model.h5'

print('[INFO] Save model to {}'.format(model_path))
tf.keras.models.save_model(model, model_path, include_optimizer=False)

