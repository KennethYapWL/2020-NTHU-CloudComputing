import os
import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow_model_optimization.sparsity import keras as sparsity
from tensorflow_model_optimization.python.core.sparsity.keras import prune
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule

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

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(96, 96),
        batch_size=8,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(96, 96),
        batch_size=8,
        class_mode='binary')

# Load model
model_path = './models/MobileNetv2_model.h5'
print('[INFO] Load model from {}'.format(model_path))
try:
    model = tf.keras.models.load_model(model_path)
except:
    raise ValueError('Model not found. Please run mobilenetv2_train.py')
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.SGD(lr=0.001), metrics=['acc'])


zip_path = './models/MobileNetv2_model.zip'
with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(model_path)

# Pruning setup
# Hint: modify the epochs to let model recover
epochs = 10
end_step = train_generator.__len__() * epochs

# Define pruning paramaters
# Hint1: pruned model needs steps to recover
# Hint2: initial sparsity too large will lead to low acc
# TODO Compare result with final sparsity 0.25, 0.5, 0.75
pruning_params = {
        'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.10,
                                                   final_sparsity=0.75,
                                                   begin_step=0,
                                                   end_step=end_step,
                                                   frequency=200)
        }

# Assign pruning paramaters
pruned_model = sparsity.prune_low_magnitude(model, **pruning_params)

# Print the converted model
pruned_model.summary()

pruned_model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.SGD(lr=0.001), metrics=['acc'])

callbacks = [
    sparsity.UpdatePruningStep(),
    sparsity.PruningSummaries(log_dir='./', profile_batch=0)
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

pruned_model_path = './models/pruned_MobileNetv2.h5'
# convert pruned model to original
final_model = sparsity.strip_pruning(pruned_model)
tf.keras.models.save_model(final_model, pruned_model_path, include_optimizer=False)

# Zip file
pruned_zip_path = './models/pruned_MobileNetv2.zip'
with zipfile.ZipFile(pruned_zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(pruned_model_path)

# Print file size
print("Size of the model before compression: %.2f Mb"
        % (os.path.getsize(model_path) / float(2**20)))

print("Size of the model after compression: %.2f Mb"
        % (os.path.getsize(zip_path) / float(2**20)))

print("Size of the pruned model before compression: %.2f Mb"
        % (os.path.getsize(pruned_model_path) / float(2**20)))

print("Size of the pruned model after compression: %.2f Mb"
        % (os.path.getsize(pruned_zip_path) / float(2**20)))

# Evaluate valid acc
model_out = model.evaluate(validation_generator, steps=validation_generator.__len__(), verbose=0)
pruned_out = pruned_model.evaluate(validation_generator, steps=validation_generator.__len__(), verbose=0)
print('[INFO] model val_acc: {}'.format(model_out[1]))
print('[INFO] pruend model val_acc: {}'.format(pruned_out[1]))
