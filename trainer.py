from keras.layers import Input, Lambda, Dense, Flatten, Dropout
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
import numpy as np
from glob import glob
#import matplotlib.pyplot as plt

# re-size all the images to this
IMAGE_SIZE = [224, 224]

train_path = 'dataset/Train/'
valid_path = 'dataset/Test/'

# add preprocessing layer to the front of VGG
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# don't train existing weights
for layer in vgg.layers:
      layer.trainable = False
  

  
# useful for getting number of classes
folders = glob('dataset/train/*')
  


# our layers - you can add more if you want
x = Flatten()(vgg.output)
x = Dense(1000, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1000, activation='relu')(x)
#x = Dropout(0.5)(x)
prediction = Dense(1, activation='sigmoid')(x)

# create a model object
model = Model(inputs=vgg.input, outputs=prediction)

# view the structure of the model
model.summary()

# tell the model what cost and optimization method to use
model.compile(
  loss='binary_crossentropy',
  optimizer=Adam(lr = 0.001),
  metrics=['accuracy']
)


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/Train/',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/Test/',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'binary')


# Enter the number of training and validation samples here
nb_train_samples = 50
nb_validation_samples = 25
batch_size = 8

from tensorflow.keras.callbacks import EarlyStopping

custom_early_stopping = EarlyStopping(
    monitor='val_accuracy', 
    patience=25, 
    min_delta=0.001, 
    mode='max',
    restore_best_weights=True
)

from keras.callbacks import ModelCheckpoint

filepath = 'epoch{epoch:02d}-accuracy{val_accuracy:.2f}.hdf5'

checkpoint = ModelCheckpoint(filepath=filepath, 
                             monitor='val_accuracy',
                             verbose=1, 
                             save_best_only=True,
                             mode='max')


# fit the model
r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=100,
  callbacks=[checkpoint, custom_early_stopping],
  steps_per_epoch= nb_train_samples // batch_size,
  validation_steps= nb_validation_samples // batch_size)


# loss
import matplotlib.pyplot as plt

plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# accuracies
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

import tensorflow as tf
from keras.models import load_model
model.save('arecanut_model.h5')