import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3

dir_train = 'Tomato/train'
dir_val = 'Tomato/val'
dir_test = 'Tomato/test'

train_gen = ImageDataGenerator(rescale = 1./255)
train_data = train_gen.flow_from_directory(dir_train, batch_size = 16, class_mode = 'categorical', target_size = (299, 299))

val_gen = ImageDataGenerator(rescale = 1./255)
val_data = val_gen.flow_from_directory(dir_val, batch_size = 16, class_mode = 'categorical', target_size = (299, 299))

test_gen = ImageDataGenerator(rescale = 1./255)
test_data = test_gen.flow_from_directory(dir_test, batch_size = 16, class_mode = 'categorical', target_size = (299, 299))

base_model = InceptionV3(input_shape = (299, 299, 3),
                                include_top = False,
                                weights = 'imagenet')
for layer in base_model.layers:
  layer.trainable = False
  
x = layers.Flatten()(base_model.output)
x= layers.Dense(64, activation='relu')(x)
x = layers.Dense(10, activation='softmax')(x)

 
model = Model( base_model.input, x)

model.compile(optimizer = 'Adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])

model.summary()

history = model.fit_generator(
            train_data,
            validation_data = val_data,
            epochs = 50,
            steps_per_epoch = 100)

import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
 
epochs = range(len(acc))
 
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
 
plt.figure()
 
plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

from sklearn.metrics import accuracy_score

print(accuracy_score(true_classes, predicted_classes))