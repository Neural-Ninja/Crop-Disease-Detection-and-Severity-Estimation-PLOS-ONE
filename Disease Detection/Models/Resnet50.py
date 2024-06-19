from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

dir_train = 'Tomato/train'
dir_val = 'Tomato/val'
dir_test = 'Tomato/test'

train_gen = ImageDataGenerator(rescale = 1./255)
train_data = train_gen.flow_from_directory(dir_train, batch_size = 16, class_mode = 'categorical', target_size = (224, 224))

val_gen = ImageDataGenerator(rescale = 1./255)
val_data = val_gen.flow_from_directory(dir_val, batch_size = 16, class_mode = 'categorical', target_size = (224, 224))

test_gen = ImageDataGenerator(rescale = 1./255)
test_data = test_gen.flow_from_directory(dir_test, batch_size = 16, class_mode = 'categorical', target_size = (224, 224))

resnet = ResNet50(input_shape=[224, 224] + [3], weights='imagenet', include_top=False)

for layer in resnet.layers:
    layer.trainable = False
    
x = Flatten()(resnet.output)
out = Dense(10, activation='softmax')(x)

model = Model(inputs=resnet.input, outputs=out)

model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

history = model.fit_generator(
  train_data,
  validation_data=val_data,
  epochs=50,
  steps_per_epoch=200)