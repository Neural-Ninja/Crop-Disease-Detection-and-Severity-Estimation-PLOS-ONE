import tensorflow as tf
from tensorflow import keras
import numpy as np

def inception_module(x, f1, f2, f3):
    conv1 =  keras.layers.Conv2D(f1, (1,1), padding='same', activation='relu')(x)
    conv3 = keras.layers.Conv2D(f2, (3,3), padding='same', activation='relu')(x)
    conv5 = keras.layers.Conv2D(f3, (5,5), padding='same', activation='relu')(x)
    pool = keras.layers.MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
    out = keras.layers.merge.concatenate([conv1, conv3, conv5, pool])
    return out

img_input = keras.Input(shape=(299, 299, 3))
classes=7
channel_axis=3

def conv2d_bn(x,filters,num_row,num_col,padding='same',strides=(1, 1)):
   
    x = keras.layers.Conv2D(filters, (num_row, num_col),strides=strides,padding=padding)(x)
    x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
    x = keras.layers.Activation('relu')(x)
    return x

def inc_block_a(x):    
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
    x = keras.layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=channel_axis)
    return x

def reduction_block_a(x):  
    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = keras.layers.concatenate([branch3x3, branch3x3dbl, branch_pool],axis=channel_axis)
    return x

def inc_block_b(x):
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 128, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 128, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = keras.layers.AveragePooling2D((3, 3), strides=(1, 1),padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = keras.layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=channel_axis)
    return x

def reduction_block_b(x): 
    branch3x3 = conv2d_bn(x, 192, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3,strides=(2, 2), padding='valid')

    branch7x7x3 = conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn( branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = keras.layers.concatenate([branch3x3, branch7x7x3, branch_pool], axis=channel_axis)
    return x

def inc_block_c(x):        
        branch1x1 = conv2d_bn(x, 320, 1, 1)

        branch3x3 = conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = keras.layers.concatenate([branch3x3_1, branch3x3_2],axis=channel_axis)

        branch3x3dbl = conv2d_bn(x, 448, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = keras.layers.concatenate([branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

        branch_pool = keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = keras.layers.concatenate( [branch1x1, branch3x3, branch3x3dbl, branch_pool],axis=channel_axis)
        return x
    
img_input = keras.Input(shape=(299, 299, 3))

x = conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid')
x = conv2d_bn(x, 32, 3, 3, padding='valid')
x = conv2d_bn(x, 64, 3, 3)

x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
x = conv2d_bn(x, 80, 1, 1, padding='valid')
x = conv2d_bn(x, 192, 3, 3, padding='valid')
x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)


x=inc_block_a(x)
x=inc_block_a(x)
x=inc_block_a(x)

x=reduction_block_a(x)

x=inc_block_b(x)
x=inc_block_b(x)
x=inc_block_b(x)
x=inc_block_b(x)

x=reduction_block_b(x)

x=inc_block_c(x)
x=inc_block_c(x)


x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)

x = keras.layers.Dense(classes, activation='softmax', name='predictions')(x)


inputs = img_input
model =  keras.Model(inputs, x, name='inception_v3')
#model.summary()

#from keras.utils import plot_model
#plot_model(model, show_shapes=True, to_file='inception_model_3.png')

from tensorflow.keras.optimizers import RMSprop
model.compile(optimizer = RMSprop(lr = 0.0001), loss = 'categorical_crossentropy', metrics = 'acc')

dir_train = '/Tomato/train'
dir_val = '/Tomato/val'
dir_test = '/Tomato/test'

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_gen = ImageDataGenerator(rescale = 1./255)
train_data = train_gen.flow_from_directory(dir_train, batch_size = 32, class_mode = 'categorical', target_size = (299, 299))

val_gen = ImageDataGenerator(rescale = 1./255)
val_data = val_gen.flow_from_directory(dir_val, batch_size = 32, class_mode = 'categorical', target_size = (299, 299))

test_gen = ImageDataGenerator(rescale = 1./255)
test_data = test_gen.flow_from_directory(dir_test, batch_size = 32, class_mode = 'categorical', shuffle = False, target_size = (299, 299))

model.fit_generator(train_data, epochs = 100, validation_data = val_data, steps_per_epoch = 200)

from sklearn.metrics import confusion_matrix, accuracy_score

y_pred = model.predict(test_data)

y_pred = np.argmax(y_pred, axis = -1)

y_true = test_data.classes

cm = confusion_matrix(y_true, y_pred)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

print(accuracy_score(y_true, y_pred))