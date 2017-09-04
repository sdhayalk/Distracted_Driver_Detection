import numpy as np
import csv
import pickle
import cv2
import os

from preprocessing import get_dataset, get_test_dataset
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, merge, ZeroPadding2D, AveragePooling2D, Flatten, Dropout, Dense, Activation, BatchNormalization
from keras.regularizers import l2
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from LRN_helper import LRN2D

number_of_classes = 10
dimension = 224
number_of_channels = 1
batch_size = 50

train_directory = 'G:/DL/distracted-driver-new/driver_imgs_list.csv/train'
test_directory = 'G:/DL/distracted-driver-new/driver_imgs_list.csv/test'

def inception(input, prefix, n1x1, r3x3, n3x3, r5x5, n5x5, m1x1):
    # input = Input(shape=shape)(input)
    layer_conv_1x1_b = Convolution2D(r3x3, 1, 1, border_mode='same', activation='relu', name=prefix+'layer_conv_1x1_b', W_regularizer=l2(0.0002))(input)
    layer_conv_1x1_b= BatchNormalization()(layer_conv_1x1_b)
    layer_conv_1x1_c = Convolution2D(r5x5, 1, 1, border_mode='same', activation='relu', name=prefix+'layer_conv_1x1_c', W_regularizer=l2(0.0002))(input)
    layer_conv_1x1_c = BatchNormalization()(layer_conv_1x1_c)
    layer_max_3x3_d = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), border_mode='same', name=prefix+'layer_max_3x3_d')(input)

    layer_conv_1x1_a = Convolution2D(n1x1, 1, 1, border_mode='same', activation='relu', name=prefix+'layer_conv_1x1_a', W_regularizer=l2(0.0002))(input)
    layer_conv_1x1_a = BatchNormalization()(layer_conv_1x1_a)
    layer_conv_3x3_b = Convolution2D(n3x3, 3, 3, border_mode='same', activation='relu', name=prefix+'layer_conv_3x3_b', W_regularizer=l2(0.0002))(layer_conv_1x1_b)
    layer_conv_3x3_b = BatchNormalization()(layer_conv_3x3_b)
    layer_conv_5x5_c = Convolution2D(n5x5, 5, 5, border_mode='same', activation='relu', name=prefix+'layer_conv_5x5_c', W_regularizer=l2(0.0002))(layer_conv_1x1_c)
    layer_conv_5x5_c = BatchNormalization()(layer_conv_5x5_c)
    layer_conv_1x1_d = Convolution2D(m1x1, 1, 1, border_mode='same', activation='relu', name=prefix+'layer_conv_1x1_d', W_regularizer=l2(0.0002))(layer_max_3x3_d)
    layer_conv_1x1_d = BatchNormalization()(layer_conv_1x1_d)

    output = merge([layer_conv_1x1_a, layer_conv_3x3_b, layer_conv_5x5_c, layer_conv_1x1_d], mode='concat')
    return output


dataset_test_features, file_names = get_test_dataset(test_directory)
print(dataset_test_features.shape)
print(file_names.shape)

# normalizing
dataset_test_features = dataset_test_features / 255.0

dataset_test_features = dataset_test_features.reshape((-1, dimension, dimension, number_of_channels))

# googlenet start:
input = Input(shape=(dimension, dimension, number_of_channels))
conv1 = Convolution2D(64, 7, 7, subsample=(2,2), border_mode='same', activation='relu', W_regularizer=l2(0.0002))(input)
conv1 = ZeroPadding2D(padding=(1,1))(conv1)
conv1 = MaxPooling2D(pool_size=(3,3), strides=(2,2), border_mode='valid')(conv1)
conv1 = LRN2D()(conv1)

conv1 = BatchNormalization()(conv1)
conv2 = Convolution2D(64, 1, 1, border_mode='same', activation='relu', W_regularizer=l2(0.0002))(conv1)
conv2 = Convolution2D(192, 3, 3, border_mode='same', activation='relu', W_regularizer=l2(0.0002))(conv2)
conv2 = LRN2D()(conv2)
conv2 = ZeroPadding2D(padding=(1,1))(conv2)
conv2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), border_mode='valid')(conv2)

conv2 = BatchNormalization()(conv2)
inception1 = inception(conv2, '3a', 64, 96, 128, 16, 32, 32)

inception1 = BatchNormalization()(inception1)
inception2 = inception(inception1,'3b', 128, 128, 192, 32, 96, 64)
inception2 = ZeroPadding2D(padding=(1,1))(inception2)
inception2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), border_mode='valid')(inception2)

inception2 = BatchNormalization()(inception2)
inception3 = inception(inception2, '4a', 192, 96, 208, 16, 48, 64)

inception3 = BatchNormalization()(inception3)
inception4 = inception(inception3, '4b', 160, 112, 224, 24, 64, 64)

inception4 = BatchNormalization()(inception4)
inception5 = inception(inception4, '4c', 128, 128, 256, 24, 64, 64)

inception5 = BatchNormalization()(inception5)
inception6 = inception(inception5, '4d', 112, 144, 288, 32, 64, 64)

inception6 = BatchNormalization()(inception6)
inception7 = inception(inception6, '4e', 256, 160, 320, 32, 128, 128)
inception7 = ZeroPadding2D(padding=(1,1))(inception7)
inception7 = MaxPooling2D(pool_size=(3,3), strides=(2,2), border_mode='valid')(inception7)

inception7 = BatchNormalization()(inception7)
inception8 = inception(inception7, '5a', 256, 160, 320, 32, 128, 128)

inception8 = BatchNormalization()(inception8)
inception9 = inception(inception8, '5b', 384, 192, 384, 48, 128, 128)
# inception9 = ZeroPadding2D(padding=(1,1))(inception9)
inception9 = AveragePooling2D(pool_size=(7,7), strides=(1,1), border_mode='valid')(inception9)

inception9 = BatchNormalization()(inception9)
flatten = Flatten()(inception9)
fc = Dense(1024, activation='relu', name='fc')(flatten)
fc = Dropout(0.7)(fc)

fc = BatchNormalization()(fc)
output_layer = Dense(number_of_classes, name='output_layer')(fc)
output_layer = Activation('softmax')(output_layer)

epochs = 50
lrate = 0.0001
decay = lrate/epochs
adam = Adam(decay=decay)


''' testing and evaluation '''

model = Model(inputs=input, outputs=output_layer)
model.load_weights('G:/DL/distracted-driver-new/floydhub/weights-improvement-03-0.93.h5py')
predictions = model.predict(dataset_test_features)

print(predictions)
# np.savetxt("/output/googlenet.csv", predictions, delimiter=",")
with open('G:/DL/distracted-driver-new/floydhub/googlenet.csv','w') as file:
    file.write('img,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9')
    file.write('\n')
    example_number = 0

    for file_name in file_names:
        file.write(file_name)
        file.write(',')

        for i in range(0, 9):
            value = predictions[example_number][i]
            file.write(str('{:f}'.format(value)))
            file.write(',')

        file.write(predictions[example_number][i])
        file.write('\n')
        example_number += 1
