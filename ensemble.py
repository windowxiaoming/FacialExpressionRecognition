from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D,Conv2D
from keras.optimizers import Adadelta
from keras.utils import np_utils
# from keras.regularizers import l2, activity_l2
import pickle 
import numpy
import csv
import scipy.misc
import scipy
from scipy import ndimage
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import dataprocessing
from keras.layers import Activation, Convolution2D, Dropout, Conv2D
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras import layers
from keras.regularizers import l2
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras import layers
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "8"
def model_generate():
    img_rows, img_cols = 48, 48
    model = Sequential()
    
    model.add(Conv2D(64,(5,5),input_shape=(img_rows, img_cols,1),padding="valid"))
    model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(2, 2), dim_ordering='tf'))
    model.add(MaxPooling2D(pool_size=(5, 5),strides=(2, 2)))
      
    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='tf')) 
    # model.add(Convolution2D(3, 3,64))
    model.add(Conv2D(64,(3,3)))
    model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='tf')) 
    # model.add(Convolution2D(3, 3,64))
    model.add(Conv2D(64,(3,3)))
    model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
    model.add(keras.layers.convolutional.AveragePooling2D(pool_size=(3, 3),strides=(2, 2)))
     
    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
    # model.add(Convolution2D(3, 3,128))
    model.add(Conv2D(128,(3,3)))
    model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
    # model.add(Convolution2D(3, 3,128))
    model.add(Conv2D(128,(3,3)))
    model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
     
    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
    model.add(keras.layers.convolutional.AveragePooling2D(pool_size=(3, 3),strides=(2, 2)))
     
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
    model.add(Dropout(0.2))
    model.add(Dense(1024))
    model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
    model.add(Dropout(0.2))
     
      
    model.add(Dense(7))
      
      
    model.add(Activation('softmax'))

    
    return model

def naive_model(input_shape, num_classes):
    img_input = Input(input_shape)
    x = Conv2D(64, (5, 5),padding="valid")(img_input)
    x = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(x)
    x = keras.layers.convolutional.ZeroPadding2D(padding=(2, 2), dim_ordering='tf')(x)
    x = MaxPooling2D(pool_size=(5, 5),strides=(2, 2))(x)

    x = keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='tf')(x)
    x = Conv2D(64,(3,3))(x)
    x = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(x)
    x = keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='tf')(x)
    x = Conv2D(64,(3,3))(x)
    x = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(x)
    x = keras.layers.convolutional.AveragePooling2D(pool_size=(3, 3),strides=(2, 2))(x)
   
    x = keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='tf')(x)
    x = Conv2D(128,(3,3))(x)
    x = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(x)
    x = keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='tf')(x)
    x = Conv2D(128,(3,3))(x)
    x = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(x)

    x = keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='tf')(x)
    x = keras.layers.convolutional.AveragePooling2D(pool_size=(3, 3),strides=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(1024)(x)
    x = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(x)
    x = Dropout(0.2)(x)
    x = Dense(1024)(x)
    x = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(x)
    x = Dense(7)(x)
    sub1 = Dropout(0.2)(x)

    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False)(img_input)
    x = BatchNormalization(name='block1_conv1_bn')(x)
    x = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(x)
    x = Conv2D(64, (3, 3), use_bias=False)(x)
    x = BatchNormalization(name='block1_conv2_bn')(x)
    x = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(x)

    residual = Conv2D(128, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization(name='block2_sepconv1_bn')(x)
    x = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(x)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization(name='block2_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    residual = Conv2D(256, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization(name='block3_sepconv1_bn')(x)
    x = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization(name='block3_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])
    x = Conv2D(num_classes, (3, 3),
            #kernel_regularizer=regularization,
            padding='same')(x)
    sub2 = GlobalAveragePooling2D()(x)
    
    
    

    Concatenate = keras.layers.Concatenate(axis=1)([sub1, sub2])
    Concatenate = Dense(7)(Concatenate)
    output = Activation('softmax',name='predictions')(Concatenate)

    model = Model(inputs = img_input, outputs = output)
    return model

def big_XCEPTION(input_shape, num_classes):
    img_input = Input(input_shape)
    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False)(img_input)
    x = BatchNormalization(name='block1_conv1_bn')(x)
    x = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(x)
    x = Conv2D(64, (3, 3), use_bias=False)(x)
    x = BatchNormalization(name='block1_conv2_bn')(x)
    x = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(x)

    residual = Conv2D(128, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization(name='block2_sepconv1_bn')(x)
    x = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(x)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization(name='block2_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    residual = Conv2D(256, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization(name='block3_sepconv1_bn')(x)
    x = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization(name='block3_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])
    x = Conv2D(num_classes, (3, 3),
            #kernel_regularizer=regularization,
            padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    # output = Activation('softmax',name='predictions')(x)

    # model = Model(img_input, output)
    return x



img_rows, img_cols = 48, 48
batch_size = 128
nb_classes = 7
nb_epoch = 1200
img_channels = 1

Train_x, Train_y, Val_x, Val_y = dataprocessing.load_data()

Train_x = numpy.asarray(Train_x) 
Train_x = Train_x.reshape(Train_x.shape[0],img_rows,img_cols)

Val_x = numpy.asarray(Val_x)
Val_x = Val_x.reshape(Val_x.shape[0],img_rows,img_cols)

Train_x = Train_x.reshape(Train_x.shape[0], img_rows, img_cols,1)
Val_x = Val_x.reshape(Val_x.shape[0],img_rows, img_cols,1)

Train_x = Train_x.astype('float32')
Val_x = Val_x.astype('float32')


Train_y = np_utils.to_categorical(Train_y, nb_classes)
Val_y = np_utils.to_categorical(Val_y, nb_classes)


model = naive_model((img_rows,img_cols,1), nb_classes)

model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
model.summary()
patience = 50
filepath='./ensemble/Model.{epoch:02d}-{val_acc:.4f}.hdf5'
log_file_path = './ensemble/training.log'
model_checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
early_stop = EarlyStopping('val_acc', patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,patience=int(patience/4), verbose=1)
csv_logger = CSVLogger(log_file_path, append=False)

datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=.1,
    horizontal_flip=True,
    data_format='channels_last') 

datagen.fit(Train_x)

model.fit_generator(datagen.flow(Train_x, Train_y,
                    batch_size=batch_size),
                    samples_per_epoch=Train_x.shape[0],
                    nb_epoch=nb_epoch,
                    validation_data=(Val_x, Val_y),
                    callbacks=[model_checkpoint, csv_logger, early_stop, reduce_lr])

