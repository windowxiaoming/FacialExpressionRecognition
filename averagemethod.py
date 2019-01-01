from __future__ import print_function

import cv2
import PIL
from PIL import Image
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adadelta
from keras.utils import np_utils
import numpy as np
import numpy
import cv2
import scipy
import csv
import dataprocessing
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D,Conv2D
from keras.optimizers import Adadelta
from keras.utils import np_utils
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
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
	output = Activation('softmax',name='predictions')(x)

	model = Model(img_input, output)
	return model

def ConvertTo3DVolume(data):
	img_rows, img_cols = 48, 48
	test_set_x = numpy.asarray(data) 
	test_set_x = test_set_x.reshape(test_set_x.shape[0],img_rows,img_cols)
	test_set_x = test_set_x.reshape(test_set_x.shape[0], img_rows, img_cols,1)
	test_set_x = test_set_x.astype('float32')
	return test_set_x

def predict_prob(number,test_set_x,model):
	toreturn = []
	for data5 in test_set_x:
		if number ==0:
			toreturn.append(dataprocessing.Flip(data5))
		elif number ==1:
			toreturn.append(dataprocessing.Roated15Left(data5))
		elif number ==2:
			toreturn.append(dataprocessing.Roated15Right(data5))
		elif number ==3:
			toreturn.append(dataprocessing.shiftedUp20(data5))
		elif number ==4:
			toreturn.append(dataprocessing.shiftedDown20(data5))
		elif number ==5:
			toreturn.append(dataprocessing.shiftedLeft20(data5))
		elif number ==6:
			toreturn.append(dataprocessing.shiftedRight20(data5))
		elif number ==7:
			toreturn.append(data5)
	toreturn = ConvertTo3DVolume(toreturn)
	proba = model.predict(toreturn)
	return proba

img_rows, img_cols = 48, 48
nb_classes = 7
# model = model_generate()
model = big_XCEPTION((img_rows,img_cols,1), nb_classes)
model.compile(loss='categorical_crossentropy',
				  optimizer='adam',
				  metrics=['accuracy'])
filepath='Model.110-0.6810.hdf5'
model.load_weights(filepath)
model.summary()
test_set_x, test_set_y = dataprocessing.load_test_data()

proba = predict_prob(0,test_set_x,model)
proba1 = predict_prob(1,test_set_x,model)
proba2 = predict_prob(2,test_set_x,model)
proba3 = predict_prob(3,test_set_x,model)
proba4 = predict_prob(4,test_set_x,model)
proba5 = predict_prob(5,test_set_x,model)
proba6 = predict_prob(6,test_set_x,model)
proba7 = predict_prob(7,test_set_x,model)
Out = []
for row in zip(proba,proba1,proba2,proba3,proba4,proba5,proba6,proba7):
	a = numpy.argmax(np.array(row).mean(axis=0))
	Out.append(a)

Out = np.array(Out)
test_set_y = np.array(test_set_y)
c = np.sum(Out == test_set_y)
print("Acc:"+str((float(c)/len(Out))))
np.savetxt('pred2',Out)  
np.savetxt('test2',test_set_y) 
# from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')

#     print(cm)

#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)

#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")

#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')

# # Compute confusion matrix
# cnf_matrix = confusion_matrix(test_set_y, Out)
# np.set_printoptions(precision=2)

# # Plot non-normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names,
#                       title='Confusion matrix, without normalization')

# # Plot normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
#                       title='Normalized confusion matrix')

# plt.savefig("./result.jpg")


