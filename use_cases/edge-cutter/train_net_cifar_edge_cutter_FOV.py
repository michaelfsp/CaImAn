"""
Created on Thu Aug 24 12:30:19 2017

@author: agiovann
"""

'''From keras example of convnet on the MNIST dataset.

TRAIN ON DATA EXTRACTED FROM RESIDUALS WITH generate_GT script. THIS IS MORE OF A OVERFEAT TYPE OF NETWORK

'''
#%%
#import os
# os.chdir('/mnt/home/agiovann/SOFTWARE/CaImAn')
#from __future__ import division
#from __future__ import print_function
#from builtins import zip
#from builtins import str
#from builtins import map
#from builtins import range
#from past.utils import old_div
import cv2
import glob

try:
    cv2.setNumThreads(1)
except:
    print('Open CV is naturally single threaded')

try:
    if __IPYTHON__:
        print(1)
        # this is used for debugging purposes only. allows to reload classes
        # when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    print('Not IPYTHON')
    pass
import caiman as cm
import numpy as np
import os
import time
import pylab as pl
import psutil
import sys
from ipyparallel import Client
from skimage.external.tifffile import TiffFile
import scipy
import copy

from caiman.utils.utils import download_demo
from caiman.base.rois import extract_binary_masks_blob
from caiman.utils.visualization import plot_contours, view_patches_bar
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.motion_correction import MotionCorrect
from caiman.components_evaluation import estimate_components_quality

from caiman.components_evaluation import evaluate_components

from caiman.tests.comparison import comparison
from caiman.motion_correction import tile_and_correct, motion_correction_piecewise
import keras
from keras.datasets import mnist
from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Activation, Flatten

import json as simplejson
from keras.models import model_from_json
from sklearn.utils import class_weight as cw
from caiman.utils.image_preprocessing_keras import ImageDataGenerator
from keras.layers import merge
from keras.layers.core import Lambda
from keras.models import Model

import tensorflow as tf
#%%


def make_parallel(model, gpu_count):
    def get_slice(data, idx, parts):
        shape = tf.shape(data)
        size = tf.concat([shape[:1] // parts, shape[1:]], axis=0)
        stride = tf.concat([shape[:1] // parts, shape[1:] * 0], axis=0)
        start = stride * idx
        return tf.slice(data, start, size)

    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])

    # Place a copy of the model on each GPU, each getting a slice of the batch
    for i in range(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:

                inputs = []
                # Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape, arguments={
                                     'idx': i, 'parts': gpu_count})(x)
                    inputs.append(slice_n)

                outputs = model(inputs)

                if not isinstance(outputs, list):
                    outputs = [outputs]

                # Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])

    # merge outputs on CPU
    with tf.device('/cpu:0'):
        merged = []
        for outputs in outputs_all:
            merged.append(merge(outputs, mode='concat', concat_axis=0))

        return Model(input=model.inputs, output=merged)


#%%
# the data, shuffled and split between train and test sets
with np.load('use_cases/edge-cutter/residual_crops_all_classes.npz') as ld:
    all_masks_gt = ld['all_masks_gt'][:, 1:-1, 1:-1]
    labels_gt = ld['labels_gt']
    all_masks_gt = all_masks_gt[labels_gt < 2]
    labels_gt = labels_gt[labels_gt < 2]
#%%
batch_size = 128
num_classes = 2
epochs = 5
test_fraction = 0.25
augmentation = True
# input image dimensions
img_rows, img_cols = 48, 48


x_train, x_test, y_train, y_test = train_test_split(
    all_masks_gt, labels_gt, test_size=test_fraction)

class_weight = cw.compute_class_weight('balanced', np.unique(y_train), y_train)

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#x_train /= 255
#x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)
#%%

# def get_conv(input_shape=(48,48,1), filename=None):
#    model = Sequential()
##    model.add(Lambda(lambda x: (x-np.mean(x))/np.std(x),input_shape=input_shape, output_shape=input_shape))
#    model.add(Conv2D(32, (3, 3), activation='relu', name='conv1', input_shape=input_shape, padding="same"))
#    model.add(Conv2D(32, (3, 3), activation='relu', name='conv2', padding="same"))
#    model.add(MaxPooling2D(pool_size=(2,2)))
#    model.add(Dropout(0.25))
#
#    model.add(Conv2D(48, (3, 3), name = 'conv3', padding='same'))
#    model.add(Activation('relu'))
#    model.add(Conv2D(48, (3, 3), name = 'conv4', padding='same'))
#    model.add(Activation('relu'))
#    model.add(MaxPooling2D(pool_size=(3, 3)))
#    model.add(Dropout(0.25))
#
#    model.add(Conv2D(256,(8,8), activation="relu", name="dense1")) # This was Dense(128)
#    model.add(Dropout(0.5))
#    model.add(Conv2D(1, (1,1), name="dense2", activation="tanh")) # This was Dense(1)
#    if filename:
#        model.load_weights(filename)
#    return model


def get_conv(input_shape=(48, 48, 1), filename=None):
    model = Sequential()
#    model.add(Lambda(lambda x: (x-np.mean(x))/np.std(x),input_shape=input_shape, output_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu', name='conv1',
                     input_shape=input_shape, padding="same"))
    model.add(Conv2D(32, (3, 3), activation='relu',
                     name='conv2', padding="same"))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), name='conv3', padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), name='conv4', padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), activation="relu",
                     name="dense1"))  # This was Dense(128)
    model.add(Dropout(0.5))
    # This was Dense(1)
    model.add(Conv2D(1, (1, 1), name="dense2", activation="tanh"))
    if filename:
        model.load_weights(filename)
    return model


#model = Sequential()
#
# model.add(Conv2D(32, kernel_size=(3, 3),
#                 activation='relu',
#                 input_shape=input_shape))
# model.add(Activation('relu'))
#model.add(Conv2D(32, (3, 3)))
# model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
#
#model.add(Conv2D(64, (3, 3), padding='same'))
# model.add(Activation('relu'))
#model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
#
#
# model.add(Conv2D(128,(8,8), activation="relu", name="dense1")) # This was Dense(128)
# model.add(Dropout(0.5))
# model.add(Conv2D(1, (1,1), name="dense2", activation="tanh")) # This was Dense(1)


model = get_conv()
model.add(Flatten())
#opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
model.compile(loss='mse', optimizer='adadelta', metrics=['accuracy'])


# model.add(Flatten())
# model.add(Dense(512))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes))
# model.add(Activation('softmax'))

#model = make_parallel(model, 2)

# initiate RMSprop optimizer

# model.compile(loss=keras.losses.categorical_crossentropy,
#              optimizer=opt,
#              metrics=['accuracy'])

if augmentation:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        #            featurewise_center=True,
        #            featurewise_std_normalization=True,
        shear_range=0.3,
        rotation_range=360,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=[0.8, 1.2],
        horizontal_flip=True,
        vertical_flip=True,
        random_mult_range=[.25, 2]
    )

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    history = model.fit_generator(datagen.flow(x_train, y_train,
                                               batch_size=batch_size),
                                  steps_per_epoch=x_train.shape[0] // batch_size,
                                  epochs=epochs,
                                  verbose=1,
                                  class_weight=class_weight,
                                  validation_data=(x_test, y_test))


else:
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))

#%%
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#%% Save model and weights
import datetime
save_dir = 'use_cases/edge-cutter/'
model_name = str(datetime.datetime.now()).replace(' ', '-').replace(':', '-')
model_json = model.to_json()
json_path = os.path.join(save_dir, model_name + '.json')

with open(json_path, "w") as json_file:
    json_file.write(simplejson.dumps(simplejson.loads(model_json), indent=4))

print('Saved trained model at %s ' % json_path)


if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name + '.h5')
model.save(model_path)
print('Saved trained model at %s ' % model_path)
#%% Turn the classifier into a heat map¶
heatmodel = get_conv(input_shape=(None, None, 1), filename=model_path)
import matplotlib.pylab as plt


def locate(data):
    #    data = cv2.cvtColor(cv2.imread("test1.jpg"), cv2.COLOR_BGR2RGB)

    heatmap = heatmodel.predict(data.reshape(
        1, data.shape[0], data.shape[1], data.shape[2]))

    plt.imshow(heatmap[0, :, :, 0])
    plt.title("Heatmap")
    plt.show()
    plt.imshow(heatmap[0, :, :, 0] > 0.99, cmap="gray")
    plt.title("Car Area")
    plt.show()

    xx, yy = np.meshgrid(
        np.arange(heatmap.shape[2]), np.arange(heatmap.shape[1]))
    x = (xx[heatmap[0, :, :, 0] > 0.99])
    y = (yy[heatmap[0, :, :, 0] > 0.99])

    for i, j in zip(x, y):
        cv2.rectangle(data, (i * 8, j * 8),
                      (i * 8 + 64, j * 8 + 64), (0, 0, 255), 5)
    return data


annotated = locate(data)

plt.title("Augmented")
plt.imshow(annotated)
plt.show()
#%% visualize_results
num_sampl = 30000
predictions = model.predict(
    all_masks_gt[:num_sampl, :, :, None], batch_size=32, verbose=1)
cm.movie(np.squeeze(all_masks_gt[np.where(predictions[:num_sampl, 0] >= 0.95)[
         0]])).play(gain=3., magnification=5, fr=10)
#%%
cm.movie(np.squeeze(all_masks_gt[np.where(predictions[:num_sampl, 1] >= 0.95)[
         0]])).play(gain=3., magnification=5, fr=10)
#%%
pl.imshow(montage2d(all_masks_gt[np.where((labels_gt[:num_sampl] == 0) & (
    predictions[:num_sampl, 1] > 0.95))[0]].squeeze()))
#%%
pl.imshow(montage2d(all_masks_gt[np.where((labels_gt[:num_sampl] == 1) & (
    predictions[:num_sampl, 0] > 0.95))[0]].squeeze()))
#%%
pl.imshow(montage2d(all_masks_gt[np.where(
    (predictions[:num_sampl, 0] > 0.95))[0]].squeeze()))
#%% retrieve and test
json_file = open(json_path, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(model_path)
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
loaded_model.compile(loss=keras.losses.categorical_crossentropy,
                     optimizer=opt,
                     metrics=['accuracy'])
print("Loaded model from disk")
score = loaded_model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
score = loaded_model.evaluate(x_train, y_train, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
#%%
from skimage.util.montage import montage2d
predictions = loaded_model.predict(
    all_masks_gt[:num_sampl], batch_size=32, verbose=1)
cm.movie(np.squeeze(all_masks_gt[np.where(predictions[:num_sampl, 1] < 0.1)[0]])).play(
    gain=3., magnification=5, fr=10)
#%%
pl.imshow(montage2d(all_masks_gt[np.where((labels_gt[:num_sampl] == 0) & (
    predictions[:num_sampl, 1] >= 0.5))[0]].squeeze()))
#%%
pl.imshow(montage2d(all_masks_gt[np.where((labels_gt == 1) & (
    predictions[:num_sampl, 0] >= 0.5) & (predictions[:, 0] >= 0.5))[0]].squeeze()))
