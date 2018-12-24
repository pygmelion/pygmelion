# coding: utf-8
import numpy as np
from keras.datasets import mnist
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from keras.callbacks import TensorBoard
from keras.models import load_model
import os

l2_reg = 1e-4
adam_lr = 1e-3
np.random.seed(1337)
epoch = 5

# download the mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# data pre-processing
X_train = X_train.reshape(-1, 1, 28, 28) / 255
X_test = X_test.reshape(-1, 1, 28, 28) / 255
Y_train = np_utils.to_categorical(Y_train, num_classes=10)
Y_test = np_utils.to_categorical(Y_test, num_classes=10)
if os.path.exists('CNN.h5'):
    model = load_model('CNN.h5')
else:

    # build CNN
    model = Sequential()

    # conv layer 1 output shape(32, 28, 28)
    model.add(Convolution2D(filters=32,
                            kernel_size=5,
                            strides=1,
                            padding='same',
                            batch_input_shape=(None, 1, 28, 28), kernel_regularizer=l2(l2_reg),
                            data_format='channels_first'))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    # pooling layer1 (max pooling) output shape(32, 14, 14)
    model.add(MaxPooling2D(pool_size=2,
                           strides=2,
                           padding='same',
                           data_format='channels_first'))

    # conv layer 2 output shape (64, 14, 14)
    model.add(Convolution2D(64, 5,
                            strides=1,
                            padding='same', kernel_regularizer=l2(l2_reg),
                            data_format='channels_first'))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))

    # pooling layer 2 (max pooling) output shape (64, 7, 7)
    model.add(MaxPooling2D(2, 2, 'same',
                           data_format='channels_first'))

    # full connected layer 1 input shape (64*7*7=3136), output shape (1024)
    model.add(Flatten())
    model.add(Dense(1024, kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))

    # full connected layer 2 to shape (10) for 10 classes
    model.add(Dense(10, kernel_regularizer=l2(l2_reg)))
    model.add(Activation('softmax'))

    # define optimizer
    model.compile(optimizer=Adam(lr=adam_lr), loss='categorical_crossentropy', metrics=['accuracy'])
    plot_model(model, to_file="model_CNN.png", show_shapes=True);

    # training

    model.fit(X_train, Y_train, epochs=epoch, batch_size=128, validation_split=0.3,callbacks=[TensorBoard(log_dir='/log')])
    model.save('CNN.h5')

# testing

loss, accuracy = model.evaluate(X_test, Y_test)
print('loss, accuracy: ', (loss, accuracy))
