from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import tensorflow as tf


if __name__ == "__main__":
    batch_size = 128
    num_classes = 10
    epochs =10

    # pretreatment 预处理
    # input image dimensions
    img_rows, img_cols = 28, 28
    pre_rows, pre_cols = 32, 32

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    sess1 = tf.Session()
    #b = a.eval(session=sess1)

    #for img in range(0,x_train.shape[0])

    if K.image_data_format() == 'channels_first':  # 判断图片格式是 channel在前还是在后（channel：黑白为1,彩色为3）
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)  # shape[0]指例子的个数
        x_train = tf.image.resize_image_with_crop_or_pad(x_train, pre_rows, pre_cols).eval(session=sess1)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        x_test = tf.image.resize_image_with_crop_or_pad(x_test, pre_rows, pre_cols).eval(session=sess1)
        input_shape = (1, pre_rows, pre_cols)

    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_train = tf.image.resize_image_with_crop_or_pad(x_train, pre_rows, pre_cols).eval(session=sess1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1);
        x_test = tf.image.resize_image_with_crop_or_pad(x_test, pre_rows, pre_cols).eval(session=sess1)
        input_shape = (pre_rows, pre_cols, 1)
        print("shape(x_train)", np.shape(x_train))  # 60000,28,28,1
        print("shape(x_test)", np.shape(x_test))    # 10000,28,28,1

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    print("shape(y_train)", np.shape(y_train))  # 60000,10

    model = Sequential()
    #Layer1
    #input: size=batch,32,32,1
    #output: size=batch,28,28,6
    model.add(Conv2D(6, kernel_size=(5, 5),
                     activation='relu',
                     input_shape=input_shape))
    # Layer2
    #input:    size= batch,28,28,6
    #output:   size= batch,14,14,6
    model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
    # Layer3
    # input:   size= batch,14,14,6
    # output:  size= batch,10,10,16
    model.add(Conv2D(16, kernel_size=(5, 5),activation='tanh'))

    # Layer4
    # input:    size= batch,10,10,16
    # output:   size= batch,5,5,16
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer5
    # input:   size= batch,5,5,16
    # output:  size= batch,1,1,120
    model.add(Conv2D(120, kernel_size=(5, 5),
                     activation='tanh'))
    # Layer6
    # input:   size= batch,1,1,120
    # output:  size= batch,84
    model.add(Flatten())
    model.add(Dense(84, activation='tanh'))

    # Layer7
    # input:  size= batch,84
    # output: size= batch,10
    model.add(Dense(10,  activation='softmax'))
    #
    model.compile(#optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False),
                  optimizer=keras.optimizers.Adadelta(),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])

    model.fit(x_train,y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=0,
              validation_split=0.1
              #validation_data=(x_test, y_test)
              )

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])     #Test loss: 0.032095696574577594
    print('Test accuracy:', score[1]) #Test accuracy: 0.989