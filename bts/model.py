import keras
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Lambda,Concatenate
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D, Add
from keras.models import Model
import keras.backend as K
import tensorflow as tf


def two_path(X_input):
    # Local path Conv1
    X = Conv2D(64, (7, 7), strides=(1, 1), padding='valid')(X_input)

    # Batch-norm
    X = BatchNormalization()(X)

    X1 = Conv2D(64, (7, 7), strides=(1, 1), padding='valid')(X_input)
    X1 = BatchNormalization()(X1)
    # Max-out
    X = layers.Maximum()([X, X1])

    X = Conv2D(64, (4, 4), strides=(1, 1), padding='valid', activation='relu')(X)
    # Global path
    X2 = Conv2D(160, (13, 13), strides=(1, 1), padding='valid')(X_input)
    X2 = BatchNormalization()(X2)
    X21 = Conv2D(160, (13, 13), strides=(1, 1), padding='valid')(X_input)
    X21 = BatchNormalization()(X21)
    # Max-out
    X2 = layers.Maximum()([X2, X21])

    # Local path Conv2
    X3 = Conv2D(64, (3, 3), strides=(1, 1), padding='valid')(X)
    X3 = BatchNormalization()(X3)
    X31 = Conv2D(64, (3, 3), strides=(1, 1), padding='valid')(X)
    X31 = BatchNormalization()(X31)
    X = layers.Maximum()([X3, X31])
    X = Conv2D(64, (2, 2), strides=(1, 1), padding='valid', activation='relu')(X)

    # Merging the two paths
    X = Concatenate()([X2, X])
    # X = Conv2D(5,(21,21),strides=(1,1))(X)
    # X = Activation('softmax')(X)
    # model = Model(inputs = X_input, outputs = X)
    return X


def input_cascade(input_shape1, input_shape2):
    X1_input = Input(input_shape1)
    # 1st two-path of cascade
    X1 = two_path(X1_input)

    X1 = Conv2D(5, (21, 21), strides=(1, 1), padding='valid', activation='relu')(X1)
    X1 = BatchNormalization()(X1)

    X2_input = Input(input_shape2)
    # Concatenating the output of 1st to input of 2nd
    X2_input1 = Concatenate()([X1, X2_input])
    # X2_input1 = Input(tensor = X2_input1)
    X2 = two_path(X2_input1)

    # Fully convolutional softmax classification
    X2 = Conv2D(5, (21, 21), strides=(1, 1), padding='valid')(X2)
    X2 = BatchNormalization()(X2)
    X2 = Activation('softmax')(X2)

    model = Model(inputs=[X1_input, X2_input], outputs=[X2])
    return model


def MFCcascade(input_shape1, input_shape2):
    # 1st two-path
    X1_input = Input(input_shape1)
    X1 = two_path(X1_input)
    X1 = Conv2D(5, (21, 21), strides=(1, 1), padding='valid', activation='relu')(X1)
    X1 = BatchNormalization()(X1)
    # X1 = MaxPooling2D((2,2))(X1)

    # 2nd two-path
    X2_input = Input(input_shape2)
    X2 = two_path(X2_input)

    # Concatenate before classification
    X2 = Concatenate()([X1, X2])
    X2 = Conv2D(5, (21, 21), strides=(1, 1), padding='valid', activation='relu')(X2)
    X2 = BatchNormalization()(X2)
    X2 = Activation('softmax')(X2)

    model = Model(inputs=[X1_input, X2_input], outputs=X1)
    return model


def two_pathcnn(input_shape):
    X_input = Input(input_shape)

    X = Conv2D(64, (7, 7), strides=(1, 1), padding='valid')(X_input)
    X = BatchNormalization()(X)
    X1 = Conv2D(64, (7, 7), strides=(1, 1), padding='valid')(X_input)
    X1 = BatchNormalization()(X1)
    X = layers.Maximum()([X, X1])
    X = Conv2D(64, (4, 4), strides=(1, 1), padding='valid', activation='relu')(X)

    X2 = Conv2D(160, (13, 13), strides=(1, 1), padding='valid')(X_input)
    X2 = BatchNormalization()(X2)
    X21 = Conv2D(160, (13, 13), strides=(1, 1), padding='valid')(X_input)
    X21 = BatchNormalization()(X21)
    X2 = layers.Maximum()([X2, X21])

    X3 = Conv2D(64, (3, 3), strides=(1, 1), padding='valid')(X)
    X3 = BatchNormalization()(X3)
    X31 = Conv2D(64, (3, 3), strides=(1, 1), padding='valid')(X)
    X31 = BatchNormalization()(X31)
    X = layers.Maximum()([X3, X31])
    X = Conv2D(64, (2, 2), strides=(1, 1), padding='valid', activation='relu')(X)

    X = Concatenate()([X2, X])
    X = Conv2D(5, (21, 21), strides=(1, 1), padding='valid')(X)
    X = Activation('softmax')(X)

    model = Model(inputs=X_input, outputs=X)
    return model

