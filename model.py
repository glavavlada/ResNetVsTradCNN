import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, BatchNormalization,
    Activation, Add, GlobalAveragePooling2D, Dense,
    Dropout, Flatten
)
# Vladyslav Glavatskti

def create_traditional_cnn(input_shape=(32, 32, 3), num_classes=10):
    inputs = Input(shape=input_shape)

    #first convolutional block
    x = Conv2D(16, (3, 3), padding='same',
               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    #second convolutional block
    x = Conv2D(32, (3, 3), padding='same',
               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    #fully connected layers
    x = Flatten()(x)
    x = Dense(64, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)  # Slightly increased dropout
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def create_resnet(input_shape=(32, 32, 3), num_classes=10):
    inputs = Input(shape=input_shape)

    #initial convolution
    x = Conv2D(32, 3, strides=1, padding='same',
               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    #first residual block
    shortcut = x
    x = Conv2D(32, 3, padding='same',
               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, 3, padding='same',
               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    #downsampling
    x = MaxPooling2D(pool_size=(2, 2))(x)

    #second residual block
    shortcut = Conv2D(64, 1, padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    shortcut = BatchNormalization()(shortcut)

    x = Conv2D(64, 3, padding='same',
               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, 3, padding='same',
               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    #global pooling and final dense layer
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    x = Dense(64, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = Activation('relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def count_parameters(model):
    return sum(p.numpy().size for p in model.trainable_weights)
