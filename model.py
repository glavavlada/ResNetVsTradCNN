import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, BatchNormalization,
    Activation, Add, GlobalAveragePooling2D, Dense,
    Dropout, Flatten, DepthwiseConv2D, Multiply,
    GlobalMaxPooling2D, Concatenate, Reshape
)


# Vladyslav Glavatskyi

def create_traditional_cnn(input_shape=(32, 32, 3), num_classes=10):
    inputs = Input(shape=input_shape)

    #block 1
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    #block 2
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    #block 3
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    #classifier
    x = Flatten()(x)
    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
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
    x = Conv2D(32, 3, strides=1, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    def residual_block(input_tensor, filters, stride=1):
        shortcut = input_tensor

        #first conv
        x = Conv2D(filters, 3, strides=stride, padding='same',
                   kernel_regularizer=tf.keras.regularizers.l2(1e-4))(input_tensor)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        #second conv
        x = Conv2D(filters, 3, strides=1, padding='same',
                   kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = BatchNormalization()(x)

        #adjust shortcut dimensions if needed
        if stride != 1 or input_tensor.shape[-1] != filters:
            shortcut = Conv2D(filters, 1, strides=stride, padding='same',
                              kernel_regularizer=tf.keras.regularizers.l2(1e-4))(shortcut)
            shortcut = BatchNormalization()(shortcut)

        #add shortcut
        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        return x

    #number of blocks for speed
    #stage 1: 32 filters (reduced from 64)
    x = residual_block(x, 32)
    x = residual_block(x, 32)

    #stage 2: 64 filters (reduced from 128)
    x = residual_block(x, 64, stride=2)
    x = residual_block(x, 64)

    #stage 3: 128 filters (reduced from 256)
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128)

    #global average pooling and classification
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def create_mobilenet(input_shape=(32, 32, 3), num_classes=10):
    inputs = Input(shape=input_shape)

    def depthwise_separable_conv(x, filters, stride=1):
        #depthwise convolution
        x = DepthwiseConv2D(3, strides=stride, padding='same',
                            depthwise_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        #pointwise convolution
        x = Conv2D(filters, 1, padding='same',
                   kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    def squeeze_excitation_block(x, ratio=16):
        filters = x.shape[-1]

        #squeeze: Global Average Pooling
        se = GlobalAveragePooling2D()(x)

        #excitation: FC -> ReLU -> FC -> Sigmoid
        se = Dense(max(filters // ratio, 8), activation='relu')(se)  # Ensure minimum neurons
        se = Dense(filters, activation='sigmoid')(se)

        #reshape to match input dimensions for multiplication
        se = Reshape((1, 1, filters))(se)

        #scale the input
        return Multiply()([x, se])

    #initial convolution - large
    x = Conv2D(64, 3, strides=1, padding='same',
               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    #stage 1
    x = depthwise_separable_conv(x, 64)
    x = squeeze_excitation_block(x)
    x = depthwise_separable_conv(x, 64)
    x = squeeze_excitation_block(x)

    #stage 2 - downsample
    x = depthwise_separable_conv(x, 128, stride=2)
    x = squeeze_excitation_block(x)
    x = depthwise_separable_conv(x, 128)
    x = squeeze_excitation_block(x)
    x = depthwise_separable_conv(x, 128)
    x = squeeze_excitation_block(x)

    #stage 3 - downsample
    x = depthwise_separable_conv(x, 256, stride=2)
    x = squeeze_excitation_block(x)
    x = depthwise_separable_conv(x, 256)
    x = squeeze_excitation_block(x)
    x = depthwise_separable_conv(x, 256)
    x = squeeze_excitation_block(x)

    #stage 4
    x = depthwise_separable_conv(x, 512)
    x = squeeze_excitation_block(x)
    x = depthwise_separable_conv(x, 512)
    x = squeeze_excitation_block(x)

    #global pooling and classification
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
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