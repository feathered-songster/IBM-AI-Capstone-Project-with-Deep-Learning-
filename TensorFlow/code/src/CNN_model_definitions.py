import tensorflow as tf
from tensorflow.keras import layers, Sequential


IMG_SIZE = 64
CHANNELS = 3

def create_medium_model(num_classes =2):
    return Sequential([
    layers.Input(shape=(IMG_SIZE , IMG_SIZE , CHANNELS)),
    # Conv Block 1: 64x64 -> 32x32
    layers.Conv2D(
        32,
        kernel_size=3,
        strides=1,
        padding="same",
        kernel_initializer=tf.keras.initializers.HeNormal()
    ),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D(pool_size=2),

    # Conv Block 2: 32x32 -> 16x16
    layers.Conv2D(
        64,
        kernel_size=3,
        strides=1,
        padding="same",
        kernel_initializer=tf.keras.initializers.HeNormal()
    ),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D(pool_size=2),

    # Conv Block 3: 16x16 -> 8x8
    layers.Conv2D(
        128,
        kernel_size=3,
        strides=1,
        padding="same",
        kernel_initializer=tf.keras.initializers.HeNormal()
    ),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D(pool_size=2),

    # Conv Block 4: 8x8 -> 4x4
    layers.Conv2D(
        256,
        kernel_size=3,
        strides=1,
        padding="same",
        kernel_initializer=tf.keras.initializers.HeNormal()
    ),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D(pool_size=2),

    # Classifier
    layers.GlobalAveragePooling2D(),
    layers.Dense(
        128,
        kernel_initializer=tf.keras.initializers.HeNormal()
    ),
    layers.ReLU(),
    layers.Dropout(0.4),
    layers.Dense(num_classes)
])

def create_small_model(num_classes=2):
    return Sequential([
    layers.Input(shape=(IMG_SIZE , IMG_SIZE , CHANNELS)),
    # Conv Block 1: 64x64 -> 32x32
    layers.Conv2D(
        16,
        kernel_size=3,
        strides=1,
        padding="same",
        kernel_initializer=tf.keras.initializers.HeNormal()
    ),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D(pool_size=2),

    # Conv Block 2: 32x32 -> 16x16
    layers.Conv2D(
        32,
        kernel_size=3,
        strides=1,
        padding="same",
        kernel_initializer=tf.keras.initializers.HeNormal()
    ),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D(pool_size=2),

    # Conv Block 3: 16x16 -> 8x8
    layers.Conv2D(
        64,
        kernel_size=3,
        strides=1,
        padding="same",
        kernel_initializer=tf.keras.initializers.HeNormal()
    ),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D(pool_size=2),

    # Conv Block 4: 8x8 -> 4x4
    layers.Conv2D(
        128,
        kernel_size=3,
        strides=1,
        padding="same",
        kernel_initializer=tf.keras.initializers.HeNormal()
    ),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D(pool_size=2),

    # Classifier
    layers.GlobalAveragePooling2D(),
    layers.Dense(
        64,
        kernel_initializer=tf.keras.initializers.HeNormal()
    ),
    layers.ReLU(),
    layers.Dropout(0.4),
    layers.Dense(num_classes)
])
