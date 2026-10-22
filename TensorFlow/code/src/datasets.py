
from pathlib import Path
import tensorflow as tf

train_dir = Path("./data/train")
val_dir = Path("./data/val")
test_dir = Path("./data/test")

class_names = sorted(
    d.name for d in train_dir.iterdir()
    if d.is_dir()
)

IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32


def preprocess_images(images, labels):
    # Convert [0, 255] pixels to [-1, 1]
    images = tf.cast(images, tf.float32) / 255.0
    images = (images - 0.5) / 0.5
    return images, labels


def create_datasets():

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=True,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )

    # Prefetch for performance
    AUTOTUNE = tf.data.AUTOTUNE

    # Normalize all datasets consistently
    train_ds = train_ds.map(
        preprocess_images,
        num_parallel_calls=AUTOTUNE,
    )
    val_ds = val_ds.map(
        preprocess_images,
        num_parallel_calls=AUTOTUNE,
    )
    test_ds = test_ds.map(
        preprocess_images,
        num_parallel_calls=AUTOTUNE,
    )

    # Cache the normalized datasets
    train_ds = train_ds.cache()
    val_ds = val_ds.cache()
    test_ds = test_ds.cache()

    # Augmentation only for training
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
    ])

    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=AUTOTUNE,
    )

    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds
