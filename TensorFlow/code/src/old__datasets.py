from pathlib import Path
import tensorflow as tf

train_dir = Path("./data/train")
val_dir = Path("./data/val")
test_dir = Path("./data/test")

class_names = sorted(
    d.name for d in train_dir.iterdir()
    if d.is_dir()
)

def create_datasets():

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(64, 64),
        batch_size=32,
        label_mode="int",
        shuffle=True,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=(64, 64),
        batch_size=32,
        label_mode="int",
        shuffle=False,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(64, 64),
        batch_size=32,
        label_mode="int",
        shuffle=False,
    )

    train_ds = train_ds.cache()

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
    ])

    # Apply augmentation to the training dataset using map
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

    # Configure for performance
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds
