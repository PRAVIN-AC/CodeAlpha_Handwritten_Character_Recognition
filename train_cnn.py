import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_datasets as tfds
import os

def main():
    print("📥 Downloading EMNIST securely via Google TensorFlow Datasets...")
    # Load the dataset from a secure Google mirror to avoid BadZipFile errors
    (ds_train, ds_test), ds_info = tfds.load(
        'emnist/byclass',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    # Preprocessing function to normalize and fix the rotation of EMNIST
    def normalize_img(image, label):
        """Normalizes images: `uint8` -> `float32` and fixes EMNIST rotation."""
        image = tf.cast(image, tf.float32) / 255.0
        # EMNIST images from tfds need to be transposed (rotated) to look normal
        image = tf.transpose(image, perm=[1, 0, 2])
        return image, label

    print("🔄 Preprocessing image matrices...")
    # Optimize data loading pipeline for speed
    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache().shuffle(ds_info.splits['train'].num_examples).batch(256).prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(256).cache().prefetch(tf.data.AUTOTUNE)

    print("🧠 Building the Convolutional Neural Network (CNN)...")
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(62, activation='softmax') # 62 outputs: 0-9, A-Z, a-z
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print("🚀 Training Model (This will take a few minutes)...")
    model.fit(ds_train, epochs=5, validation_data=ds_test)

    os.makedirs('models', exist_ok=True)
    model.save('models/emnist_cnn_model.h5')
    print("\n✅ Model successfully trained and saved to 'models/emnist_cnn_model.h5'")

if __name__ == "__main__":
    main()