__author__ = "https://www.linkedin.com/in/bongsang/"
__license__ = "MIT"
from utils import download

import os
from os.path import join
import zipfile
import random
from shutil import copytree
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import Callback


def data_generator(dataset_train_path, dataset_validation_path, augmentation=True):
    if augmentation:
        # Reducing over-fitting by various augmentation
        train_data_generator = image.ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

        valid_data_generator = image.ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')
    else:
        train_data_generator = image.ImageDataGenerator(rescale=1.0 / 255)
        valid_data_generator = image.ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_data_generator.flow_from_directory(
        dataset_train_path,
        class_mode="categorical",
        target_size=(150, 150))

    validation_generator = valid_data_generator.flow_from_directory(
        dataset_validation_path,
        class_mode="categorical",
        target_size=(150, 150)
    )

    return train_generator, validation_generator


def model_design():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    return model


class AccCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        elif logs.get('acc') > 0.98:
            self.model.stop_training = True
            print(f"\nEarly stopping! Epoch: {epoch} \t Accuracy: {round(logs.get('acc') * 100)}%")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="ex) python main.py --mode=train --epochs=100 --url=\"http://...\"")
    parser.add_argument('--mode', default='train', choices=['train', 'test'], required=False)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--url', default="http://aws-proserve-data-science.s3.amazonaws.com/geological_similarity.zip")
    parser.add_argument('--download_path', default='download')
    parser.add_argument('--dataset_path', default='dataset')
    parser.add_argument('--split_rate', type=float, default=1.)
    parser.add_argument('--test_path', default='tests')
    parser.add_argument('--result_path', default='results')
    args = parser.parse_args()

    # Tensorflow GPU setup
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    # ----------------------------------------
    # Download data from URL and setup dataset
    # ----------------------------------------
    download_path = join('.', args.download_path)
    download_train_path = join('.', download_path, 'rps')
    download_validation_path = join('.', download_path, 'rps-test-set')
    if not os.path.exists(download_path):
        os.makedirs(download_train_path)
        os.makedirs(download_validation_path)

        urls = [args.url_train, args.url_test]
        for url in urls:
            file = download.from_url(url, download_path)
            zfile = zipfile.ZipFile(file, 'r')
            zfile.extractall(download_path)
            zfile.close()

    dataset_path = join('.', args.dataset_path)
    dataset_train_path = join(dataset_path, 'train')
    dataset_validation_path = join(dataset_path, 'validation')

    if not os.path.exists(dataset_path):
        copytree(download_train_path, dataset_train_path)
        copytree(download_validation_path, dataset_validation_path)

    # ---------------
    # model designing
    # ---------------
    model = model_design()
    model.compile(
        optimizer=RMSprop(lr=1e-3),
        loss=CategoricalCrossentropy(),
        metrics=['acc'])

    model.summary()

    # -------------
    # model fitting
    # -------------
    print(f"args.mode={args.mode}")


    callback = AccCallback()
    train_generator, validation_generator = data_generator(dataset_train_path, dataset_validation_path)

    history = model.fit_generator(
        train_generator,
        epochs=args.epochs,
        validation_data=validation_generator,
        # callbacks=[callback],
        verbose=2
    )

    # ----------------------
    # Train history plotting
    # ----------------------
    print("###### Model Plotting ######")

    fig_path = join(".", args.result_path)
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))
    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig(join(fig_path, "accuracy.jpg"))

    plt.figure()
    plt.plot(epochs, loss, 'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(join(fig_path, "loss.jpg"))

    # ----------------------
    # Model testing
    # ----------------------
    print("###### Model testing ######")
    files = []
    test_path = join('.', args.test_path)
    for filename in os.listdir(test_path):
        if os.path.getsize(join(test_path, filename)) > 0 and filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
            files.append(filename)
        else:
            print(filename + " is not image file or abnormal, so ignoring.")

    shuffled_files = random.sample(files, len(files))
    for file in shuffled_files:
        test_image = image.load_img(join(test_path, file), target_size=(150, 150))
        x = image.img_to_array(test_image)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        classes = model.predict(images, batch_size=10)
        print(classes)

        plt.imshow(test_image)
        plt.title(classes)
        plt.savefig(join(fig_path, "prediction_"+file))



