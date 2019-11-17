__author__ = "https://www.linkedin.com/in/bongsang/"
__license__ = "MIT"

from utils import download
import os
from os.path import join
from pathlib import Path
import zipfile
import argparse
import glob
from tqdm import tqdm
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import Callback


def image_data_generator(dataset_train_path, dataset_validation_path, labels, augmentation=True):
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
        target_size=(28, 28),
        classes=labels,
        class_mode="sparse",
        batch_size=128,
        shuffle=True)

    validation_generator = valid_data_generator.flow_from_directory(
        dataset_validation_path,
        target_size=(28, 28),
        classes=labels,
        class_mode="sparse",
        batch_size=128)

    return train_generator, validation_generator


class AccCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        elif logs.get('acc') > 0.99:
            self.model.stop_training = True
            print(f"\nEarly stopping at epoch: {epoch} \t Accuracy: {round(logs.get('acc')*100)}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Amazon's Geographic Mass Classification (Author: Bongsang Kim)")
    parser.add_argument('--mode', default='train', choices=['train', 'test'], required=False)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--url', default="http://aws-proserve-data-science.s3.amazonaws.com/geological_similarity.zip")
    parser.add_argument('--labels', type=list, nargs='+', default=['andesite', 'gneiss', 'marble', 'quartzite', 'rhyolite', 'schist'])
    parser.add_argument('--download_path', default='./download')
    parser.add_argument('--download_data_path', default='geological_similarity')
    parser.add_argument('--dataset_path', default='./dataset')
    parser.add_argument('--test_path', default='./tests')
    parser.add_argument('--model_path', default='./models')
    parser.add_argument('--split_rate', type=float, default=0.9)
    parser.add_argument('--test_num', type=int, default=2)
    parser.add_argument('--result_path', default='results')
    args = parser.parse_args()

    # --------------------
    # Tensorflow GPU setup
    # --------------------
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
    print(args.labels)
    download_path = Path(args.download_path)
    dataset_path = Path(args.dataset_path)
    test_path = Path(args.test_path)
    dataset_train_path = dataset_path / 'train'
    dataset_validation_path = dataset_path / 'validation'

    if not os.path.exists(download_path):
        os.makedirs(download_path)
        file = download.from_url(args.url, download_path)
        zfile = zipfile.ZipFile(file, 'r')
        zfile.extractall(download_path)
        zfile.close()

    if not os.path.exists(test_path):
        os.makedirs(test_path)

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

        for label in args.labels:
            download_data_label_path = os.path.join(download_path, args.download_data_path, label)
            label_images = glob.glob(download_data_label_path + '/*.jpg')
            print(f"{label}: total {len(label_images)} Images")
            num_train = int(round(len(label_images) * args.split_rate))

            # For testing
            test_images = label_images[:args.test_num]
            for idx, source in enumerate(test_images):
                destination = test_path / f'{label}_test_{idx}.jpg'
                print(source)
                print(destination)
                shutil.copy(source, destination)

            # For training
            train_images, validation_images = label_images[args.test_num:num_train], label_images[num_train:]
            for source_image in tqdm(train_images):
                destination_path = os.path.join(dataset_train_path, label)
                if not os.path.exists(destination_path):
                    os.makedirs(destination_path)
                shutil.copy(source_image, destination_path)

            # For validating
            for source_image in tqdm(validation_images):
                destination_path = os.path.join(dataset_validation_path, label)
                if not os.path.exists(destination_path):
                    os.makedirs(destination_path)
                shutil.copy(source_image, destination_path)

    # ---------------
    # model designing
    # ---------------
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(rate=0.2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Dense(len(args.labels), activation='softmax')
    ])
    model.compile(
        optimizer=Adam(lr=1e-3),
        loss=SparseCategoricalCrossentropy(),
        metrics=['acc'])
    print(f'classes : {len(args.labels)} labels')
    model.summary()

    # -------------
    # model fitting
    # -------------
    callback = AccCallback()
    train_generator, validation_generator = \
        image_data_generator(dataset_train_path, dataset_validation_path, args.labels)

    print("###### Model Pitting ######")
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=int(np.ceil(train_generator.n / float(args.batch_size))),
        epochs=args.epochs,
        validation_data=validation_generator,
        validation_steps=int(np.ceil(validation_generator.n / float(args.batch_size))),
        verbose=2)

    # ------------
    # Model Saving
    # ------------
    print("###### Model Saving ######")
    timestamp = int(time.time())
    model_path = Path(args.model_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    file_path = model_path / f"model-{timestamp}.h5"
    model.save(file_path)
    print(file_path.name + " saved successfully!")


    # ----------------------
    # Train history plotting
    # ----------------------
    print("###### Model Plotting ######")
    result_path = Path(args.result_path)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))
    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig(result_path / f"model-{timestamp}_acc_history.jpg")

    plt.figure()
    plt.plot(epochs, loss, 'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(result_path / f"model-{timestamp}_loss_history.jpg")
