__author__ = "https://www.linkedin.com/in/bongsang/"
__license__ = "MIT"

import os
from pathlib import Path
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing import image


if __name__ == '__main__':
    # --------------------
    # Arguments setup
    # --------------------
    parser = argparse.ArgumentParser(
        description="Amazon's Geographic Mass Classification (Author: Bongsang Kim)")
    parser.add_argument('--labels', type=list, nargs='+',
                        default=['andesite', 'gneiss', 'marble', 'quartzite', 'rhyolite', 'schist'])
    parser.add_argument('--test_image', action='store')
    parser.add_argument('--test_path', default='./tests')
    parser.add_argument('--result_path', default='./results')
    parser.add_argument('--model_path', default='./models')
    parser.add_argument('--model_file', action='store')
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

    # --------------------
    # Load saved model
    # --------------------
    if args.model_file is not None:
        model_file = Path(args.model_file)
        print(model_file)
    else:
        saved_model_files = []
        model_path = Path(args.model_path)
        for filename in os.listdir(model_path):
            if os.path.getsize(model_path/filename) > 0 and filename.lower().endswith('.h5'):
                saved_model_files.append(filename)

        model_file = model_path/saved_model_files[-1]

    reloaded_model = tf.keras.models.load_model(model_file)
    reloaded_model.summary()
    print(model_file)
    print("Model loaded successfully!")

    # ----------------------
    # Predicting test images
    # ----------------------
    labels = args.labels
    if args.test_image:
        filename = Path(args.test_image)
        if not filename.exists():
            print(f'Oops! {filename} is not existing!')
        else:
            test_image = image.load_img(filename, target_size=(28, 28))
            x = image.img_to_array(test_image)
            x = np.expand_dims(x, axis=0)
            images = np.vstack([x])
            classes = reloaded_model.predict(images)
            idx = np.argmax(classes[0])
            result = f"Prediction: {filename.name} is similar with {labels[idx]}"
            print(result)
            plt.imshow(test_image)
            plt.title(result)
            plt.savefig(filename.parent / model_file.stem + '_' + filename.name)
    else:
        files = []
        test_path = Path(args.test_path)
        for filename in os.listdir(test_path):
            file_path = test_path / filename
            if os.path.getsize(file_path) > 0 and \
                    filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
                files.append(filename)

        shuffled_files = random.sample(files, len(files))
        result_path = Path(args.result_path)
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        for file_name in shuffled_files:
            file_path = test_path / file_name
            test_image = image.load_img(file_path, target_size=(28, 28))
            x = image.img_to_array(test_image)
            x = np.expand_dims(x, axis=0)
            images = np.vstack([x])
            classes = reloaded_model.predict(images)
            idx = np.argmax(classes[0])
            result = f"Origin : {file_name} >> Prediction: {labels[idx]}"
            print(result)
            plt.imshow(test_image)
            plt.title(result)
            result_file = result_path / f"{model_file.stem}_{file_name}"
            plt.savefig(result_file)
