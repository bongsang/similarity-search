__author__ = "https://www.linkedin.com/in/bongsang/"
__license__ = "MIT"

import os
import random
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image


if __name__ == '__main__':
    # --------------------
    # Arguments setup
    # --------------------
    parser = argparse.ArgumentParser(
        description="Amazon's Geographic Mass Classification (Author: Bongsang Kim)")
    parser.add_argument('--labels', type=list, nargs='+',
                        default=['andesite', 'gneiss', 'marble', 'quartzite', 'rhyolite', 'schist'])
    parser.add_argument('--test_image', action='store_true')
    parser.add_argument('--test_path', default='tests')
    parser.add_argument('--result_path', default='results')
    parser.add_argument('--model_path', default='model')
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
    saved_model_files = []
    model_path = os.path.join('.', args.model_path)
    for filename in os.listdir(model_path):
        if os.path.getsize(os.path.join(model_path, filename)) > 0 and filename.lower().endswith('.h5'):
            saved_model_files.append(filename)

    latest_model_file = saved_model_files[-1]
    reloaded_model = tf.keras.models.load_model(os.path.join(model_path, latest_model_file))
    reloaded_model.summary()
    print(latest_model_file + " saved model loaded successfully!")

    # ----------------------
    # Predicting test images
    # ----------------------
    files = []
    test_path = os.path.join('.', args.test_path)
    for filename in os.listdir(test_path):
        if os.path.getsize(os.path.join(test_path, filename)) > 0 and \
                filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
            files.append(filename)
        else:
            print(filename + " is not image file or abnormal, so ignoring.")

    shuffled_files = random.sample(files, len(files))
    fig_path = os.path.join('.', args.result_path)
    labels = args.labels
    for file in shuffled_files:
        test_image = image.load_img(os.path.join(test_path, file), target_size=(28, 28))
        x = image.img_to_array(test_image)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        classes = reloaded_model.predict(images)
        idx = np.argmax(classes[0])
        result = f"Prediction: {file} is {labels[idx]}"
        print(result)
        plt.imshow(test_image)
        plt.title(result)
        plt.savefig(os.path.join(fig_path, "predicted_"+file))




# ----------------------
# Model testing
# ----------------------
# print("###### Model testing ######")
# files = []
# test_path = join('.', args.test_path)
# for filename in os.listdir(test_path):
#     if os.path.getsize(join(test_path, filename)) > 0 and filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
#         files.append(filename)
#     else:
#         print(filename + " is not image file or abnormal, so ignoring.")
#
# shuffled_files = random.sample(files, len(files))
# for file in shuffled_files:
#     test_image = image.load_img(join(test_path, file), target_size=(150, 150))
#     x = image.img_to_array(test_image)
#     x = np.expand_dims(x, axis=0)
#     images = np.vstack([x])
#     classes = model.predict(images, batch_size=10)
#     print(classes)
#
#     plt.imshow(test_image)
#     plt.title(classes)
#     plt.savefig(join(fig_path, "prediction_"+file))
#
#