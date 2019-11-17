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
    parser.add_argument('--train_path', default='./dataset/train')
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

    # Generator for our similarity work
    test_data_generator = image.ImageDataGenerator(rescale = 1./255)
    train_path = Path(args.train_path)
    similarity_generator = test_data_generator.flow_from_directory(
        train_path,
        target_size=(28, 28),
        batch_size=1,
        shuffle=False,
        class_mode='sparse')

    layer_name = 'dense_1'
    intermediate_layer_model = tf.keras.models.Model(
        inputs=reloaded_model.input,
        outputs=reloaded_model.get_layer(layer_name).output)

    intermediate_output = intermediate_layer_model.predict_generator(
        generator=similarity_generator,
        steps=128)

    plt.figure(figsize=(10, 10))
    plt.scatter(intermediate_output[:, 0], intermediate_output[:, 1], cmap='brg')
    plt.colorbar()
    plt.show()

    print(intermediate_output.shape)

    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import normalize

    # base similarity matrix (all dot products)
    # replace this with A.dot(A.T).toarray() for sparse representation
    similarity = np.dot(intermediate_output, intermediate_output.T)

    # squared magnitude of preference vectors (number of occurrences)
    square_mag = np.diag(similarity)

    # inverse squared magnitude
    inv_square_mag = 1 / square_mag

    # if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
    inv_square_mag[np.isinf(inv_square_mag)] = 0

    # inverse of the magnitude
    inv_mag = np.sqrt(inv_square_mag)

    # cosine similarity (elementwise multiply by inverse magnitudes)
    cosine = similarity * inv_mag
    cosine = cosine.T * inv_mag

    origin = 13
    print(similarity_generator.filenames[origin])
    similarity = np.argsort(cosine[origin])[::-1][:5]

    file_path = Path('./dataset/train/', similarity_generator.filenames[origin])
    print(file_path)
    img = image.load_img(file_path, target_size=(28, 28))
    plt.title(f"Original! {origin} : {file_path.name}")
    plt.imshow(img)
    plt.show()


    for idx in similarity:
        file_path = Path('./dataset/train/', similarity_generator.filenames[idx])
        img = image.load_img(file_path, target_size=(28, 28))
        plt.imshow(img)
        plt.title(f"{idx} : {file_path.name}")
        plt.show()
