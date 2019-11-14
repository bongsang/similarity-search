__author__ = "https://www.linkedin.com/in/bongsang/"
__license__ = "MIT"
import os
import random
from os.path import join
from os.path import getsize
from shutil import copyfile


def split(source_path, train_path, test_path, split_rate):
    files = []
    for filename in os.listdir(source_path):
        if getsize(join(source_path, filename)) > 0 and filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
            files.append(filename)
        else:
            print(filename + " is not image file or abnormal, so ignoring.")

    train_size = int(len(files) * split_rate)
    test_size = int(len(files) - train_size)
    shuffled_set = random.sample(files, len(files))
    train_set = shuffled_set[0:train_size]
    test_set = shuffled_set[-test_size:]

    for filename in train_set:
        source = join(source_path, filename)
        destination = join(train_path, filename)
        print(source)
        print(destination)
        copyfile(source, destination)

    for filename in test_set:
        source = join(source_path, filename)
        destination = join(test_path, filename)
        copyfile(source, destination)


def test(path):
    files = []
    for filename in os.listdir(path):
        if getsize(join(path, filename)) > 0 and filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
            files.append(filename)
        else:
            print(filename + " is not image file or abnormal, so ignoring.")

    shuffled_files = random.sample(files, len(files))

    return shuffled_files
