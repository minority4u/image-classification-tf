# -*- coding: utf-8 -*-
import click
import logging
import os
import sys
import yaml
import json
import cv2

sys.path.append(os.path.abspath("."))
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from argparse import ArgumentParser

from src.utils_io import Console_and_file_logger, ensure_dir
from sklearn.model_selection import train_test_split

global config
global class_names
class_names = []


def save_images(X_train, y_train, path):
    for idx, image in enumerate(X_train):
        path_name = os.path.join(path, y_train[idx])
        ensure_dir(path_name)
        file_name = os.path.join(path_name, str(idx) + '.jpg')

        cv2.imwrite(file_name, image)
        logging.debug("Writing: filename: {}".format(file_name))


def split_dataset(src_path, dest_path):
    # load all images
    images = load_all_images(src_path)

    X = []
    y = []

    # transform image shapes
    for label, images in images:
        for image in images:
            X.append(image)
            y.append(label)

    # split images per class
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    # write split images to disk
    save_images(X_train, y_train, os.path.join(dest_path, '/train'))
    save_images(X_test, y_test, os.path.join(dest_path, '/test'))


def __get_image_data_generator__(validation_split):
    """
    Gets Image Data Generator, scales and augments images
    :param validation_split: Defines how many images will be in validation set, in % e.g. '0.2' = 20%
    :return: ImageDataGenerator
    """
    data_generator = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rescale=(1. / 255),
        shear_range=0,
        zoom_range=0.2,
        horizontal_flip=False,
        vertical_flip=False,
        # preprocessing_function=__preprocess__,
        validation_split=validation_split)
    return data_generator


def get_train_and_validation_generator(path_to_data, validation_split, image_size, batch_size_train, batch_size_val,
                                       class_mode, color_mode):
    """
    Returns Training and Validation Generator for Keras fit_generator usage
    :param path_to_data: Path to data directory. Subfolders describe the classes
    :param validation_split: Defines how many images will be in validation set, in % e.g. '0.2' = 20%
    :param image_size: Input size of images (e.g. '(224, 224)' for VGG16/19)
    :param batch_size: Defines how many images will be loaded for each batch
    :param class_mode: e.g. "binary", 'categorical', 'sparse'
    :return: Returns 2 generators, training and validation
    """
    image_data_generator = __get_image_data_generator__(validation_split)
    train_generator = __get_generator__(image_data_generator, path_to_data, image_size, batch_size_train, class_mode,
                                        'training', shuffle=True)
    validation_generator = __get_generator__(image_data_generator, path_to_data, image_size, batch_size_val, class_mode,
                                             'validation', shuffle=False)
    return train_generator, validation_generator


def __get_all_images__(path_to_data, image_size, batch_size, class_mode):
    """
    Returns all images with groundtruth, not splitted in training and validation
    :param path_to_data: Path to data directory. Subfolders describe the classes
    :param image_size: Input size of images (e.g. '(224, 224)' for VGG16/19)
    :param batch_size: Defines how many images will be loaded for each batch
    :param class_mode: e.g. "binary", 'categorical', 'sparse'
    :return: (images, true class)
    """
    data_generator = __get_image_data_generator__()
    generator = data_generator.flow_from_directory(
        path_to_data,
        target_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        class_mode=class_mode)
    inputs, targets = next(generator)
    return inputs, targets


def __get_generator__(image_data_generator, path_to_data, image_size, batch_size, class_mode, subset, shuffle=True, color_mode='rgb'):
    """
    Get training or validation generator
    :param image_data_generator: data generator (e.g. augmentation)
    :param path_to_data: Pa     th to data directory. Subfolders describe the classes
    :param image_size: Input size of images (e.g. '(224, 224)' for VGG16/19)
    :param batch_size: Defines how many images will be loaded for each batch
    :param class_mode: e.g. "binary", 'categorical', 'sparse'
    :param subset: Defines whether training or validation
    :return: Keras Data Generator
    """
    train_generator = image_data_generator.flow_from_directory(
        path_to_data,
        target_size=image_size,
        batch_size=batch_size,
        shuffle=shuffle,
        class_mode=class_mode,
        subset=subset,
        color_mode=color_mode)
    return train_generator


def preprocess(img):
    """
    preprocesses an Image with several options
    :param img: Image (type: keras_preprocessing.image)
    :return: preprocessed Image (type: keras_preprocessing.image)
    """

    # define preprocess options
    # possible options: width, height,
    options = {}

    # CROPPING
    cropping = False

    # check for cropping options
    if 'width' in options and options['width'] < img.shape[0]:
        desired_width = options['width']
        cropping = True
    else:
        desired_width = img.shape[0]

    if 'height' in options and options['height'] < img.shape[1]:
        desired_height = options['height']
        cropping = True
    else:
        desired_height = img.shape[1]

    # crop if necessary
    if (cropping):
        # initialise width and height from img shape
        width = img.shape[0]
        height = img.shape[1]
        img = image.array_to_img(img, scale=False)

        left = np.max(0, int((width - desired_width) / 2))
        bottom = np.max(0, int((height - desired_height) / 2))

        # crop and resize img
        img = img.crop((left, bottom + desired_height, left + desired_width, bottom))
        img = img.resize((desired_width, desired_height))

        # normalize img
        img = image.img_to_array(img)
        img = img / 255.

    return img


def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def create_patches(image, slice_width, slice_height):
    (winW, winH) = (slice_width, slice_height)
    result = []
    # loop over the image pyramid
    # for resized in pyramid(image, scale=100):
    # loop over the sliding window for each layer of the pyramid
    for (x, y, window) in sliding_window(image, stepSize=400, windowSize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        clone = image.copy()

        crop_img = clone[y:y + winW, x:x + winH]
        result.append(crop_img)

    return result


def get_class_names():
    return ['Etechnik', 'Fliesbilder', 'Lageplan', 'Stahlbau', 'Tabellen']


def load_image(path='data/raw/test/Fliesbilder/image001.jpg'):
    if config['color_mode'] == 'grayscale':
        return cv2.imread(path, 0)
    elif config['color_mode'] == 'rgb':
        return cv2.imread(path)

def load_all_images(path_to_folder='data/raw/test/'):
    """
    recursive function, if it is called with path to testfiles
    it calls itself and takes the folder name as class name
    otherwise it loads all images from the class folder.

    folder should have this structure:
    root
        Classname
            Images (jpg)
    :param path_to_folder: e.g. /data/raw/test/
    :return: list of tuples (label, list of images)
    """

    # logging.debug('load_all_images')
    images = []
    for file in sorted(os.listdir(path_to_folder)):
        current_file = os.path.join(path_to_folder, file)
        if os.path.isdir(current_file):
            # class / label found
            class_names.append(file)
            images.append((file, load_all_images(current_file)))
        # logging.debug('current file: {}'.format(current_file))
        filename, file_extension = os.path.splitext(current_file)
        if file_extension == '.jpg':
            images.append(load_image(current_file))
    return images


if __name__ == '__main__':
    # Define argument parser
    parser = ArgumentParser()
    Console_and_file_logger('Predict_model', log_lvl=logging.DEBUG)

    # define arguments and default values to parse
    # define tha path to your config file
    parser.add_argument("--config", "-c", help="Define the path to config.yml",
                        default="config/experiments/inception_v3_base.yml", required=False)

    parser.add_argument("--working_dir", help="Define the absolute path to the project root",
                        default="../../", required=False)
    # parser.add_argument("--modelskiptraining", help="Skip Training", default="None", required=False)

    args = parser.parse_args()
    logging.debug(args.config)
    # Make sure the config exists
    assert os.path.exists(
        args.config), "Config does not exist {}!, Please create a config.yml in root or set the path with --config.".format(
        args.config)

    # Load config and other global objects
    config = yaml.load(open(args.config, "r"))
    logging.debug(json.dumps(config, indent=2))

    source_root = 'data/raw/classification_data/'
    destination_root = 'data/processed/split/'

    split_dataset(source_root, destination_root)
