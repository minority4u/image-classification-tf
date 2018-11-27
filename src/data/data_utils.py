# -*- coding: utf-8 -*-
import click
import logging
import os
import sys
import cv2

sys.path.append(os.path.abspath("."))
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator


from src.utils_io import Console_and_file_logger, ensure_dir
from sklearn.model_selection import train_test_split

global config
global class_names
class_names = []
num_images = 0


def set_config(conf):
    global config
    config = conf


def save_images(X_train, y_train, path):
    for idx, image in enumerate(X_train):
        path_name = os.path.join(path, y_train[idx])
        ensure_dir(path_name)
        file_name = os.path.join(path_name, str(idx) + '.jpg')

        cv2.imwrite(file_name, image)
        logging.debug("Writing: filename: {}".format(file_name))


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


def get_stride(img_size, patch_size):
    x_count = np.ceil(1.25*(img_size[0]/patch_size[0]))
    y_count = np.ceil(1.25*(img_size[1]/patch_size[1]))
    x_stride = patch_size[0] - np.ceil(((patch_size[0] * x_count)-img_size[0]) / (x_count-1))
    y_stride = patch_size[1] - np.ceil(((patch_size[1] * y_count)-img_size[1]) / (y_count-1))
    return int(x_stride), int(y_stride)


def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    # stride x and stride y > 0
    if stepSize[0] > 0 and stepSize[1] > 0:
        for x in range(0, image.shape[0]-windowSize[0]+1, stepSize[0]):
            for y in range(0, image.shape[1]-windowSize[1]+1, stepSize[1]):
                # yield the current window
                yield (x, y, image[x:x + windowSize[0], y:y + windowSize[1]])
    # stride x > 0, stride y == 0
    if stepSize[0] > 0 and stepSize[1] == 0:
        for x in range(0, image.shape[0]-windowSize[0]+1, stepSize[0]):
            # yield the current window
            yield (x, 0, image[x:x + windowSize[0], 0:0 + windowSize[1]])
    # stride x == 0, stride y > 0
    if stepSize[0] == 0 and stepSize[1] > 0:
        for y in range(0, image.shape[1]-windowSize[1]+1, stepSize[1]):
            # yield the current window
            yield (0, y, image[0:0 + windowSize[0], y:y + windowSize[1]])
    # stride x and stride y == 0
    if stepSize[0] == 0 and stepSize[1] == 0:
        yield (0, 0, image[0:0 + windowSize[0], 0:0 + windowSize[1]])


def create_patches(image, slice_width, slice_height):
    # resize image width
    if image.shape[0] < slice_width:
        logging.debug('resize width from: {}'.format(image.shape))
        factor = slice_width/image.shape[0]
        image = cv2.resize(image, (0, 0), fx=factor, fy=factor)
        logging.debug('resize width to: {}'.format(image.shape))
    # resize image height
    if image.shape[1] < slice_height:
        logging.debug('resize height from: {}'.format(image.shape))
        factor = slice_height / image.shape[1]
        image = cv2.resize(image, (0, 0), fx=factor, fy=factor)
        logging.debug('resize height to: {}'.format(image.shape))
    result = []
    stride = get_stride(image.shape, (slice_width, slice_height))
    logging.debug('stride: {}'.format(stride))
    # loop over the sliding window for each layer of the pyramid
    for (y, x, window) in sliding_window(image, stepSize=stride, windowSize=(slice_width, slice_height)):
        # if the window does not meet our desired window size, ignore it
        logging.debug('x: {}, y: {}, window: {}'.format(x, y, window.shape))
        if window.shape[1] != slice_height or window.shape[0] != slice_width:
            continue

        clone = image.copy()

        crop_img = clone[y:y + slice_width, x:x + slice_height]
        result.append(crop_img)

    return result


def get_class_names():
    return ['Etechnik', 'Fliesbilder', 'Lageplan', 'Stahlbau', 'Tabellen']


def load_image(path='data/raw/test/Fliesbilder/image001.jpg'):
    global config
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
    global num_images
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
            num_images += 1
    return images



