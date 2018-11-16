# -*- coding: utf-8 -*-
import click
import logging
import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


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
                                       class_mode):
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


def __get_generator__(image_data_generator, path_to_data, image_size, batch_size, class_mode, subset, shuffle=True):
    """
    Get training or validation generator
    :param image_data_generator: data generator (e.g. augmentation)
    :param path_to_data: Path to data directory. Subfolders describe the classes
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
        subset=subset)
    return train_generator


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())


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
