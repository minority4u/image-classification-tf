# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


# Hardcode Config für Training und Validation
def __get_image_data_generator__(validationSplit):
    """
    :param validationSplit: Defines how many images will be in validation set, in % e.g. 0.2 = 20%
    :return: ImageDataGenerator
    """
    data_generator = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rescale=(1. / 255),
        shear_range=0,
        zoom_range=0,
        horizontal_flip=False,
        vertical_flip=False,
        # preprocessing_function=preprocess,
        validation_split=validationSplit)
    return data_generator


def get_train_and_validation_generator(path_to_data, validation_split, image_size, batch_size, class_mode):
    """
    Returns Training and Validation Generator for Keras fit_generator usage
    :param path_to_data: Path to data directory. subfolders describe the classes
    :param validation_split:
    :param image_size:
    :param batch_size:
    :param class_mode:
    :return:
    """
    image_data_generator = __get_image_data_generator__(validation_split)
    train_generator = __get_generator__(image_data_generator, path_to_data, image_size, batch_size, class_mode,
                                        'training')
    validation_generator = __get_generator__(image_data_generator, path_to_data, image_size, batch_size, class_mode,
                                             'validation')
    return train_generator, validation_generator


def __get_all_images__(path_to_data, image_size, batch_size, class_mode):
    data_generator = __get_image_data_generator__()
    generator = data_generator.flow_from_directory(
        path_to_data,
        target_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        class_mode=class_mode)
    inputs, targets = next(generator)
    return inputs, targets


def __get_generator__(image_data_generator, path_to_data, image_size, batch_size, class_mode, subset):
    train_generator = image_data_generator.flow_from_directory(
        path_to_data,
        target_size=image_size,
        batch_size=batch_size,
        shuffle=True,
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
    options = {}

    width, desired_width = img.shape[0]
    height, desired_height = img.shape[1]
    img = image.array_to_img(img, scale=False)

    if ("width" in options & options["width"] < width):
        desired_width = options["width"]

    if ("height" in options & options["height"] < height):
        desired_height = options["height"]

    start_x = np.maximum(0, int((width - desired_width) / 2))

    img = img.crop((start_x, np.maximum(0, height - desired_height), start_x + desired_width, height))
    img = img.resize((48, 48))

    img = image.img_to_array(img)
    img = img / 255.

    return img
