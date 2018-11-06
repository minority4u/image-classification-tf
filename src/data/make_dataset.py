# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from scipy.misc import imresize


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

def GetAllImages(dirPath):
    datagen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        preprocessing_function=preprocess)

    generator = datagen.flow_from_directory(
        dirPath,
        target_size=(48, 48),
        batch_size=1024,
        shuffle=True,
        class_mode='sparse')

    inputs, targets = next(generator)
    return inputs, targets

def GetTrainGenerator(dirPath):
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        preprocessing_function=preprocess)

    train_generator = train_datagen.flow_from_directory(
        dirPath,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

    return train_generator

def GetValidationGenerator(dirPath):
    test_datagen = ImageDataGenerator(rescale=1. / 255,
                                      preprocessing_function=preprocess)
    validation_generator = test_datagen.flow_from_directory(
        dirPath,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

    return validation_generator

    #model.fit_generator(
    #   train_generator,
    #    samples_per_epoch=2000,
    #    nb_epoch=50,
    #    validation_data=validation_generator,
    #    nb_val_samples=800)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()


def preprocess(img):
    width, height = img.shape[0], img.shape[1]
    img = image.array_to_img(img, scale=False)

    # Crop 48x48px
    desired_width, desired_height = 48, 48

    if width < desired_width:
        desired_width = width
    start_x = np.maximum(0, int((width-desired_width)/2))

    img = img.crop((start_x, np.maximum(0, height-desired_height), start_x+desired_width, height))
    img = img.resize((48, 48))

    img = image.img_to_array(img)
    return img / 255.


