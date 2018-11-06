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
def __GetImageDataGenerator__(validationSplit = 0.2):
    datagen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rescale=(1. / 255),
        shear_range=0,
        zoom_range=0,
        horizontal_flip=False,
        vertical_flip=False,
        #preprocessing_function=preprocess,
        validation_split=validationSplit)
    return datagen

def GetTrainAndValidationGenerator(pathToData, imageSize = (224,224), batchSize =  32, classmode = 'categorical'):
    imageDataGenerator = __GetImageDataGenerator__()
    train_generator = __GetTrainGenerator__(imageDataGenerator, pathToData, imageSize, batchSize, classmode)
    validation_generator = __GetValidationGenerator__(imageDataGenerator, pathToData, imageSize, batchSize, classmode)
    return train_generator, validation_generator

def __GetAllImages__(pathToData, imageSize = (224,224), batchSize =  32, classmode = 'categorical'):
    datagen = __GetImageDataGenerator__()
    generator = datagen.flow_from_directory(
        pathToData,
        target_size=imageSize,
        batch_size=batchSize,
        shuffle=True,
        class_mode=classmode)
    inputs, targets = next(generator)
    return inputs, targets

def __GetTrainGenerator__(imageDataGenerator,pathToData, imageSize = (224, 224), batchSize = 32, classmode = 'categorical'):
    train_generator = imageDataGenerator.flow_from_directory(
        pathToData,
        target_size=imageSize,
        batch_size=batchSize,
        shuffle=True,
        class_mode=classmode,
        subset='training')
    return train_generator

def __GetValidationGenerator__(imageDataGenerator, pathToData, imageSize = (224, 224), batchSize = 32, classmode = 'categorical'):
    validation_generator = imageDataGenerator.flow_from_directory(
        pathToData,
        target_size=imageSize,
        batch_size=batchSize,
        shuffle=True,
        class_mode=classmode,
        subset='validation')
    return validation_generator


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()