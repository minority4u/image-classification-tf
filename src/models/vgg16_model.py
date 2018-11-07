import keras
from keras.layers.convolutional import Convolution2D, Conv2D
# old keras API 1.0 layers
# from keras.layers.convolutional import Convolution2D,
# from keras.engine.topology import merge
# from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.layers import Input, MaxPooling2D
from keras.models import Model
from keras.regularizers import *
from keras.layers.merge import concatenate
import logging
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

from keras.applications.vgg16 import VGG16

def get_model():
    """
    Create a keras model
    :return: aliases, model
    """
    aliases = {}

    # LOAD VGG16
    # Generate a model with all layers (with top)
    vgg16 = VGG16(weights='imagenet', include_top=True)

    # Add a layer where input is the output of the  second last layer
    x = Dense(3, activation='softmax', name='predictions')(vgg16.layers[-2].output)

    # Then create the corresponding model
    model = Model(input=vgg16.input, output=x)
    model.summary()

    return aliases, model


from keras.optimizers import *


def get_optimizer():
    """
    Central optimizer definition
    :return: e.g.: Adadelta()
    """
    return Adadelta()


def is_custom_loss_function():
    return False


def get_loss_function():
    """
    Define a loss function,
    currently handeled by the config-file
    :return: e.g.: 'categorical_crossentropy'
    """
    return 'categorical_crossentropy'


def get_batch_size():
    return 8


def get_num_epoch():
    return 1000


def get_data_config():
    return '{"kfold": 1, "numPorts": 1, "samples": {"validation": 450, "training": 2100, "split": 3, "test": 450}, "datasetLoadOption": "batch", "mapping": {"Filename": {"port": "InputPort0", "type": "Image", "shape": "", "options": {"horizontal_flip": false, "Height": "224", "rotation_range": 0, "vertical_flip": false, "width_shift_range": 0, "Normalization": false, "Width": "224", "shear_range": 0, "pretrained": "None", "Scaling": 1, "Augmentation": false, "Resize": true, "height_shift_range": 0}}, "Label": {"port": "OutputPort0", "type": "Categorical", "shape": "", "options": {}}}, "dataset": {"samples": 3000, "name": "Classify1000", "type": "private"}, "shuffle": true}'
