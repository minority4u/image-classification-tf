from keras.layers.convolutional import Convolution2D, Conv2D
from keras.layers.core import Flatten, Dropout, Dense
from keras.layers import Input, MaxPooling2D
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras import regularizers
from keras.layers.merge import concatenate
import logging


def get_model(config):
    """
    Entry point for model creation
    Returns different models depending on the config
    :param config:
    :return:
    """
    architecture = config['model_architecture']

    if architecture == 'inception_v3':
        return get_inception_v3_model((config['input_image_height'],config['input_image_width'], config['input_image_depth']))

    if architecture == 'inception_v3_small':
        return get_inception_v3_model_small((config['input_image_height'],config['input_image_width'], config['input_image_depth']))

    if architecture == 'inception_v3_big':
        return get_inception_v3_model_big((config['input_image_height'],config['input_image_width'], config['input_image_depth']))

    if architecture == 'inception_v3_reg':
        return get_inception_v3_model_l2((config['input_image_height'],config['input_image_width'], config['input_image_depth']),
                                         regularization_factor=config['regularization_factor'])
    if architecture == 'inception_v3_drop':
        return get_inception_v3_model_dropout((config['input_image_height'],config['input_image_width'], config['input_image_depth']),
                                              dropout_rate=config['dropout_rate'])
    if architecture == 'vgg16':
        return get_VGG_model()
    if architecture == 'vgg16_imagenet_include_top_false':
        return get_VGG_model_imagenet_custom_top()

    if architecture == 'vgg16_imagenet_include_top_false_SGD':
        return get_VGG_model_imagenet_custom_top_SGD()

    if architecture == 'inception_resnet_v2':
        return get_InceptionResNetV2_model()


    return get_inception_v3_model()


def get_inception_v3_model(input_shape=(224, 224, 3)):
    """
    Create a keras model
    :return: aliases, model
    """
    aliases = {}
    Input_1 = Input(shape=input_shape, name='Input_1')

    Convolution2D_236 = Conv2D(name="Convolution2D_236", activation="relu", kernel_size=(3, 3), filters=32,
                               strides=(2, 2), padding="same")(Input_1)
    Convolution2D_235 = Conv2D(name="Convolution2D_235", activation="relu", kernel_size=(3, 3), filters=32,
                               padding="same")(Convolution2D_236)
    Convolution2D_237 = Conv2D(name="Convolution2D_237", activation="relu", kernel_size=(3, 3), filters=64)(
        Convolution2D_235)

    MaxPooling2D_69 = MaxPooling2D(name="MaxPooling2D_69", strides=(2, 2), pool_size=(3, 3), padding="same")(
        Convolution2D_237)

    Convolution2D_238 = Conv2D(name="Convolution2D_238", activation="relu", kernel_size=(3, 3), filters=80)(
        MaxPooling2D_69)
    Convolution2D_239 = Conv2D(name="Convolution2D_239", activation="relu", kernel_size=(3, 3), filters=192,
                               strides=(2, 2))(Convolution2D_238)

    MaxPooling2D_7 = MaxPooling2D(name="MaxPooling2D_7", strides=(2, 2), pool_size=(3, 3), padding="same")(
        Convolution2D_239)

    Convolution2D_10 = Conv2D(name="Convolution2D_10", activation="relu", kernel_size=(1, 1), filters=64,
                              padding="same")(MaxPooling2D_7)
    Convolution2D_3 = Conv2D(name="Convolution2D_3", activation="relu", kernel_size=(1, 1), filters=32, strides=(2, 2))(
        Convolution2D_239)
    Convolution2D_1 = Conv2D(name="Convolution2D_1", activation="relu", kernel_size=(1, 1), filters=32, padding="same")(
        Convolution2D_239)
    Convolution2D_4 = Conv2D(name="Convolution2D_4", activation="relu", kernel_size=(3, 1), filters=64, padding="same")(
        Convolution2D_1)
    Convolution2D_6 = Conv2D(name="Convolution2D_6", activation="relu", kernel_size=(1, 3), filters=64, padding="same")(
        Convolution2D_4)
    Convolution2D_2 = Conv2D(name="Convolution2D_2", activation="relu", kernel_size=(1, 1), filters=32)(
        Convolution2D_239)
    Convolution2D_8 = Conv2D(name="Convolution2D_8", activation="relu", kernel_size=(3, 1), filters=64, strides=(2, 1),
                             padding="same")(Convolution2D_2)
    Convolution2D_9 = Conv2D(name="Convolution2D_9", activation="relu", kernel_size=(1, 3), filters=64, strides=(1, 2),
                             padding="same")(Convolution2D_8)
    Convolution2D_5 = Conv2D(name="Convolution2D_5", activation="relu", kernel_size=(3, 1), filters=64, strides=(2, 1),
                             padding="same")(Convolution2D_6)
    Convolution2D_7 = Conv2D(name="Convolution2D_7", activation="relu", kernel_size=(1, 3), filters=64, strides=(1, 2),
                             padding="same")(Convolution2D_5)

    merge_1 = concatenate(inputs=[Convolution2D_10, Convolution2D_9, Convolution2D_7, Convolution2D_3], name='merge_1',
                          axis=-1)

    Convolution2D_130 = Conv2D(name="Convolution2D_130", activation="relu", kernel_size=(1, 1), filters=128,
                               strides=(2, 2))(merge_1)
    MaxPooling2D_15 = MaxPooling2D(name="MaxPooling2D_15", strides=(2, 2), pool_size=(3, 3), padding="same")(merge_1)
    Convolution2D_129 = Conv2D(name="Convolution2D_129", activation="relu", kernel_size=(1, 1), filters=128,
                               padding="same")(MaxPooling2D_15)
    Convolution2D_126 = Conv2D(name="Convolution2D_126", activation="relu", kernel_size=(1, 1), filters=128)(merge_1)
    Convolution2D_127 = Conv2D(name="Convolution2D_127", activation="relu", kernel_size=(3, 1), filters=128,
                               strides=(2, 1), padding="same")(Convolution2D_126)
    Convolution2D_128 = Conv2D(name="Convolution2D_128", activation="relu", kernel_size=(1, 3), filters=128,
                               strides=(1, 2), padding="same")(Convolution2D_127)
    Convolution2D_121 = Conv2D(name="Convolution2D_121", activation="relu", kernel_size=(1, 1), filters=128,
                               padding="same")(merge_1)
    Convolution2D_122 = Conv2D(name="Convolution2D_122", activation="relu", kernel_size=(3, 1), filters=128,
                               padding="same")(Convolution2D_121)
    Convolution2D_123 = Conv2D(name="Convolution2D_123", activation="relu", kernel_size=(1, 3), filters=128,
                               padding="same")(Convolution2D_122)
    Convolution2D_124 = Conv2D(name="Convolution2D_124", activation="relu", kernel_size=(3, 1), filters=128,
                               strides=(2, 1), padding="same")(Convolution2D_123)
    Convolution2D_125 = Conv2D(name="Convolution2D_125", activation="relu", kernel_size=(1, 3), filters=128,
                               strides=(1, 2), padding="same")(Convolution2D_124)

    merge_10 = concatenate(inputs=[Convolution2D_128, Convolution2D_129, Convolution2D_125, Convolution2D_130],
                           name='merge_10', axis=-1)

    MaxPooling2D_70 = MaxPooling2D(name='MaxPooling2D_70', pool_size=(7, 7))(merge_10)

    Flatten_5 = Flatten(name='Flatten_5')(MaxPooling2D_70)
    Dense_14 = Dense(name="Dense_14", activation="linear", units=5)(Flatten_5)

    Dense_2 = Dense(name="Dense_2", activation="softmax", units=5)(Dense_14)

    model = Model([Input_1], [Dense_2])
    model._make_predict_function()

    for l in model.layers:
        logging.debug('Layer Shape: {} {}'.format(l.name, l.output_shape))

    logging.info(model.summary())
    return aliases, model


def get_inception_v3_model_big(input_shape=(224, 224, 3)):
    """
    Create a keras model
    :return: aliases, model
    """
    aliases = {}
    Input_1 = Input(shape=input_shape, name='Input_1')

    Convolution2D_236 = Conv2D(name="Convolution2D_236", activation="relu", kernel_size=(3, 3), filters=32,
                               strides=(2, 2), padding="same")(Input_1)
    Convolution2D_235 = Conv2D(name="Convolution2D_235", activation="relu", kernel_size=(3, 3), filters=32,
                               padding="same")(Convolution2D_236)
    Convolution2D_237 = Conv2D(name="Convolution2D_237", activation="relu", kernel_size=(3, 3), filters=64)(
        Convolution2D_235)

    MaxPooling2D_69 = MaxPooling2D(name="MaxPooling2D_69", strides=(2, 2), pool_size=(3, 3), padding="same")(
        Convolution2D_237)

    Convolution2D_238 = Conv2D(name="Convolution2D_238", activation="relu", kernel_size=(3, 3), filters=80)(
        MaxPooling2D_69)
    Convolution2D_239 = Conv2D(name="Convolution2D_239", activation="relu", kernel_size=(3, 3), filters=192,
                               strides=(2, 2))(Convolution2D_238)

    MaxPooling2D_7 = MaxPooling2D(name="MaxPooling2D_7", strides=(2, 2), pool_size=(3, 3), padding="same")(
        Convolution2D_239)

    Convolution2D_10 = Conv2D(name="Convolution2D_10", activation="relu", kernel_size=(1, 1), filters=64,
                              padding="same")(MaxPooling2D_7)
    Convolution2D_3 = Conv2D(name="Convolution2D_3", activation="relu", kernel_size=(1, 1), filters=32, strides=(2, 2))(
        Convolution2D_239)
    Convolution2D_1 = Conv2D(name="Convolution2D_1", activation="relu", kernel_size=(1, 1), filters=32, padding="same")(
        Convolution2D_239)
    Convolution2D_4 = Conv2D(name="Convolution2D_4", activation="relu", kernel_size=(3, 1), filters=64, padding="same")(
        Convolution2D_1)
    Convolution2D_6 = Conv2D(name="Convolution2D_6", activation="relu", kernel_size=(1, 3), filters=64, padding="same")(
        Convolution2D_4)
    Convolution2D_2 = Conv2D(name="Convolution2D_2", activation="relu", kernel_size=(1, 1), filters=32)(
        Convolution2D_239)
    Convolution2D_8 = Conv2D(name="Convolution2D_8", activation="relu", kernel_size=(3, 1), filters=64, strides=(2, 1),
                             padding="same")(Convolution2D_2)
    Convolution2D_9 = Conv2D(name="Convolution2D_9", activation="relu", kernel_size=(1, 3), filters=64, strides=(1, 2),
                             padding="same")(Convolution2D_8)
    Convolution2D_5 = Conv2D(name="Convolution2D_5", activation="relu", kernel_size=(3, 1), filters=64, strides=(2, 1),
                             padding="same")(Convolution2D_6)
    Convolution2D_7 = Conv2D(name="Convolution2D_7", activation="relu", kernel_size=(1, 3), filters=64, strides=(1, 2),
                             padding="same")(Convolution2D_5)

    merge_1 = concatenate(inputs=[Convolution2D_10, Convolution2D_9, Convolution2D_7, Convolution2D_3], name='merge_1',
                          axis=-1)

    Convolution2D_130 = Conv2D(name="Convolution2D_130", activation="relu", kernel_size=(1, 1), filters=128,
                               strides=(2, 2))(merge_1)
    MaxPooling2D_15 = MaxPooling2D(name="MaxPooling2D_15", strides=(2, 2), pool_size=(3, 3), padding="same")(merge_1)
    Convolution2D_129 = Conv2D(name="Convolution2D_129", activation="relu", kernel_size=(1, 1), filters=128,
                               padding="same")(MaxPooling2D_15)
    Convolution2D_126 = Conv2D(name="Convolution2D_126", activation="relu", kernel_size=(1, 1), filters=128)(merge_1)
    Convolution2D_127 = Conv2D(name="Convolution2D_127", activation="relu", kernel_size=(3, 1), filters=128,
                               strides=(2, 1), padding="same")(Convolution2D_126)
    Convolution2D_128 = Conv2D(name="Convolution2D_128", activation="relu", kernel_size=(1, 3), filters=128,
                               strides=(1, 2), padding="same")(Convolution2D_127)
    Convolution2D_121 = Conv2D(name="Convolution2D_121", activation="relu", kernel_size=(1, 1), filters=128,
                               padding="same")(merge_1)
    Convolution2D_122 = Conv2D(name="Convolution2D_122", activation="relu", kernel_size=(3, 1), filters=128,
                               padding="same")(Convolution2D_121)
    Convolution2D_123 = Conv2D(name="Convolution2D_123", activation="relu", kernel_size=(1, 3), filters=128,
                               padding="same")(Convolution2D_122)
    Convolution2D_124 = Conv2D(name="Convolution2D_124", activation="relu", kernel_size=(3, 1), filters=128,
                               strides=(2, 1), padding="same")(Convolution2D_123)
    Convolution2D_125 = Conv2D(name="Convolution2D_125", activation="relu", kernel_size=(1, 3), filters=128,
                               strides=(1, 2), padding="same")(Convolution2D_124)

    merge_10 = concatenate(inputs=[Convolution2D_128, Convolution2D_129, Convolution2D_125, Convolution2D_130],
                           name='merge_10', axis=-1)

    Convolution2D_230 = Conv2D(name="Convolution2D_230", activation="relu", kernel_size=(1, 1), filters=128,
                               strides=(2, 2))(merge_10)
    MaxPooling2D_25 = MaxPooling2D(name="MaxPooling2D_25", strides=(2, 2), pool_size=(3, 3), padding="same")(merge_10)
    Convolution2D_229 = Conv2D(name="Convolution2D_229", activation="relu", kernel_size=(1, 1), filters=128,
                               padding="same")(MaxPooling2D_25)
    Convolution2D_226 = Conv2D(name="Convolution2D_226", activation="relu", kernel_size=(1, 1), filters=128)(merge_10)
    Convolution2D_227 = Conv2D(name="Convolution2D_227", activation="relu", kernel_size=(3, 1), filters=128,
                               strides=(2, 1), padding="same")(Convolution2D_226)
    Convolution2D_228 = Conv2D(name="Convolution2D_228", activation="relu", kernel_size=(1, 3), filters=128,
                               strides=(1, 2), padding="same")(Convolution2D_227)
    Convolution2D_221 = Conv2D(name="Convolution2D_221", activation="relu", kernel_size=(1, 1), filters=128,
                               padding="same")(merge_10)
    Convolution2D_222 = Conv2D(name="Convolution2D_222", activation="relu", kernel_size=(3, 1), filters=128,
                               padding="same")(Convolution2D_221)
    Convolution2D_223 = Conv2D(name="Convolution2D_223", activation="relu", kernel_size=(1, 3), filters=128,
                               padding="same")(Convolution2D_222)
    Convolution2D_224 = Conv2D(name="Convolution2D_224", activation="relu", kernel_size=(3, 1), filters=128,
                               strides=(2, 1), padding="same")(Convolution2D_223)
    Convolution2D_225 = Conv2D(name="Convolution2D_225", activation="relu", kernel_size=(1, 3), filters=128,
                               strides=(1, 2), padding="same")(Convolution2D_224)

    merge_20 = concatenate(inputs=[Convolution2D_228, Convolution2D_229, Convolution2D_225, Convolution2D_230],
                           name='merge_20', axis=-1)

    MaxPooling2D_70 = MaxPooling2D(name='MaxPooling2D_70', pool_size=(4, 4))(merge_20)

    Flatten_5 = Flatten(name='Flatten_5')(MaxPooling2D_70)
    Dense_14 = Dense(name="Dense_14", activation="linear", units=5)(Flatten_5)

    Dense_2 = Dense(name="Dense_2", activation="softmax", units=5)(Dense_14)

    model = Model([Input_1], [Dense_2])
    model._make_predict_function()

    for l in model.layers:
        logging.debug('Layer Shape: {} {}'.format(l.name, l.output_shape))

    logging.info(model.summary())
    return aliases, model

def get_inception_v3_model_small(input_shape=(224, 224, 3)):
    """
    Create a keras model
    :return: aliases, model
    """
    aliases = {}
    Input_1 = Input(shape=input_shape, name='Input_1')

    Convolution2D_236 = Conv2D(name="Convolution2D_236", activation="relu", kernel_size=(3, 3), filters=32,
                               strides=(2, 2), padding="same")(Input_1)
    Convolution2D_235 = Conv2D(name="Convolution2D_235", activation="relu", kernel_size=(3, 3), filters=32,
                               padding="same")(Convolution2D_236)
    Convolution2D_237 = Conv2D(name="Convolution2D_237", activation="relu", kernel_size=(3, 3), filters=64)(
        Convolution2D_235)

    MaxPooling2D_69 = MaxPooling2D(name="MaxPooling2D_69", strides=(2, 2), pool_size=(3, 3), padding="same")(
        Convolution2D_237)

    Convolution2D_238 = Conv2D(name="Convolution2D_238", activation="relu", kernel_size=(3, 3), filters=80)(
        MaxPooling2D_69)
    Convolution2D_239 = Conv2D(name="Convolution2D_239", activation="relu", kernel_size=(3, 3), filters=192,
                               strides=(2, 2))(Convolution2D_238)

    MaxPooling2D_7 = MaxPooling2D(name="MaxPooling2D_7", strides=(2, 2), pool_size=(3, 3), padding="same")(
        Convolution2D_239)

    Convolution2D_10 = Conv2D(name="Convolution2D_10", activation="relu", kernel_size=(1, 1), filters=64,
                              padding="same")(MaxPooling2D_7)
    Convolution2D_3 = Conv2D(name="Convolution2D_3", activation="relu", kernel_size=(1, 1), filters=32, strides=(2, 2))(
        Convolution2D_239)
    Convolution2D_1 = Conv2D(name="Convolution2D_1", activation="relu", kernel_size=(1, 1), filters=32, padding="same")(
        Convolution2D_239)
    Convolution2D_4 = Conv2D(name="Convolution2D_4", activation="relu", kernel_size=(3, 1), filters=64, padding="same")(
        Convolution2D_1)
    Convolution2D_6 = Conv2D(name="Convolution2D_6", activation="relu", kernel_size=(1, 3), filters=64, padding="same")(
        Convolution2D_4)
    Convolution2D_2 = Conv2D(name="Convolution2D_2", activation="relu", kernel_size=(1, 1), filters=32)(
        Convolution2D_239)
    Convolution2D_8 = Conv2D(name="Convolution2D_8", activation="relu", kernel_size=(3, 1), filters=64, strides=(2, 1),
                             padding="same")(Convolution2D_2)
    Convolution2D_9 = Conv2D(name="Convolution2D_9", activation="relu", kernel_size=(1, 3), filters=64, strides=(1, 2),
                             padding="same")(Convolution2D_8)
    Convolution2D_5 = Conv2D(name="Convolution2D_5", activation="relu", kernel_size=(3, 1), filters=64, strides=(2, 1),
                             padding="same")(Convolution2D_6)
    Convolution2D_7 = Conv2D(name="Convolution2D_7", activation="relu", kernel_size=(1, 3), filters=64, strides=(1, 2),
                             padding="same")(Convolution2D_5)

    merge_1 = concatenate(inputs=[Convolution2D_10, Convolution2D_9, Convolution2D_7, Convolution2D_3], name='merge_1',
                          axis=-1)

    MaxPooling2D_70 = MaxPooling2D(name='MaxPooling2D_70', pool_size=(7, 7))(merge_1)

    Flatten_5 = Flatten(name='Flatten_5')(MaxPooling2D_70)
    Dense_14 = Dense(name="Dense_14", activation="linear", units=5)(Flatten_5)
    Dense_2 = Dense(name="Dense_2", activation="softmax", units=5)(Dense_14)

    model = Model([Input_1], [Dense_2])
    model._make_predict_function()

    for l in model.layers:
        logging.debug('Layer Shape: {} {}'.format(l.name, l.output_shape))

    logging.info(model.summary())
    return aliases, model


def get_inception_v3_model_dropout(input_shape=(224, 224, 3), dropout_rate=0.2):
    """
    Create a keras model
    :return: aliases, model
    """
    aliases = {}
    Input_1 = Input(shape=input_shape, name='Input_1')

    Convolution2D_236 = Conv2D(name="Convolution2D_236", activation="relu", kernel_size=(3, 3), filters=32,
                               strides=(2, 2), padding="same")(Input_1)
    Convolution2D_235 = Conv2D(name="Convolution2D_235", activation="relu", kernel_size=(3, 3), filters=32,
                               padding="same")(Convolution2D_236)
    Convolution2D_237 = Conv2D(name="Convolution2D_237", activation="relu", kernel_size=(3, 3), filters=64)(
        Convolution2D_235)

    MaxPooling2D_69 = MaxPooling2D(name="MaxPooling2D_69", strides=(2, 2), pool_size=(3, 3), padding="same")(
        Convolution2D_237)

    Convolution2D_238 = Conv2D(name="Convolution2D_238", activation="relu", kernel_size=(3, 3), filters=80)(
        MaxPooling2D_69)
    Convolution2D_239 = Conv2D(name="Convolution2D_239", activation="relu", kernel_size=(3, 3), filters=192,
                               strides=(2, 2))(Convolution2D_238)

    MaxPooling2D_7 = MaxPooling2D(name="MaxPooling2D_7", strides=(2, 2), pool_size=(3, 3), padding="same")(
        Convolution2D_239)

    Convolution2D_10 = Conv2D(name="Convolution2D_10", activation="relu", kernel_size=(1, 1), filters=64,
                              padding="same")(MaxPooling2D_7)
    Convolution2D_3 = Conv2D(name="Convolution2D_3", activation="relu", kernel_size=(1, 1), filters=32, strides=(2, 2))(
        Convolution2D_239)
    Convolution2D_1 = Conv2D(name="Convolution2D_1", activation="relu", kernel_size=(1, 1), filters=32, padding="same")(
        Convolution2D_239)
    Convolution2D_4 = Conv2D(name="Convolution2D_4", activation="relu", kernel_size=(3, 1), filters=64, padding="same")(
        Convolution2D_1)
    Convolution2D_6 = Conv2D(name="Convolution2D_6", activation="relu", kernel_size=(1, 3), filters=64, padding="same")(
        Convolution2D_4)
    Convolution2D_2 = Conv2D(name="Convolution2D_2", activation="relu", kernel_size=(1, 1), filters=32)(
        Convolution2D_239)
    Convolution2D_8 = Conv2D(name="Convolution2D_8", activation="relu", kernel_size=(3, 1), filters=64, strides=(2, 1),
                             padding="same")(Convolution2D_2)
    Convolution2D_9 = Conv2D(name="Convolution2D_9", activation="relu", kernel_size=(1, 3), filters=64, strides=(1, 2),
                             padding="same")(Convolution2D_8)
    Convolution2D_5 = Conv2D(name="Convolution2D_5", activation="relu", kernel_size=(3, 1), filters=64, strides=(2, 1),
                             padding="same")(Convolution2D_6)
    Convolution2D_7 = Conv2D(name="Convolution2D_7", activation="relu", kernel_size=(1, 3), filters=64, strides=(1, 2),
                             padding="same")(Convolution2D_5)

    merge_1 = concatenate(inputs=[Convolution2D_10, Convolution2D_9, Convolution2D_7, Convolution2D_3], name='merge_1',
                          axis=-1)

    Convolution2D_130 = Conv2D(name="Convolution2D_130", activation="relu", kernel_size=(1, 1), filters=128,
                               strides=(2, 2))(merge_1)
    MaxPooling2D_15 = MaxPooling2D(name="MaxPooling2D_15", strides=(2, 2), pool_size=(3, 3), padding="same")(merge_1)
    Convolution2D_129 = Conv2D(name="Convolution2D_129", activation="relu", kernel_size=(1, 1), filters=128,
                               padding="same")(MaxPooling2D_15)
    Convolution2D_126 = Conv2D(name="Convolution2D_126", activation="relu", kernel_size=(1, 1), filters=128)(merge_1)
    Convolution2D_127 = Conv2D(name="Convolution2D_127", activation="relu", kernel_size=(3, 1), filters=128,
                               strides=(2, 1), padding="same")(Convolution2D_126)
    Convolution2D_128 = Conv2D(name="Convolution2D_128", activation="relu", kernel_size=(1, 3), filters=128,
                               strides=(1, 2), padding="same")(Convolution2D_127)
    Convolution2D_121 = Conv2D(name="Convolution2D_121", activation="relu", kernel_size=(1, 1), filters=128,
                               padding="same")(merge_1)
    Convolution2D_122 = Conv2D(name="Convolution2D_122", activation="relu", kernel_size=(3, 1), filters=128,
                               padding="same")(Convolution2D_121)
    Convolution2D_123 = Conv2D(name="Convolution2D_123", activation="relu", kernel_size=(1, 3), filters=128,
                               padding="same")(Convolution2D_122)
    Convolution2D_124 = Conv2D(name="Convolution2D_124", activation="relu", kernel_size=(3, 1), filters=128,
                               strides=(2, 1), padding="same")(Convolution2D_123)
    Convolution2D_125 = Conv2D(name="Convolution2D_125", activation="relu", kernel_size=(1, 3), filters=128,
                               strides=(1, 2), padding="same")(Convolution2D_124)

    merge_10 = concatenate(inputs=[Convolution2D_128, Convolution2D_129, Convolution2D_125, Convolution2D_130],
                           name='merge_10', axis=-1)

    MaxPooling2D_70 = MaxPooling2D(name='MaxPooling2D_70', pool_size=(7, 7))(merge_10)

    Flatten_5 = Flatten(name='Flatten_5')(MaxPooling2D_70)
    Dense_14 = Dense(name="Dense_14", activation="linear", units=5)(Flatten_5)
    Dropout_1 = Dropout(name='Dropout_1', rate=dropout_rate, seed=42)(Dense_14)
    Dense_2 = Dense(name="Dense_2", activation="softmax", units=5)(Dropout_1)

    model = Model([Input_1], [Dense_2])
    model._make_predict_function()

    for l in model.layers:
        logging.debug('Layer Shape: {} {}'.format(l.name, l.output_shape))

    logging.info(model.summary())
    return aliases, model


def get_inception_v3_model_l2(input_shape=(224, 224, 3), regularization_factor=0.01):
    """
    Create a keras model
    :return: aliases, model
    """
    aliases = {}
    Input_1 = Input(shape=input_shape, name='Input_1')

    Convolution2D_236 = Conv2D(name="Convolution2D_236", activation="relu", kernel_size=(3, 3), filters=32,
                               strides=(2, 2), padding="same")(Input_1)
    Convolution2D_235 = Conv2D(name="Convolution2D_235", activation="relu", kernel_size=(3, 3), filters=32,
                               padding="same")(Convolution2D_236)
    Convolution2D_237 = Conv2D(name="Convolution2D_237", activation="relu", kernel_size=(3, 3), filters=64)(
        Convolution2D_235)

    MaxPooling2D_69 = MaxPooling2D(name="MaxPooling2D_69", strides=(2, 2), pool_size=(3, 3), padding="same")(
        Convolution2D_237)

    Convolution2D_238 = Conv2D(name="Convolution2D_238", activation="relu", kernel_size=(3, 3), filters=80)(
        MaxPooling2D_69)
    Convolution2D_239 = Conv2D(name="Convolution2D_239", activation="relu", kernel_size=(3, 3), filters=192,
                               strides=(2, 2))(Convolution2D_238)

    MaxPooling2D_7 = MaxPooling2D(name="MaxPooling2D_7", strides=(2, 2), pool_size=(3, 3), padding="same")(
        Convolution2D_239)

    Convolution2D_10 = Conv2D(name="Convolution2D_10", activation="relu", kernel_size=(1, 1), filters=64,
                              padding="same")(MaxPooling2D_7)
    Convolution2D_3 = Conv2D(name="Convolution2D_3", activation="relu", kernel_size=(1, 1), filters=32,
                             strides=(2, 2))(Convolution2D_239)
    Convolution2D_1 = Conv2D(name="Convolution2D_1", activation="relu", kernel_size=(1, 1), filters=32,
                             padding="same")(Convolution2D_239)
    Convolution2D_4 = Conv2D(name="Convolution2D_4", activation="relu", kernel_size=(3, 1), filters=64,
                             padding="same")(Convolution2D_1)
    Convolution2D_6 = Conv2D(name="Convolution2D_6", activation="relu", kernel_size=(1, 3), filters=64,
                             padding="same")(Convolution2D_4)
    Convolution2D_2 = Conv2D(name="Convolution2D_2", activation="relu", kernel_size=(1, 1), filters=32)(
        Convolution2D_239)
    Convolution2D_8 = Conv2D(name="Convolution2D_8", activation="relu", kernel_size=(3, 1), filters=64,
                             strides=(2, 1), padding="same")(Convolution2D_2)
    Convolution2D_9 = Conv2D(name="Convolution2D_9", activation="relu", kernel_size=(1, 3), filters=64,
                             strides=(1, 2), padding="same")(Convolution2D_8)
    Convolution2D_5 = Conv2D(name="Convolution2D_5", activation="relu", kernel_size=(3, 1), filters=64,
                             strides=(2, 1), padding="same")(Convolution2D_6)
    Convolution2D_7 = Conv2D(name="Convolution2D_7", activation="relu", kernel_size=(1, 3), filters=64,
                             strides=(1, 2), padding="same")(Convolution2D_5)

    merge_1 = concatenate(inputs=[Convolution2D_10, Convolution2D_9, Convolution2D_7, Convolution2D_3],
                          name='merge_1', axis=-1)

    Convolution2D_130 = Conv2D(name="Convolution2D_130", activation="relu", kernel_size=(1, 1), filters=128,
                               strides=(2, 2),kernel_regularizer=regularizers.l2(regularization_factor))(merge_1)
    MaxPooling2D_15 = MaxPooling2D(name="MaxPooling2D_15", strides=(2, 2), pool_size=(3, 3), padding="same")(
        merge_1)
    Convolution2D_129 = Conv2D(name="Convolution2D_129", activation="relu", kernel_size=(1, 1), filters=128,
                               padding="same",kernel_regularizer=regularizers.l2(regularization_factor))(MaxPooling2D_15)
    Convolution2D_126 = Conv2D(name="Convolution2D_126", activation="relu", kernel_size=(1, 1), filters=128,kernel_regularizer=regularizers.l2(regularization_factor))(
        merge_1)
    Convolution2D_127 = Conv2D(name="Convolution2D_127", activation="relu", kernel_size=(3, 1), filters=128,
                               strides=(2, 1), padding="same",kernel_regularizer=regularizers.l2(regularization_factor))(Convolution2D_126)
    Convolution2D_128 = Conv2D(name="Convolution2D_128", activation="relu", kernel_size=(1, 3), filters=128,
                               strides=(1, 2), padding="same",kernel_regularizer=regularizers.l2(regularization_factor))(Convolution2D_127)
    Convolution2D_121 = Conv2D(name="Convolution2D_121", activation="relu", kernel_size=(1, 1), filters=128,
                               padding="same",kernel_regularizer=regularizers.l2(regularization_factor))(merge_1)
    Convolution2D_122 = Conv2D(name="Convolution2D_122", activation="relu", kernel_size=(3, 1), filters=128,
                               padding="same",kernel_regularizer=regularizers.l2(regularization_factor))(Convolution2D_121)
    Convolution2D_123 = Conv2D(name="Convolution2D_123", activation="relu", kernel_size=(1, 3), filters=128,
                               padding="same",kernel_regularizer=regularizers.l2(regularization_factor))(Convolution2D_122)
    Convolution2D_124 = Conv2D(name="Convolution2D_124", activation="relu", kernel_size=(3, 1), filters=128,
                               strides=(2, 1), padding="same",kernel_regularizer=regularizers.l2(regularization_factor))(Convolution2D_123)
    Convolution2D_125 = Conv2D(name="Convolution2D_125", activation="relu", kernel_size=(1, 3), filters=128,
                               strides=(1, 2), padding="same",kernel_regularizer=regularizers.l2(regularization_factor))(Convolution2D_124)

    merge_10 = concatenate(inputs=[Convolution2D_128, Convolution2D_129, Convolution2D_125, Convolution2D_130],
                           name='merge_10', axis=-1)

    MaxPooling2D_70 = MaxPooling2D(name='MaxPooling2D_70', pool_size=(7, 7))(merge_10)

    Flatten_5 = Flatten(name='Flatten_5')(MaxPooling2D_70)
    Dense_14 = Dense(name="Dense_14", activation="linear", units=5)(Flatten_5)

    Dense_2 = Dense(name="Dense_2", activation="softmax", units=5)(Dense_14)

    model = Model([Input_1], [Dense_2])
    model._make_predict_function()

    for l in model.layers:
        logging.debug('Layer Shape: {} {}'.format(l.name, l.output_shape))

    logging.info(model.summary())
    return aliases, model


def get_VGG_model():
    """
    Create a keras model
    :return: aliases, model
    """
    aliases = {}

    # LOAD VGG16
    # Generate a model with all layers (with top)
    vgg16 = VGG16(weights='imagenet', include_top=True)

    # Add a layer where input is the output of the  second last layer
    x = Dense(5, activation='softmax', name='predictions')(vgg16.layers[-2].output)

    # Then create the corresponding model
    model = Model(input=vgg16.input, output=x)
    model.summary()

    return aliases, model


def get_VGG_model_imagenet_custom_top():
    """
    Create a keras model
    :return: aliases, model
    """
    aliases = {}

    # LOAD VGG16
    # Generate a model with all layers (with top)
    vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
    #for layer in vgg16.layers:
    #    layer.trainable = False
    x = vgg16.output
    x = Flatten(name='flatten')(x)
    x = Dense(2048, activation='relu', name='fc1')(x)
    x = Dense(2048, activation='relu', name='fc2')(x)
    x = Dense(5, activation='softmax', name='predictions')(x)

    # Add a layer where input is the output of the  second last layer
    #x = Dense(5, activation='softmax', name='predictions')(vgg16.layers[-2].output)

    # Then create the corresponding model
    model = Model(input=vgg16.input, output=x)
    model.summary()

    return aliases, model

def get_VGG_model_imagenet_custom_top_SGD():
    """
    Create a keras model
    :return: aliases, model
    """
    aliases = {}

    # LOAD VGG16
    # Generate a model with all layers (with top)
    vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
    #for layer in vgg16.layers:
    #    layer.trainable = False
    x = vgg16.output
    x = Flatten(name='flatten')(x)
    x = Dense(2048, activation='relu', name='fc1')(x)
    x = Dense(2048, activation='relu', name='fc2')(x)
    x = Dense(5, activation='softmax', name='predictions')(x)

    # Add a layer where input is the output of the  second last layer
    #x = Dense(5, activation='softmax', name='predictions')(vgg16.layers[-2].output)

    # Then create the corresponding model
    model = Model(input=vgg16.input, output=x)
    model.summary()

    return aliases, model

def get_InceptionResNetV2_model(input_shape=(299,299,3)):

    aliases = {}
    input_1 = Input(shape=input_shape, name='input_1')

    inception_model = InceptionResNetV2(include_top=True, weights='imagenet', input_tensor=input_1, pooling=None, classes=1000)

    for l in inception_model.layers[:-5]:
        l.trainable = False

    out_layer = inception_model.output

    dense_layer = Dense(name="Dense_layer", activation="softmax", units=5)(out_layer)

    model = Model(input_1, dense_layer)
    model._make_predict_function()

    logging.info(model.summary())


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
