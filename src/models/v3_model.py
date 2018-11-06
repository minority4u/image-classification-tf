import keras
from keras.layers.convolutional import Convolution2D
#from keras.engine.topology import merge
from keras.layers.core import Dense
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Flatten
from keras.layers import Input
from keras.models import Model
from keras.regularizers import *
from keras.layers import merge


def get_model():
    aliases = {}
    Input_1 = Input(shape=(3, 224, 224), name='Input_1')
    Convolution2D_236 = Convolution2D(name='Convolution2D_236', activation='relu', subsample=(2, 2), border_mode='same',
                                      nb_row=3, nb_col=3, nb_filter=32)(Input_1)
    Convolution2D_235 = Convolution2D(name='Convolution2D_235', activation='relu', border_mode='same', nb_row=3,
                                      nb_col=3, nb_filter=32)(Convolution2D_236)
    Convolution2D_237 = Convolution2D(name='Convolution2D_237', activation='relu', nb_row=3, nb_col=3, nb_filter=64)(
        Convolution2D_235)
    MaxPooling2D_69 = MaxPooling2D(name='MaxPooling2D_69', border_mode='same', strides=(2, 2), pool_size=(3, 3))(
        Convolution2D_237)
    Convolution2D_238 = Convolution2D(name='Convolution2D_238', activation='relu', nb_row=3, nb_col=3, nb_filter=80)(
        MaxPooling2D_69)
    Convolution2D_239 = Convolution2D(name='Convolution2D_239', activation='relu', nb_row=3, subsample=(2, 2), nb_col=3,
                                      nb_filter=192)(Convolution2D_238)
    MaxPooling2D_7 = MaxPooling2D(name='MaxPooling2D_7', border_mode='same', strides=(2, 2), pool_size=(3, 3))(
        Convolution2D_239)
    Convolution2D_10 = Convolution2D(name='Convolution2D_10', activation='relu', border_mode='same', nb_row=1, nb_col=1,
                                     nb_filter=64)(MaxPooling2D_7)
    Convolution2D_3 = Convolution2D(name='Convolution2D_3', activation='relu', nb_row=1, subsample=(2, 2), nb_col=1,
                                    nb_filter=32)(Convolution2D_239)
    Convolution2D_1 = Convolution2D(name='Convolution2D_1', activation='relu', border_mode='same', nb_row=1, nb_col=1,
                                    nb_filter=32)(Convolution2D_239)
    Convolution2D_4 = Convolution2D(name='Convolution2D_4', activation='relu', border_mode='same', nb_row=3, nb_col=1,
                                    nb_filter=64)(Convolution2D_1)
    Convolution2D_6 = Convolution2D(name='Convolution2D_6', activation='relu', border_mode='same', nb_row=1, nb_col=3,
                                    nb_filter=64)(Convolution2D_4)
    Convolution2D_2 = Convolution2D(name='Convolution2D_2', activation='relu', nb_row=1, nb_col=1, nb_filter=32)(
        Convolution2D_239)
    Convolution2D_8 = Convolution2D(name='Convolution2D_8', activation='relu', subsample=(2, 1), border_mode='same',
                                    nb_row=3, nb_col=1, nb_filter=64)(Convolution2D_2)
    Convolution2D_9 = Convolution2D(name='Convolution2D_9', activation='relu', subsample=(1, 2), border_mode='same',
                                    nb_row=1, nb_col=3, nb_filter=64)(Convolution2D_8)
    Convolution2D_5 = Convolution2D(name='Convolution2D_5', activation='relu', subsample=(2, 1), border_mode='same',
                                    nb_row=3, nb_col=1, nb_filter=64)(Convolution2D_6)
    Convolution2D_7 = Convolution2D(name='Convolution2D_7', activation='relu', subsample=(1, 2), border_mode='same',
                                    nb_row=1, nb_col=3, nb_filter=64)(Convolution2D_5)
    merge_1 = merge(inputs=[Convolution2D_10, Convolution2D_9, Convolution2D_7, Convolution2D_3], name='merge_1',
                    concat_axis=1, mode='concat')
    Convolution2D_130 = Convolution2D(name='Convolution2D_130', activation='relu', nb_row=1, subsample=(2, 2), nb_col=1,
                                      nb_filter=128)(merge_1)
    MaxPooling2D_15 = MaxPooling2D(name='MaxPooling2D_15', border_mode='same', strides=(2, 2), pool_size=(3, 3))(
        merge_1)
    Convolution2D_129 = Convolution2D(name='Convolution2D_129', activation='relu', border_mode='same', nb_row=1,
                                      nb_col=1, nb_filter=128)(MaxPooling2D_15)
    Convolution2D_126 = Convolution2D(name='Convolution2D_126', activation='relu', nb_row=1, nb_col=1, nb_filter=128)(
        merge_1)
    Convolution2D_127 = Convolution2D(name='Convolution2D_127', activation='relu', subsample=(2, 1), border_mode='same',
                                      nb_row=3, nb_col=1, nb_filter=32)(Convolution2D_126)
    Convolution2D_128 = Convolution2D(name='Convolution2D_128', activation='relu', subsample=(1, 2), border_mode='same',
                                      nb_row=1, nb_col=3, nb_filter=128)(Convolution2D_127)
    Convolution2D_121 = Convolution2D(name='Convolution2D_121', activation='relu', border_mode='same', nb_row=1,
                                      nb_col=1, nb_filter=128)(merge_1)
    Convolution2D_122 = Convolution2D(name='Convolution2D_122', activation='relu', border_mode='same', nb_row=3,
                                      nb_col=1, nb_filter=128)(Convolution2D_121)
    Convolution2D_123 = Convolution2D(name='Convolution2D_123', activation='relu', border_mode='same', nb_row=1,
                                      nb_col=3, nb_filter=128)(Convolution2D_122)
    Convolution2D_124 = Convolution2D(name='Convolution2D_124', activation='relu', subsample=(2, 1), border_mode='same',
                                      nb_row=3, nb_col=1, nb_filter=128)(Convolution2D_123)
    Convolution2D_125 = Convolution2D(name='Convolution2D_125', activation='relu', subsample=(1, 2), border_mode='same',
                                      nb_row=1, nb_col=3, nb_filter=128)(Convolution2D_124)
    merge_10 = merge(inputs=[Convolution2D_128, Convolution2D_129, Convolution2D_125, Convolution2D_130],
                     name='merge_10', concat_axis=1, mode='concat')
    MaxPooling2D_70 = MaxPooling2D(name='MaxPooling2D_70', pool_size=(7, 7))(merge_10)
    Flatten_5 = Flatten(name='Flatten_5')(MaxPooling2D_70)
    Dense_14 = Dense(name='Dense_14', activation='linear', output_dim=3)(Flatten_5)
    Dense_2 = Dense(name='Dense_2', activation='softmax', output_dim=3)(Dense_14)

    model = Model([Input_1], [Dense_2])
    return aliases, model


from keras.optimizers import *


def get_optimizer():
    return Adadelta()


def is_custom_loss_function():
    return False


def get_loss_function():
    return 'categorical_crossentropy'


def get_batch_size():
    return 8


def get_num_epoch():
    return 1000


def get_data_config():
    return '{"kfold": 1, "numPorts": 1, "samples": {"validation": 450, "training": 2100, "split": 3, "test": 450}, "datasetLoadOption": "batch", "mapping": {"Filename": {"port": "InputPort0", "type": "Image", "shape": "", "options": {"horizontal_flip": false, "Height": "224", "rotation_range": 0, "vertical_flip": false, "width_shift_range": 0, "Normalization": false, "Width": "224", "shear_range": 0, "pretrained": "None", "Scaling": 1, "Augmentation": false, "Resize": true, "height_shift_range": 0}}, "Label": {"port": "OutputPort0", "type": "Categorical", "shape": "", "options": {}}}, "dataset": {"samples": 3000, "name": "Classify1000", "type": "private"}, "shuffle": true}'
