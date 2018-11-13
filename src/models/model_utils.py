from keras.callbacks import Callback
import keras
import os
import logging
from src.utils_io import ensure_dir


def get_callbacks(config):
    """
    get a list of keras callbacks
    :param config:
    :return: a list of allback objects
    """
    model_path = os.path.join(config['model_path'], 'weights.{epoch:02d}-{val_loss:.2f}.hdf5')

    callbacks = []
    callbacks.append(WeightsSaver(config['steps_per_epoch'], model_path))
    #callbacks.append(keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', verbose=0, save_best_only=False,
    #                                                save_weights_only=False, mode='auto', period=1))
    callbacks.append(keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=config['batch_size_train'], write_graph=True,
                                                 write_grads=False, write_images=False, embeddings_freq=0,
                                                 embeddings_layer_names=None, embeddings_metadata=None,
                                                 embeddings_data=None, update_freq='epoch'))
    return callbacks


class WeightsSaver(Callback):
    def __init__(self, N, model_path):
        self.model.path = model_path
        self.N = N
        self.batch = 0

    def on_batch_end(self, batch, logs={}):
        if self.batch % self.N == 0:
            name = 'weights%08d.h5' % self.batch
            #self.model.save_weights(name)
            # Save the model
            # serialize model to JSON
            model_json = self.model.to_json()
            model_path = self.model_path
            ensure_dir(model_path)
            with open(os.path.join(model_path, 'model.json'), "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            self.model.save_weights("models/model.h5")
            logging.info("Saved model to disk: {}".format(model_path))
        self.batch += 1
