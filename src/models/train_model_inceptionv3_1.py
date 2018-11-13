import os
import sys
from src.visualization.utils import plot_confusion_matrix, plot_history
sys.path.append(os.path.abspath("."))
print(sys.path)
from src.utils_io import Console_and_file_logger, ensure_dir, parameter_logger
import logging
from argparse import ArgumentParser
import yaml
from src.models.v3_model import *
from src.data.make_dataset import get_train_and_validation_generator
import json
from time import time
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from src.visualization.utils import plot_history, plot_confusion_matrix, create_reports

def train(config):
    """
    training entrance, all heavy work is done here
    :param config: Json representation of our config file
    :return:
    """

    logging.info('training starts')

    # get train generator
    train_generator, validation_generator = get_train_and_validation_generator(path_to_data=config['data_dir'],
                                                                               validation_split=config[
                                                                                   'validation_split'],
                                                                               image_size=(config['input_image_width'],
                                                                                           config[
                                                                                               'input_image_height']),
                                                                               batch_size_train=config['batch_size_train'],
                                                                               batch_size_val=config['batch_size_val'],
                                                                               class_mode=config['class_mode'])

    # get model
    aliases, model = get_model()

    # compile model
    model.compile(loss=config['loss_function'],
                  optimizer=get_optimizer(),
                  metrics=config['metrics'])


    callbacks = []
    model_path = os.path.join(config['model_path'], '/weights.{epoch:02d}-{val_loss:.2f}.hdf5')
    callbacks.append(keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1))
    callbacks.append(keras.callbacks.ProgbarLogger(count_mode='steps', stateful_metrics=None))
    callbacks.append(keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch'))


    # model fit with generator
    history = model.fit_generator(train_generator, steps_per_epoch=int(config['steps_per_epoch']),
                                  epochs=int(config['epochs']), verbose=2, callbacks=callbacks,
                                  validation_data=validation_generator,
                                  validation_steps=20, class_weight=None, max_queue_size=10, workers=1,
                                  use_multiprocessing=False,
                                  shuffle=False, initial_epoch=0)


    predictions = model.predict_generator(validation_generator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
    ground_truth = validation_generator.classes
    predicted_classes = np.argmax(predictions, axis=1)

    logging.info('ground truth {0}'.format(ground_truth))
    logging.info("predictions {0}".format(predictions))
    ground_truth_max = np.argmax(ground_truth, axis=1)
    logging.info('ground truth {0}'.format(ground_truth))
    number_of_classes = len(validation_generator.class_indices)
    create_reports(ground_truth_max, predicted_classes, number_of_classes, config)


    # Save the model
    # serialize model to JSON
    model_json = model.to_json()
    model_path = config['model_path']
    ensure_dir(model_path)
    with open(os.path.join(model_path, 'model.json'), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("models/model.h5")
    logging.info("Saved model to disk: {}".format(model_path))

    logging.info('History:')
    logging.info(history.history)
    plot_history(history)


if __name__ == '__main__':
    # Define argument parser
    parser = ArgumentParser()

    # define arguments and default values to parse
    # define tha path to your config file
    parser.add_argument("--config", "-c", help="Define the path to config.yml",
                        default="config/experiments/inception_v3_base.yml", required=False)

    parser.add_argument("--working_dir", help="Define the absolute path to the project root",
                        default="../../", required=False)

    # parser.add_argument("--modelskiptraining", help="Skip Training", default="None", required=False)

    args = parser.parse_args()

    print(args.config)

    # Make sure the config exists
    assert os.path.exists(
        args.config), "Config does not exist {}!, Please create a config.yml in root or set the path with --config.".format(
        args.config)

    # Load config
    params = yaml.load(open(args.config, "r"))

    # Make sure that source and destination are set
    assert {"batch_size_train", "epochs", "data_dir", "experiment_name"} <= set(
        params.keys()), "Configuration is incomplete! Please define dir_to_src and dir_to_dest in config.yml"

    # Make sure source folder exists
    assert os.path.exists(params["data_dir"]), "Path to src {} does not exist!".format(params["data_dir"])

    # Define central logger, set name and logging level
    Console_and_file_logger(logfile_name=params["experiment_name"], log_lvl="INFO")
    logging.info('Starting experiment {}'.format(params["experiment_name"]))
    logging.info(json.dumps(params, indent=2))

    train(params)