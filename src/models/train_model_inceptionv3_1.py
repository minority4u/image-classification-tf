import os
import sys

sys.path.append(os.path.abspath("."))
print(sys.path)

# import own lobs
from src.models.v3_model import *
from src.data.make_dataset import get_train_and_validation_generator
from src.models.model_utils import get_callbacks
from src.utils_io import Console_and_file_logger, ensure_dir
from src.visualization.utils import create_reports
from keras.backend.tensorflow_backend import set_session
from collections import Counter

# import external libs
import json
from argparse import ArgumentParser
import yaml
import numpy as np
import tensorflow as tf


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
                                                                               batch_size_train=config[
                                                                                   'batch_size_train'],
                                                                               batch_size_val=config['batch_size_val'],
                                                                               class_mode=config['class_mode'])

    counter = Counter(train_generator.classes)
    max_val = float(max(counter.values()))
    class_weights = {class_id: max_val / num_images for class_id, num_images in counter.items()}
    logging.info('Class weights: {0}'.format(class_weights))
    # get model
    aliases, model = get_model()

    # compile model
    model.compile(loss=config['loss_function'],
                  optimizer=get_optimizer(),
                  metrics=config['metrics'])

    callbacks = get_callbacks(config)

    # model fit with generator
    history = model.fit_generator(train_generator, steps_per_epoch=int(config['steps_per_epoch']),
                                  epochs=int(config['epochs']), verbose=2, callbacks=callbacks,
                                  validation_data=validation_generator,
                                  validation_steps=20, class_weight=class_weights, max_queue_size=10, workers=1,
                                  use_multiprocessing=False,
                                  shuffle=True, initial_epoch=0)

    predictions = model.predict_generator(validation_generator, steps=None, max_queue_size=10, workers=1,
                                          use_multiprocessing=False, verbose=0)
    ground_truth = validation_generator.classes
    predicted_classes = np.argmax(predictions, axis=1)

    logging.info('ground truth {0}'.format(ground_truth))
    logging.info("predictions {0}".format(predictions))
    ground_truth_max = np.argmax(ground_truth, axis=1)
    logging.info('ground truth {0}'.format(ground_truth))
    number_of_classes = len(validation_generator.class_indices)
    create_reports(ground_truth_max, predicted_classes, validation_generator.class_indices, config)




if __name__ == '__main__':
    #gpu_options = tf.GPUOptions(allow_growth=True)
    #session_config =tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = False
    session = tf.Session(config=config)
    set_session(session)
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
