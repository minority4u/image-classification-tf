import os
import sys
import logging

sys.path.append(os.path.abspath("."))
import cv2
from src.data.data_utils import create_patches
from models.load import init
from src.utils_io import Console_and_file_logger
from src.visualization.utils import create_reports
from src.data.data_utils import load_all_images
from src.models.predict_model import external_predict_images
from src.data.data_utils import __get_generator__
from keras.preprocessing.image import ImageDataGenerator
from src.data.data_utils import get_class_names
from scipy.misc import imsave, imread, imresize
import numpy as np
from argparse import ArgumentParser
import yaml
import json

from collections import Counter
import operator

global model, graph
global config
global class_names
class_names = get_class_names()


def evaluate_on_patch_level(evaluation_path):
    test_generator = __get_generator__(ImageDataGenerator(), evaluation_path,
                                       (config['input_image_width'], config['input_image_height']),
                                       config['batch_size_test'],
                                       config['class_mode'],
                                       'validation',
                                       shuffle=False)

    # evaluate on patch level
    predictions = model.predict_generator(test_generator, steps=None, max_queue_size=10, workers=1,
                                          use_multiprocessing=False, verbose=0)
    ground_truth = test_generator.classes
    predicted_classes = np.argmax(predictions, axis=1)

    logging.info('ground truth {0}'.format(ground_truth))
    logging.info("predictions {0}".format(predictions))
    ground_truth_max = np.argmax(ground_truth, axis=1)
    logging.info('ground truth {0}'.format(ground_truth))
    number_of_classes = len(test_generator.class_indices)
    create_reports(ground_truth_max, predicted_classes, test_generator.class_indices, config)


def evaluate_on_image_level(evaluation_path):
    logging.debug('Inference with: {}'.format(evaluation_path))

    inference_images = load_all_images(evaluation_path)
    logging.debug('Classes found: {}'.format(len(inference_images)))
    # logging.debug('Images found: {}'.format(inference_images))
    global model, graph
    results = [(label, external_predict_images(images, model, graph)) for label, images in inference_images]

    logging.info(results)

    test_pred = []
    test_label = []

    for label, list_of_pred in results:
        # class_names.append(label)
        if list_of_pred:
            for pred in list_of_pred:
                test_pred.append(pred)
                test_label.append(label)
    #logging.info("Results: " + len(results))
    logging.info(test_label)
    logging.info(test_pred)
    logging.info(class_names)
    create_reports(ground_truth=test_label, predicted_classes=test_pred, class_names=class_names, config=config)

    logging.info(results)


if __name__ == '__main__':
    # Define argument parser
    parser = ArgumentParser()
    Console_and_file_logger('Predict_model', log_lvl=logging.DEBUG)

    # define arguments and default values to parse
    # define tha path to your config file
    parser.add_argument("--config", "-c", help="Define the path to config.yml",
                        default="config/experiments/inception_v3_base.yml", required=False)

    parser.add_argument("--working_dir", help="Define the absolute path to the project root",
                        default="../../", required=False)
    # parser.add_argument("--modelskiptraining", help="Skip Training", default="None", required=False)

    args = parser.parse_args()
    logging.debug(args.config)
    # Make sure the config exists
    assert os.path.exists(
        args.config), "Config does not exist {}!, Please create a config.yml in root or set the path with --config.".format(
        args.config)

    # Load config and other global objects
    config = yaml.load(open(args.config, "r"))
    logging.debug(json.dumps(config, indent=2))
    model, graph = init(config)
    class_names = []

    #evaluate_on_image_level(config['test_dir'])
    evaluate_on_patch_level(config['test_dir'])
