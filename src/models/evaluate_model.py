import os
import sys
import logging

sys.path.append(os.path.abspath("."))

from models.load import init
from src.utils_io import Console_and_file_logger
from src.visualization.utils import create_reports
from src.data.data_utils import load_all_images
from src.models.predict_model import external_predict_images
from src.data.data_utils import get_class_names, set_config
from src.models.Result import Result
from argparse import ArgumentParser
import yaml
import json

global model, graph
global config
global class_names
class_names = get_class_names()


def evaluate(evaluation_path):
    logging.info('Inference with: {}'.format(evaluation_path))
    set_config(config)

    inference_images = load_all_images(evaluation_path)
    logging.info('Classes found: {}'.format(len(inference_images)))

    global model, graph, class_names

    results = []
    [results.extend(external_predict_images(images, label, model, graph, conf=config, resize=True)) for label, images in inference_images]

    result = Result(results)

    class_names = result.get_class_names()
    image_truth = result.get_image_truth()
    patch_truth = result.get_patch_truth()
    image_prediction = result.get_image_results_as_class_name()
    patch_predictions = result.get_patch_results_as_class_name()

    logging.info('patch_truth: {}'.format(len(patch_truth)))
    logging.info('patch_predictions: {}'.format(len(patch_predictions)))

    create_reports(ground_truth=image_truth,
                   predicted_classes=image_prediction,
                   class_names=class_names,
                   config=config,
                   report_name=config['experiment_name'],
                   f_name_suffix='image')

    create_reports(ground_truth=patch_truth,
                   predicted_classes=patch_predictions,
                   class_names=class_names,
                   config=config,
                   report_name=config['experiment_name'],
                   f_name_suffix='patch')


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
    #logging.debug(args.config)
    # Make sure the config exists
    assert os.path.exists(
        args.config), "Config does not exist {}!, Please create a config.yml in root or set the path with --config.".format(
        args.config)

    # Load config and other global objects
    config = yaml.load(open(args.config, "r"))
    Console_and_file_logger('predict_' + config['experiment_name'], log_lvl=logging.INFO)
    logging.debug(json.dumps(config, indent=2))
    model, graph = init(config)
    logging.info(model.summary())

    evaluate(config['test_dir_image'])
