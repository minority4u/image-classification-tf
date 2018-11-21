

import logging
import os
import sys
import yaml
import json


from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
sys.path.append(os.path.abspath("."))
from src.data.data_utils import load_all_images, save_images
from src.utils_io import Console_and_file_logger


def split_dataset(src_path, dest_path):
    # load all images
    images = load_all_images(src_path)

    X = []
    y = []

    # transform image shapes
    for label, images in images:
        for image in images:
            X.append(image)
            y.append(label)

    # split images per class
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    # write split images to disk
    save_images(X_train, y_train, os.path.join(dest_path, '/train'))
    save_images(X_test, y_test, os.path.join(dest_path, '/test'))

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

    source_root = 'data/raw/classification_data/'
    destination_root = 'data/processed/split/'

    split_dataset(source_root, destination_root)