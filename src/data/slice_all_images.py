import os
import sys
import logging
import yaml
import json
import cv2
from argparse import ArgumentParser
sys.path.append(os.path.abspath("."))

from src.utils_io import Console_and_file_logger, ensure_dir
from src.models.predict_model import load_all_images
from src.data.preprocessing import create_slides

global source_root, destination_root

def save_image_slices(images):
    for label, image_list in images:
        full_path = os.path.join(destination_root, label)
        ensure_dir(full_path)
        for idx, image in enumerate(image_list):
            file_n = str(label) + str(idx)
            slices = create_slides(image)
            for idy, slice in enumerate(slices):
                slice_name = file_n + "_" + str(idy) + ".jpg"
                filename = os.path.join(full_path, slice_name)
                cv2.imwrite(filename, slice)
                del slice
            del image
        del image_list




def slice_all_images():
    images = load_all_images(source_root)
    save_image_slices(images)


if __name__ == '__main__':
    # Define argument parser
    parser = ArgumentParser()
    Console_and_file_logger('Predict_model',log_lvl=logging.DEBUG)


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

    source_root = 'data/raw/test/'
    destination_root = 'data/processed/transformed/'

    slice_all_images()