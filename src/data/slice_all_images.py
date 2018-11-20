import os
import sys
import logging
import yaml
import json
import cv2
import math
from argparse import ArgumentParser

sys.path.append(os.path.abspath("."))

from src.utils_io import Console_and_file_logger, ensure_dir
from src.models.predict_model import load_all_images
from src.data.preprocessing import create_patches

global source_root, destination_root


def crop_edges(image):
    """
    Crop the edges of a given image
    :param image:
    :return: croped_image
    """
    crop_precentage = 10
    height, width = image.shape[:2]
    margin_width = math.floor(width / 100 * crop_precentage)
    margin_height = math.floor(height / 100 * crop_precentage)

    crop_img = image[margin_height:-margin_height, margin_width:-margin_width]
    return crop_img


def save_patches_to_disk(patches, path_n, file_n):
    """
    Saves a list of patches to disk
    :param patches:
    :param path_n:
    :param file_n:
    :return:
    """

    for idy, patch in enumerate(patches):
        #
        patch_name = file_n + "_" + str(idy) + ".jpg"

        # define new destination for filtered images
        if is_patch_empty(patch):
            path_n = os.path.join(path_n, "/kicked")

        ensure_dir(path_n)
        filename = os.path.join(path_n, patch_name)
        cv2.imwrite(filename, patch)
        logging.debug("OK: filename: " + filename)
        del patch

def create_image_patches(images):
    """
    Speichere alle Bilder
    :param images: gets a list of tuples with:
    [label_a, images[], label_b, images[]]

    :return:
    """
    patch_width = 600
    patch_height = 600
    img_count = 1

    for label, image_list in images:

        path_n = os.path.join(destination_root, label)
        for idx, image in enumerate(image_list):

            # read all images and transform it to a list of patches
            logging.debug("img count:" + str(img_count))
            img_count +=1
            file_n = str(label) + str(idx)
            crop_img = crop_edges(image)
            patches = create_patches(crop_img, patch_width, patch_height)

            # save all patches to disk
            save_patches_to_disk(patches, path_n, file_n)

            del image
            del crop_img
        del image_list
    del images


def is_patch_empty(image):
    """
    Tests whether an image patch has enough content or not
    :param image: image patch to test
    :return: True for empty. False for enough content
    """
    theshhold = 0.98

    is_empty = False
    # create a gray histogram
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

    # collect black and white shades
    black_shades = hist[0:180]
    white_shades = hist[216:256]

    white_sum = 0
    black_sum = 0
    height, width = image.shape[:2]
    pixel_amount = height * width

    for white_shade in white_shades:
        white_sum += white_shade[0:1][0]
    for black_shade in black_shades:
        black_sum += black_shade[0:1][0]

    white_amount = white_sum / pixel_amount
    black_amount = black_sum / pixel_amount
    logging.debug("black amount: " + str(black_amount))
    logging.debug("white amount: " + str(white_amount))
    if white_amount > theshhold:
        is_empty = True

    logging.debug("Is patch empty: " + str(is_empty))
    return is_empty


def patch_all_images():
    images = load_all_images(source_root)
    create_image_patches(images)


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

    source_root = 'data/raw/test/'
    destination_root = 'data/processed/transformed/'

    patch_all_images()
