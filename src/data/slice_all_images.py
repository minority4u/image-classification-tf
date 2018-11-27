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
from src.data.data_utils import load_all_images, set_config, create_patches

global source_root, destination_root
num_skipped_img = 0
images = 0


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


def save_patches_to_disk(patches, path_name, file_n):
    """
    Saves a list of patches to disk
    :param patches: list of patches from one image
    :param path_name: destination path
    :param file_n: filename (typically the image number)
    :return:
    """
    global num_skipped_img

    for idy, patch in enumerate(patches):
        path_n = path_name
        # label + image number + _ + idx patch number + .jpg
        patch_name = file_n + "_" + str(idy) + ".jpg"

        # define new destination for filtered images
        if is_patch_empty(patch):
            path_n = os.path.join(path_n, "/filter/")
            logging.info('Path: {}'.format(path_n))
            num_skipped_img += 1


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
            crop_img = image #crop_edges(image)
            patches = create_patches(crop_img, patch_width, patch_height)

            # save all patches to disk
            save_patches_to_disk(patches, path_n, file_n)

            del image
            del crop_img
        del image_list
    del images
    logging.info('images: {}'.format(img_count))
    logging.info('images skipped: {}'.format(num_skipped_img))



def is_patch_empty(image):
    """
    Tests whether an image patch has enough content or not
    :param image: image patch to test
    :return: True for empty. False for enough content
    """

    threshold = 0.98
    black_value_threshold = 216

    # create a gray histogram
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    logging.debug('shape of image : {} shape of gray : {}'.format(image.shape, gray.shape))

    # count all pixels
    pixel_amount = gray.shape[0] * gray.shape[1]
    # get number of pixels < black_value_threshold
    relevant_pixel = (gray < black_value_threshold).sum()
    # calculate percentage of irrelevant pixels
    irrelevant_percentage = 1 - (relevant_pixel / pixel_amount)

    logging.debug('pixel total : {} pixel relevant: {} irrelevant portion: {}'.format(pixel_amount, relevant_pixel,
                                                                                      irrelevant_percentage))

    return threshold < irrelevant_percentage


def patch_all_images():
    set_config(config)
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

    source_root = 'data/raw/classification_data/'
    destination_root = 'data/processed/transformed/'

    patch_all_images()
