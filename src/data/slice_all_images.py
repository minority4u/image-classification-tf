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
from src.data.preprocessing import create_slides

global source_root, destination_root

# each side
crop_precentage = 10

def save_image_slices(images):
    """
    Speichere alle Bilder
    :param images:
    :return:
    """
    slice_width = 600
    slice_height = 600
    img_count = 1
    for label, image_list in images:

        full_path = os.path.join(destination_root, label)
        full_path_kicked = os.path.join(destination_root, label, "kicked")
        ensure_dir(full_path)
        ensure_dir(full_path_kicked)
        for idx, image in enumerate(image_list):
            print("img count:" + str(img_count))
            img_count +=1
            height, width = image.shape[:2]
            margin_width = math.floor(width / 100 * crop_precentage)
            margin_height = math.floor(height / 100 * crop_precentage)
            #crop_img = image[margin_height:height - margin_height * 2, margin_width:width - margin_width * 2]
            #crop_img = image[margin_width:width - margin_width * 2, margin_height:height - margin_height * 2]

            y_size, x_size, chan = image.shape
            x_off = (x_size - margin_width) // 2
            y_off = (y_size - margin_height) // 2
            crop_img = image[margin_height:-margin_height, margin_width:-margin_width]



            file_n = str(label) + str(idx)
            crop_height = height-2*margin_height
            crop_width = width-2*margin_width

            slices = create_slides(crop_img, slice_width, slice_height)
            #slices = []
            #slices.append(crop_img)
            for idy, slice in enumerate(slices):
                is_slice_empty = test_is_image_empty(slice)
                slice_name = file_n + "_" + str(idy) + ".jpg"


                if is_slice_empty == False:
                    filename = os.path.join(full_path, slice_name)
                    cv2.imwrite(filename, slice)
                    print("OK: filename: " + filename)
                    del slice
                else:
                    filename = os.path.join(full_path_kicked, slice_name)
                    cv2.imwrite(filename, slice)
                    print("KICKED: filename: " + filename)
                    del slice

            del image
            del crop_img
        del image_list
    del images


def test_is_image_empty(image):
    """
    Tests whether an image slice has enough content or not
    :param image: image slice to test
    :return: True for empty. False for enough content
    """
    is_empty = False
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
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
    print("black amount: " + str(black_amount))
    print("white amount: " + str(white_amount))
    if white_amount > 0.98:
        is_empty = True

    print("Is slice empty: " + str(is_empty))
    return is_empty


def slice_all_images():
    images = load_all_images(source_root)
    save_image_slices(images)


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

    slice_all_images()
