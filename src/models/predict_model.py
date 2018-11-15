import os
import sys
import logging

sys.path.append(os.path.abspath("."))
import cv2
from src.data.preprocessing import create_slides
from models.load import init
from src.utils_io import Console_and_file_logger
from src.visualization.utils import create_reports
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
class_names = []


def load_image(path='data/raw/test/Fliesbilder/image001.jpg'):
    return cv2.imread(path)


def load_all_images(path_to_folder='data/raw/test/'):
    """
    recursive function, if it is called with path to testfiles
    it calls itself and takes the folder name as class name
    otherwise it loads all images from the class folder.
    
    folder should have this structure:
    root
        Classname
            Images (jpg)
    :param path_to_folder: e.g. /data/raw/test/
    :return: list of tuples (label, list of images)
    """

    # logging.debug('load_all_images')
    images = []
    for file in sorted(os.listdir(path_to_folder)):
        current_file = os.path.join(path_to_folder, file)
        if os.path.isdir(current_file):
            # class / label found
            class_names.append(file)
            images.append((file, load_all_images(current_file)))
        # logging.debug('current file: {}'.format(current_file))
        filename, file_extension = os.path.splitext(current_file)
        if file_extension == '.jpg':
            images.append(load_image(current_file))
    return images


def predict_single_slice(image):
    # resize and reshape with opencv
    x = imresize(image, (224, 224))
    x = x.reshape(1, 224, 224, 3)

    # in our computation graph
    with graph.as_default():
        # perform the prediction
        out = model.predict(x)
        # logging.debug('SinglePrediction: {}'.format(out))
        prediction = np.argmax(out, axis=1)
        return prediction


def predict_single_img(imgData):
    slice_predictions = []
    # class_names = config['all_target_names']

    logging.debug('shape original image: {}'.format(imgData.shape))
    slices = create_slides(imgData)

    for slice in slices:
        slice_predictions.append(predict_single_slice(slice))

    slice_pred_names = [class_names[int(cls)] for cls in slice_predictions]

    counter = Counter(slice_pred_names)
    logging.debug('Classes: {}'.format(counter))

    prediction_max = max(counter.items(), key=operator.itemgetter(1))[0]
    logging.debug('Max class: {}'.format(prediction_max))
    return prediction_max


def predict_imges(images):
    predictions = [predict_single_img(image) for image in images]
    return predictions


def inference():
    path_to_test_data = 'data/raw/test/'
    logging.debug('Inference with: {}'.format(path_to_test_data))

    inference_images = load_all_images(path_to_test_data)
    logging.debug('Classes found: {}'.format(len(inference_images)))
    # logging.debug('Images found: {}'.format(inference_images))

    results = [(label, predict_imges(images)) for label, images in inference_images]

    logging.info(results)

    test_pred = []
    test_label = []

    for label, list_of_pred in results:
        # class_names.append(label)
        if list_of_pred:
            for pred in list_of_pred:
                test_pred.append(pred)
                test_label.append(label)

    # logging.info('prediction: {}'.format(test_pred))
    # logging.info('label: {}'.format(test_label))
    # logging.info('length: {}'.format(len(test_label)))

    create_reports(ground_truth=test_label, predicted_classes=test_pred, class_names=class_names, config=config)

    print(results)

    # predict_single_img(inference_images[0])

    # for img, cls in results:
    #     #cv2.imshow(img)
    #     logging.info(cls)


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

    inference()
