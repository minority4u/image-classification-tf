import os
import sys
import logging

sys.path.append(os.path.abspath("."))
import cv2
from src.data.data_utils import create_patches, load_image
from models.load import init
from src.utils_io import Console_and_file_logger, parameter_logger
from src.visualization.utils import create_reports
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




@parameter_logger
def predict_single_slice(image):
    # resize and reshape with opencv

    #x = imresize(image, (224, 224))
    #x = cv2.resize(image, (224, 224))


    x = image.reshape(1, 224, 224, 3)

    # in our computation graph
    with graph.as_default():
        # perform the prediction
        out = model.predict(x)
        # logging.debug('SinglePrediction: {}'.format(out))
        prediction = np.argmax(out, axis=1)
        return prediction

@parameter_logger
def predict_single_img(imgData, resize=False):
    patch_predictions = []
    # class_names = config['all_target_names']

    logging.debug('shape original image: {}'.format(imgData.shape))
    patch_width = 600
    patch_height = 600
    patches = create_patches(imgData, patch_width, patch_height, resize=resize)

    for patch in patches:
        patch_predictions.append(predict_single_slice(patch))

    slice_pred_names = [class_names[int(cls)] for cls in patch_predictions]

    counter = Counter(slice_pred_names)
    logging.debug('Classes: {}'.format(counter))

    prediction_max = max(counter.items(), key=operator.itemgetter(1))[0]
    logging.debug('Max class: {}'.format(prediction_max))
    return prediction_max

@parameter_logger
def external_predict_image(image, mod, gra, conf, resize=False):
    # wrapper function to reuse the loaded model + graph
    global model
    global graph
    global config
    config = conf
    model = mod
    graph = gra
    return predict_single_img(image, resize=resize)


def predict_imges(images, resize=False):
    predictions = [predict_single_img(image, resize=resize) for image in images]
    return predictions

def external_predict_images(images, mod, gra, conf, resize=False):
    # wrapper function to reuse the loaded model + graph
    global model
    global graph
    global config
    config = conf
    model = mod
    graph = gra
    return predict_imges(images, resize=resize)



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

    prediction_image = load_image('/data/processed/predict.jpg')
    result = predict_single_img(prediction_image)
    logging.info('Predition: {}'.format(result))
