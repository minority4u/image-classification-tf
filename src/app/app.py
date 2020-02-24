# our web app framework!

# you could also generate a skeleton from scratch via
# http://flask-appbuilder.readthedocs.io/en/latest/installation.html

# Generating HTML from within Python is not fun, and actually pretty cumbersome because you have to do the
# HTML escaping on your own to keep the application secure. Because of that Flask configures the Jinja2 template engine
# for you automatically.
# requests are objects that flask handles (get set post, etc)
# system level operations (like loading files)
import sys
# for reading operating system data
import os
import numpy as np

# tell our app where our saved model is
sys.path.append(os.path.abspath("."))
from flask import Flask, render_template, request
from flask import jsonify
import yaml
import json
from imageio import imread
from argparse import ArgumentParser
from src.utils_io import parameter_logger, Console_and_file_logger
from src.models.predict_model import external_predict_images


from models.load import *
from time import time
# Define central logger, set name and logging level
Console_and_file_logger(logfile_name='app', log_lvl="INFO")

# initalize our flask app
app = Flask(__name__)
# global vars for easy reusability
global model, graph, config

# Define argument parser
parser = ArgumentParser()

# define arguments and default values to parse
# define tha path to your config file
config_path = os.path.join(os.getcwd(), 'config/app/inception_v3_base.yml')
parser.add_argument("--config", "-c", help="Define the path to config.yml",
                    default=config_path, required=False)

parser.add_argument("--working_dir", help="Define the absolute path to the project root",
                    default="../../", required=False)
# parser.add_argument("--modelskiptraining", help="Skip Training", default="None", required=False)

args = parser.parse_args()
logging.info(args.config)
logging.info(os.environ['PATH'])
logging.info('working app dir: {}'.format(os.getcwd()))
# Make sure the config exists
assert os.path.exists(
    args.config), "Config does not exist {} !, Please create a config.yml in root or set the path with --config.".format(
    args.config)
# Load config
params = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
# initialize these variables
model, graph = init(params)

config = params
# neccessary for python 3
import base64


# decoding an image from base64 into raw representation
@parameter_logger
def convertImage(img):

    if config['color_mode'] == 'grayscale':
        return np.dot(img[...,:3], [0.299, 0.587, 0.114])
    elif config['color_mode'] == 'rgb':
        return img



@app.route('/')
def index():
    # initModel()
    # render out pre-built HTML file right on the index page
    return render_template("upload.html")


@app.route('/update/', methods=['GET', 'POST'])
def update_model():
    print('reloading model and graph')
    global model, graph
    model, graph = init(params)
    return jsonify({'update model': 'success'})

#@parameter_logger
@app.route('/predict/', methods=['POST'])
def predict():
    # whenever the predict method is called, we're going
    # to input the user drawn character as an image into the model
    # perform inference, and return the classification
    # get the raw data format of the image
    global model, graph

    file = request.files['image']
    logging.debug(file)
    image = imread(file)
    image = convertImage(image)

    x = image.astype(np.uint8)

    logging.debug(x)

    t1 = time()
    try:
        result = external_predict_images([x], '', model, graph, config, resize=True)

        logging.debug(result)
        prediction = result.get_image_results_as_class_name()
        prob = json.dumps(result.get_probabillities())
        patches = result.get_number_of_patches()
    except Exception as e:
        logging.error(str(e))
        prediction = 'Internal error {}'.format(str(e))
        prob = '0.0'
        patches = '0'



    resp = {}
    resp['result'] = str(prediction)
    resp['time'] = str((time()-t1))
    resp['probability'] = prob
    resp['patches'] = patches

    response = jsonify(resp)
    logging.debug(response)
    return response



if __name__ == "__main__":
    # decide what port to run the app in
    port = int(os.environ.get('PORT', 5000))
    # run the app locally on the givn port
    app.run(debug=True, host='0.0.0.0', port=port)
    # optional if we want to run in debugging mode
    #app.run(debug=True)
