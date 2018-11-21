#our web app framework!

#you could also generate a skeleton from scratch via
#http://flask-appbuilder.readthedocs.io/en/latest/installation.html

#Generating HTML from within Python is not fun, and actually pretty cumbersome because you have to do the
#HTML escaping on your own to keep the application secure. Because of that Flask configures the Jinja2 template engine 
#for you automatically.
#requests are objects that flask handles (get set post, etc)
from flask import Flask, render_template,request
from flask import jsonify
import yaml
from argparse import ArgumentParser
#scientific computing library for saving, reading, and resizing images
from scipy.misc import imsave, imread, imresize
import numpy as np
#for importing our keras model
import keras.models
#for regular expressions, saves time dealing with string data
import re

#system level operations (like loading files)
import sys 
#for reading operating system data
import os
#tell our app where our saved model is
sys.path.append(os.path.abspath("."))
from models.load import *
#initalize our flask app
app = Flask(__name__)
#global vars for easy reusability
global model, graph

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
print(args.config)
# Make sure the config exists
assert os.path.exists(
	args.config), "Config does not exist {}!, Please create a config.yml in root or set the path with --config.".format(
	args.config)
# Load config
params = yaml.load(open(args.config, "r"))
#initialize these variables
model, graph = init(params)

# neccessary for python 3
import base64


#decoding an image from base64 into raw representation
def convertImage(imgData1):
	imgstr = re.search(b'base64,(.*)',imgData1).group(1)
	#print(imgstr)
	with open('output.png','wb') as output:
		#output.write(imgstr.decode('base64'))
		output.write(base64.b64decode(imgstr))
	

@app.route('/')
def index():
	#initModel()
	#render out pre-built HTML file right on the index page
	return render_template("index.html")

@app.route('/update/', methods=['GET','POST'])
def update_model():
	print('reloading model and graph')
	model, graph = init()
	return jsonify({'update model': 'success'})

@app.route('/predict/',methods=['GET','POST'])
def predict():
	#whenever the predict method is called, we're going
	#to input the user drawn character as an image into the model
	#perform inference, and return the classification
	#get the raw data format of the image
	imgData = request.get_data()
	print(imgData)
	#encode it into a suitable format
	convertImage(imgData)
	print("debug")
	#read the image into memory
	x = imread('output.png',mode='L')
	#compute a bit-wise inversion so black becomes white and vice versa
	x = np.invert(x)
	#make it the right size
	x = imresize(x,(28,28))
	#imshow(x)
	#convert to a 4D tensor to feed into our model
	x = x.reshape(1,28,28,1)
	print("debug2")
	#in our computation graph
	with graph.as_default():
		#perform the prediction
		out = model.predict(x)
		#print(out)
		print(np.argmax(out,axis=1))
		print("debug3")
		#convert the response to a string
		response = np.array_str(np.argmax(out,axis=1))
		return response	
	

if __name__ == "__main__":
	#decide what port to run the app in
	port = int(os.environ.get('PORT', 5000))
	#run the app locally on the givn port
	app.run(host='0.0.0.0', port=port)
	#optional if we want to run in debugging mode
	#app.run(debug=True)
