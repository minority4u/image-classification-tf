import numpy as np
import keras.models
import logging
from keras.models import model_from_json
from src.models.model_manager import get_optimizer
from scipy.misc import imread, imresize,imshow
import tensorflow as tf
import os


def init(config):
	model_path = os.path.join(os.getcwd(), config['model_json'])
	json_file = open(model_path,'r')
	loaded_model_json = json_file.read()
	json_file.close()
	# load the json file
    #json_file = open('models/model.json','r')


	# convert json to keras model file
	loaded_model = model_from_json(loaded_model_json)
	#load woeights into new model
	loaded_model.load_weights(config['model_h5'])
	logging.info("Loaded Model from disk")

	#compile and evaluate loaded model
	loaded_model.compile(loss=config['loss_function'],
						 optimizer=get_optimizer(),
						 metrics=config['metrics'])
	#loss,accuracy = model.evaluate(X_test,y_test)
	#print('loss:', loss)
	#print('accuracy:', accuracy)
	graph = tf.get_default_graph()

	return loaded_model,graph