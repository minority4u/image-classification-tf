import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix


model = load_trained_model('.\model.h5')
probabilities = model.predict_generator(generator, 2000)

def load_trained_model(weights_path):
   model = create_model()
   model.load_weights(weights_path)


def print_confusion_matrix(thismodel):
    Y_pred = thismodel.predict_generator(validation_generator, num_of_test_samples // batch_size + 1)
    y_pred = np.argmax(Y_pred, axis=1)
    print('Confusion Matrix')
    print(confusion_matrix(validation_generator.classes, y_pred))
    print('Classification Report')
    target_names = ['Cats', 'Dogs', 'Horse']
    print(classification_report(validation_generator.classes, y_pred, target_names=target_names))