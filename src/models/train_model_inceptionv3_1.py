import os
import sys

sys.path.append(os.path.abspath("."))
print(sys.path)
from src.utils_io import Console_and_file_logger, ensure_dir, parameter_logger
import logging
from argparse import ArgumentParser
import yaml
from src.models.v3_model import *
from src.data.make_dataset import get_train_and_validation_generator
import json
from sklearn.metrics import classification_report, confusion_matrix
from time import time
import numpy as np
import matplotlib.pyplot as plt
import itertools
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical


def plot_history(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]

    if len(loss_list) == 0:
        print('Loss is missing in history')
        return

        ## As loss always exists
    epochs = range(1, len(history.history[loss_list[0]]) + 1)

    ## Loss
    plt.figure(3)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b',
                 label='Training loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g',
                 label='Validation loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))

    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    ## Accuracy
    plt.figure(3)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'r',
                 label='Training accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')
    for l in val_acc_list:
        plt.plot(epochs, history.history[l], 'c',
                 label='Validation accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('history.png')
    plt.clf()


def plot_confusion_matrix(cm, classes, pathtosave,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(pathtosave)
    plt.clf()


def train(config):
    """
    training entrance, all heavy work is done here
    :param config: Json representation of our config file
    :return:
    """

    logging.info('training starts')

    # get train generator
    train_generator, validation_generator = get_train_and_validation_generator(path_to_data=config['data_dir'],
                                                                               validation_split=config[
                                                                                   'validation_split'],
                                                                               image_size=(config['input_image_width'],
                                                                                           config[
                                                                                               'input_image_height']),
                                                                               batch_size=config['batch_size'],
                                                                               class_mode=config['class_mode'])

    valdir = config['data_dir'] + "Validation/"
    traindir = config['data_dir'] + "Training/"

    img_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_generator2 = img_datagen.flow_from_directory(
        valdir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')
    train_generator2 = img_datagen.flow_from_directory(
        traindir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

    # get model
    aliases, model = get_model()

    # compile model
    model.compile(loss=config['loss_function'],
                  optimizer=get_optimizer(),
                  metrics=config['metrics'])

    img_datagen2 = ImageDataGenerator(rescale=1. / 255)
    itr = img_datagen2.flow_from_directory(valdir, target_size=(224, 224), batch_size=600, class_mode='categorical')
    endX, endY = itr.next()

    # model fit with generator
    history = model.fit_generator(train_generator2, steps_per_epoch=int(config['steps_per_epoch']),
                                  epochs=int(config['epochs']), verbose=2, callbacks=None,
                                  validation_data=validation_generator2,
                                  validation_steps=None, class_weight=None, max_queue_size=10, workers=1,
                                  use_multiprocessing=False,
                                  shuffle=False, initial_epoch=0)

    # Confusion Matrix & Reports
    print('Classes: {0}'.format(len(validation_generator2.class_indices)))
    if len(validation_generator2.class_indices) == 3:
        target_names = ['Etechnik', 'Fliesbilder', 'Tabellen']

    if len(validation_generator2.class_indices) == 5:
        target_names = ['Etechnik', 'Fliesbilder', 'Lageplan', 'Stahlbau', 'Tabellen']

    validation_generator2.reset()

    predictions = model.predict(endX)
    print("Validation Data")
    print("endY:{0}".format(endY))
    predicted_classes = np.argmax(predictions, axis=1)
    print("predicted_classes:{0}".format(predicted_classes))

    # predictions = model.predict_generator(validation_generator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
    print("predictions {0}".format(predictions))
    ground_truth = endY
    ground_truth = np.argmax(ground_truth, axis=1)
    print('ground truth {0}'.format(ground_truth))
    cm = confusion_matrix(ground_truth, predicted_classes)
    print(classification_report(ground_truth, predicted_classes, target_names=target_names))
    # plot_confusion_matrix(cm, classes = target_names)

    plt.figure()
    plot_confusion_matrix(cm, classes=target_names, normalize=False,
                          title='Confusion matrix, without normalization', pathtosave='cm.png')

    plt.figure()
    plot_confusion_matrix(cm, classes=target_names, normalize=True,
                          title='Normalized confusion matrix', pathtosave='cmnormalized.png')

    # Save the model
    # serialize model to JSON
    model_json = model.to_json()
    model_path = config['model_path']
    ensure_dir(model_path)
    with open(os.path.join(model_path, 'model.json'), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("models/model.h5")
    logging.info("Saved model to disk: {}".format(model_path))

    print('History:')
    print(history.history)
    plot_history(history)


if __name__ == '__main__':
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

    # Make sure that source and destination are set
    assert {"batch_size", "epochs", "data_dir", "experiment_name"} <= set(
        params.keys()), "Configuration is incomplete! Please define dir_to_src and dir_to_dest in config.yml"

    # Make sure source folder exists
    assert os.path.exists(params["data_dir"]), "Path to src {} does not exist!".format(params["data_dir"])

    # Define central logger, set name and logging level
    Console_and_file_logger(logfile_name=params["experiment_name"], log_lvl="INFO")
    logging.info('Starting experiment {}'.format(params["experiment_name"]))
    logging.info(json.dumps(params, indent=2))

    train(params)