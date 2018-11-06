import logging
import os
from src.utils_io import Console_and_file_logger, ensure_dir
from argparse import ArgumentParser
import yaml
from src.models.v3_model import *
from src.data.make_dataset import GetTrainGenerator, GetValidationGenerator
import json



def train(config):
    logging.info('training starts')
    x_train = {}
    y_train = {}
    x_test = {}
    y_test = {}


    # get train generator
    train_generator = GetTrainGenerator(config.data_dir)
    validation_generator = GetValidationGenerator(config.data_dir)

    # get model
    aliases, model = get_model()


    # compile model
    model.compile(loss=get_loss_function(),
                  optimizer=get_optimizer(),
                  metrics=['accuracy'])



    # model fit
    #model.fit(x_train, y_train,batch_size=config.batch_size,epochs=config.epochs,verbose=1,validation_data=(x_test, y_test))

    # model fit with generator
    model.fit_generator(train_generator, steps_per_epoch=config.batch_size, epochs=config.epochs, verbose=2, callbacks=None, validation_data=validation_generator,
                  validation_steps=None, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False,
                  shuffle=True, initial_epoch=0)



    # get score
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # evaluate with generator
    model.evaluate_generator(validation_generator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)


    # Save the model
    # serialize model to JSON
    model_json = model.to_json()
    model_path = config.model_path
    ensure_dir(model_path)
    with open("models/inception_v3/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("models/model.h5")
    print("Saved model to disk")







def main():
    train()


if __name__ == '__main__':
    # Define argument parser

    logging.info('loading config')
    parser = ArgumentParser()

    # define arguments and default values to parse
    # define tha path to your config file
    parser.add_argument("--config", "-c", help="Define the path to config.yml", default="./config/experiments/inception_v3_base.yml", required=False)

    args = parser.parse_args()

    # Make sure the config exists
    assert os.path.exists(
        args.config), "Config does not exist!, Please create a config.yml in root or set the path with --config."

    # Load config
    params = yaml.load(open(args.config, "r"))

    # Make sure that source and destination are set
    assert {"batch_size", "epochs", "data_dir"} <= set(
        params.keys()), "Configuration is incomplete! Please define dir_to_src and dir_to_dest in config.yml"

    # Make sure source folder exists
    assert os.path.exists(params["data_dir"]), "Path to src {} does not exist!".format(params["data_dir"])

    logging.info(json.dumps(params, indent=2))

    logging.info(('old config:'))
    logging.info(json.dumps({"kfold": 1, "numPorts": 1, "samples": {"validation": 450, "training": 2100, "split": 3, "test": 450}, "datasetLoadOption": "batch", "mapping": {"Filename": {"port": "InputPort0", "type": "Image", "shape": "", "options": {"horizontal_flip": False, "Height": "224", "rotation_range": 0, "vertical_flip": False, "width_shift_range": 0, "Normalization": False, "Width": "224", "shear_range": 0, "pretrained": "None", "Scaling": 1, "Augmentation": False, "Resize": True, "height_shift_range": 0}}, "Label": {"port": "OutputPort0", "type": "Categorical", "shape": "", "options": {}}}, "dataset": {"samples": 3000, "name": "Classify1000", "type": "private"}, "shuffle": True}, indent=2))


    Console_and_file_logger('Train_inception_v3')
    #main(params)
    #train(params)