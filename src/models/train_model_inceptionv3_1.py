import logging
import os
from src.utils_io import Console_and_file_logger
from argparse import ArgumentParser
import yaml



def train():
    logging.info('training starts')







def main():
    train()


if __name__ == '__main__':
    # Define argument parser
    parser = ArgumentParser()

    # define arguments and default values to parse
    parser.add_argument("--config", "-c", help="Define the path to config.yml", default="config.yml", required=False)

    args = parser.parse_args()

    # Make sure the config exists
    assert os.path.exists(
        args.config), "Config does not exist!, Please create a config.yml in root or set the path with --config."

    # Load config
    params = yaml.load(open(args.config, "r"))

    # Make sure that source and destination are set
    assert {"dir_to_src", "dir_to_dest"} <= set(
        params.keys()), "Configuration is incomplete! Please define dir_to_src and dir_to_dest in config.yml"

    # Make sure source folder exists
    assert os.path.exists(params["dir_to_src"]), "Path to src {} does not exist!".format(params["dir_to_src"])

    Console_and_file_logger('Train_inception_v3')
    main()