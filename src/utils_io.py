import os
import logging
from time import time
import platform
if not platform.system() == 'Windows':
    import matplotlib as mpl
    mpl.use('TkAgg')
import matplotlib.pyplot as plt

# define some helper classes
# Define an individual logger
class Console_and_file_logger():
    def __init__(self, logfile_name='Log', log_lvl = logging.INFO, path='./reports/logs/'):
        """
        Create your own logger
        prints all messages into the given logfile and ouput it on console
        :param logfile_name:
        :param log_dir:
        """

        # Define the general formatting schema
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        logger = logging.getLogger()
        if not logger.handlers:

            # Create log directory
            #log_dir = os.getcwd() + path
            log_dir = path
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            # Define logfile handler and file
            hdlr = logging.FileHandler(os.path.join(log_dir,logfile_name + '.log'))
            hdlr.setFormatter(formatter)

            # Define console output handler
            hdlr_console = logging.StreamHandler()
            hdlr_console.setFormatter(formatter)

            # Add both handlers to our logger instance
            logger.addHandler(hdlr)
            logger.addHandler(hdlr_console)
            logger.setLevel(log_lvl)

            cwd = os.getcwd()
            print('Working directory: {}.'.format(cwd))
            print('Log dir: ' + log_dir)

            logging.info('{} {} {}'.format('--' * 10, 'Start', '--' * 10))
            logging.info('Filename: {}'.format(logfile_name))
            logging.info('Log directory: {}'.format(log_dir))


def ensure_dir(file_path):
    """
    Make sure a directory exists or create it
    :param file_path:
    :return:
    """
    if not os.path.exists(file_path):
        logging.debug('Creating directory {}'.format(file_path))
        os.makedirs(file_path)

def save_plot(fig, path, filename=''):
    """
    Saves an matplotlib figure to the given path + filename
    If the figure exists, ad a number at the end and increase it
    as long as there is already an image with this name
    :param fig:
    :param path:
    :param filename:
    :return:
    """
    logging.debug('Trying to save to {0}'.format(path))
    ensure_dir(path)
    plt.tight_layout()

    i = 0
    while True:
        i += 1
        newname = '{}{:d}.png'.format(filename + '_', i)
        if os.path.exists(os.path.join(path , newname)):
            continue
        fig.savefig(os.path.join(path , newname))
        break
    logging.debug('Image saved: {}'.format(os.path.join(path , newname)))
    # free memory, close fig
    plt.close(fig)

def parameter_logger(func):
    def function_wrapper(*args, **kwargs):
        t1 = time()
        logging.debug("Before calling {}".format(func.__name__))
        logging.debug('Args: {}, Kwargs'.format(*args, **kwargs))
        res = func(*args, **kwargs)
        logging.debug(res)
        logging.debug("{} done in {:0.3f}s.".format(func.__name__, time() - t1))

    return function_wrapper