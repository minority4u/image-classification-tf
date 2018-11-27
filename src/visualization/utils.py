import matplotlib.pyplot as plt
import numpy as np
import itertools
import logging
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from src.utils_io import save_plot
from collections import Counter

def plot_history(history):
    """
    Plots Training an d Validation History
    :param history: Return Value of Model.Fit (Keras)
    :return: none
    """
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]

    if len(loss_list) == 0:
        logging.debug('Loss is missing in history')
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


def plot_confusion_matrix(cm, classes, path_to_save, filename,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
     This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    :param cm: Confusion Matrix (SKlearn)
    :param classes: One-Hot encoded classes
    :param pathtosave:  path to store image
    :param normalize:
    :param title:
    :param cmap:
    :return: none
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

    save_plot(plt.gcf(), path_to_save, filename)

    #plt.tight_layout()
    #plt.savefig(path_to_save)
    #plt.clf()

def create_reports(ground_truth, predicted_classes, class_names, config, report_name, f_name_suffix):
    """
    Create validation report (confusion matrix, train/val history)
    :param ground_truth: One-Hot encoded truth (validation_generator)
    :param predicted_classes: Predicted results of model.predict...
    :param validation_generator: validation_generator with batch size = testsize
    :param config: config of experiment to read paths
    :return: none
    """
    logging.info('Classes: {0}'.format(len(class_names)))
    path_to_save = os.path.join(config['report_path'], report_name)


    target_names = class_names
    counter = Counter(target_names)
    max_val = float(max(counter.values()))
    class_weights = {class_id: max_val / num_images for class_id, num_images in counter.items()}

    cm = confusion_matrix(y_true=ground_truth, y_pred=predicted_classes, sample_weight=class_weights)

    logging.info('\n' + classification_report(y_true=ground_truth, y_pred=predicted_classes, target_names=target_names, sample_weight=class_weights))
    logging.info('Accuracy: {}'.format(accuracy_score(ground_truth, predicted_classes)))

    plt.figure()
    plot_confusion_matrix(cm, classes=target_names, normalize=False,
                          title='Confusion matrix, without normalization', path_to_save=path_to_save, filename=f_name_suffix + '_confusion_matrix.png')

    plt.figure()
    plot_confusion_matrix(cm, classes=target_names, normalize=True,
                          title='Normalized confusion matrix', path_to_save=path_to_save,filename = f_name_suffix + '_confusion_matrix_normalized.png')