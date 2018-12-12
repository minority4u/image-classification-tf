import numpy as np
from collections import Counter
import operator
from src.data.data_utils import get_class_names
import logging


class Result():
    image_results = []
    class_names = get_class_names()

    def __init__(self, image_results):
        self.image_results = image_results

    def get_image_results_as_indicies(self):
        return [image.get_prediction() for image in self.image_results]

    def get_image_truth(self):
        return [image.get_truth() for image in self.image_results]

    def get_patch_results_as_indicies(self):
        patch_results = []
        [patch_results.extend(image.get_patch_predictions()) for image in self.image_results]
        return patch_results

    def get_patch_results_as_class_name(self):
        patch_results = self.get_patch_results_as_indicies()
        return [self.class_names[int(cls)] for cls in patch_results]

    def get_patch_truth(self):
        patch_truth = []
        [patch_truth.extend(image.get_patch_truth()) for image in self.image_results]
        return  patch_truth

    def get_image_results_as_class_name(self):
        image_results = self.get_image_results_as_indicies()
        return [self.class_names[int(cls)] for cls in image_results]

    def get_class_names(self):
        return self.class_names

    def get_number_of_patches(self):
        return [('image_' + str(idx) + " = " + str(len(image.patch_results))) for idx, image in enumerate(self.image_results)]

    def get_probabillities(self):
        result_probs = []
        cls = self.get_class_names()
        probs = [image.get_image_prob() for idx, image in enumerate(self.image_results)]
        for idx, img in enumerate(self.image_results):
            p = {}
            for i, c in enumerate(cls):
                p[c] = str(probs[idx][i])
            result_probs.append(p)
            logging.debug(result_probs)

        return result_probs



class ImageResult():
    patch_results = []
    image_prob = {}

    def __init__(self, patch_results):
        self.patch_results = patch_results

    def get_prediction(self):
        patch_classes = [patch.get_prediction() for patch in self.patch_results]
        counter = Counter(patch_classes)
        return max(counter.items(), key=operator.itemgetter(1))[0]

    def get_truth(self):
        return self.patch_results[0].get_truth()

    def get_patch_predictions(self):
        return [patch.get_prediction() for patch in self.patch_results]

    def get_patch_truth(self):
        return [patch.get_truth() for patch in self.patch_results]

    def get_image_prob(self):

        probs = [patch.get_prediction_probability()[0] for patch in self.patch_results]
        #print(probs)
        np_probs = np.array(probs)
        #print(np_probs)
        avg_probs = np.mean(np_probs, axis=0)
        #print(avg_probs)

        return avg_probs


class PatchResult():
    prob = {}
    ground_truth = ''

    def __init__(self, class_prob, ground_truth):
        self.prob = class_prob

        self.ground_truth = ground_truth

    def get_prediction_probability(self):
        return self.prob

    def get_prediction(self):
        return int(np.argmax(self.prob, axis=1))

    def get_truth(self):
        return self.ground_truth
