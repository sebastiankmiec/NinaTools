from abc import ABC, abstractmethod

class ClassifierModel(ABC):

    @abstractmethod
    def train_model(self, train_features, train_labels):
        pass

    @abstractmethod
    def perform_inference(self, test_features, test_labels):
        """
            Given test features and labels, compute predictions and classifier accuracy,

        :param test_features:Features from the test split.
        :param test_labels: Labels from the test split.
        :return: Classifier accuracy from 0 ~ 1.0.
        """
        pass

    def classifier_accuracy(self, predictions, test_labels):
        errors      = predictions == test_labels
        acc_rate    = len([x for x in errors if (x == True)]) / len(errors)
        return acc_rate


class FeatureExtractor(ABC):

    @abstractmethod
    def extract_feature_point(self, raw_samples):
        """
        :param raw_samples: A window of emg samples.
        :return: A single feature point.
        """
        pass

    @abstractmethod
    def global_setup(self, all_raw_samples):
        """
              :param all_raw_samples: All windows of emg samples.
        """
        pass