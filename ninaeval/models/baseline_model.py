from ninaeval.models.model import ClassifierModel, FeatureExtractor
import numpy as np
from sklearn.ensemble import RandomForestClassifier

#
# Baseline Classifiers
#
class RandomForest(ClassifierModel):

    num_trees = 128

    def __init__(self):
        self.classifier = RandomForestClassifier(n_estimators=self.num_trees)

    def train_model(self, train_features, train_labels):
        self.classifier.fit(train_features, train_labels)

    def perform_inference(self, test_features, test_labels):
        predictions = self.classifier.predict(test_features)
        return self.classifier_accuracy(predictions, test_labels)

#
# Baseline Feature Extractors
#
class RMS(FeatureExtractor):

    def extract_feature_point(self, raw_samples):
        return np.sqrt(np.mean(np.square(raw_samples), axis=0))

