from ninaeval.models.model import ClassifierModel, FeatureExtractor, TorchModel, NUM_GESTURES
import numpy as np
from kymatio import Scattering1D
import torch

#
# Advanced Classifiers
#

# Works well on baseline dataset, overfits on LogicalDatasetV1
class FullyConnectedNN(TorchModel):

    def define_model(self, dim_in):
        model = torch.nn.Sequential(
            torch.nn.BatchNorm1d(dim_in),
            torch.nn.Linear(dim_in, dim_in * 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(dim_in * 32, dim_in * 64),
            torch.nn.BatchNorm1d(dim_in * 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(dim_in * 64, NUM_GESTURES),
            torch.nn.Softmax(dim=1)
        )
        return model

    def forward_pass(self, sample):
        targets     = torch.LongTensor(sample[1]).to(self.device)
        predictions = self.model(sample[0].to(self.device))
        return torch.nn.functional.cross_entropy(predictions, targets), predictions

class FullyConnectedNNV2(TorchModel):

    def define_model(self, dim_in):
        model = torch.nn.Sequential(
            torch.nn.BatchNorm1d(dim_in),
            torch.nn.Linear(dim_in, dim_in * 8),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(dim_in * 8, NUM_GESTURES),
            torch.nn.Softmax(dim=1)
        )
        return model

    def forward_pass(self, sample):
        targets     = torch.LongTensor(sample[1]).to(self.device)
        predictions = self.model(sample[0].to(self.device))
        return torch.nn.functional.cross_entropy(predictions, targets), predictions


#
# Advanced Feature Extractors
#
class ScatteringFeat1D(FeatureExtractor):

    # (1, T) -> (1, P, T/2**J)
    #   Where P ~= 1 + J Q + J (J-1) Q / 2.
    #
    J = 5       # Account for translation up to 2^6 samples
    T = 200     # Number of samples per feature vector
    Q = 2       # Resolution per octave

    def __init__(self):
        self.scattering_transform = Scattering1D(self.J, self.T, self.Q)

    def extract_feature_point(self, raw_samples):
        #print(raw_samples)
        #print(raw_samples.shape)
        shape = raw_samples.shape
        raw_samples = np.reshape(raw_samples, (1, shape[1], shape[0]))
        raw_samples = torch.from_numpy(raw_samples).float()
        Sx = self.scattering_transform.forward(raw_samples)
        Sx = Sx.numpy()
        #Sx = np.sum(Sx, axis=2)
        Sx = np.reshape(Sx, (Sx.shape[1] * Sx.shape[2] * Sx.shape[3]))
        #print(Sx.shape)
        #exit()
        return Sx

    def global_setup(self, all_raw_samples):
        pass