from ninaeval.models.model import ClassifierModel, FeatureExtractor, TorchModel, NUM_GESTURES
import numpy as np
from kymatio import Scattering1D
import torch
from sklearn.decomposition import PCA

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
        shape = raw_samples.shape
        raw_samples = np.reshape(raw_samples, (1, shape[1], shape[0]))
        raw_samples = torch.from_numpy(raw_samples).float()
        Sx = self.scattering_transform.forward(raw_samples)
        Sx = Sx.numpy()
        Sx = np.reshape(Sx, (Sx.shape[1] * Sx.shape[2] * Sx.shape[3]))
        return Sx

    def global_setup(self, all_raw_samples):
        pass

class Testing(FeatureExtractor):

    requires_global_setup   = False
    custom_mask             = []

    # def __init__(self):
    #
    #     #
    #     # base_mask = [0, 1, ..., 10, 11 (appears 10 times), 12 (appears 10 times), ..., 19 (appears 10 times)]
    #     #
    #     final_mask  = []
    #     base_mask   = [0]
    #     base_mask.extend([1 for x in range(5)])
    #     base_mask.extend([2 for x in range(5)])
    #     base_mask.extend([3 for x in range(5)])
    #     base_mask.extend([4 for x in range(10)])
    #
    #     for j in range(16):
    #         if j > 0:
    #             for k in range(len(base_mask)):
    #                 base_mask[k] += 5
    #         final_mask.extend(base_mask)
    #     self.custom_mask = final_mask
    #
    # def extract_feature_point(self, raw_samples):
    #     split_data      = np.split(raw_samples, indices_or_sections = 4, axis=0)
    #     mag_feat_list   = np.array([])
    #
    #     for data in split_data:
    #         ft_feat     = np.fft.rfft(data, axis=0, norm="ortho")
    #         mag_feat    = np.abs(ft_feat)
    #         mag_feat    = mag_feat.flatten("F")
    #         mag_feat    = np.bincount(self.custom_mask, weights=mag_feat)
    #         mag_feat_list = np.append(mag_feat_list, mag_feat)
    #
    #     return mag_feat_list
    #
    # def global_setup(self, all_raw_samples):
    #     # shape = all_raw_samples.shape
    #     # sample  = np.reshape(all_raw_samples, (shape[0], shape[1]*shape[2]))
    #     # #sample  = np.reshape(sample, (40*16))
    #     # pca = PCA()
    #     # z   = pca.fit(sample)
    #     pass


    def __init__(self):

        #
        # base_mask = [0, 1, ..., 10, 11 (appears 10 times), 12 (appears 10 times), ..., 19 (appears 10 times)]
        #
        final_mask  = []
        base_mask   = [0]
        base_mask.extend([1 for x in range(10)])
        base_mask.extend([2 for x in range(10)])
        base_mask.extend([3 for x in range(20)])
        base_mask.extend([4 for x in range(20)])
        base_mask.extend([5 for x in range(40)])

        for j in range(16):
            if j > 0:
                for k in range(len(base_mask)):
                    base_mask[k] += 6
            final_mask.extend(base_mask)

        self.custom_mask = final_mask

    def extract_feature_point(self, raw_samples):
        ft_feat     = np.fft.rfft(raw_samples, axis=0, norm="ortho")
        mag_feat    = np.abs(ft_feat)
        mag_feat    = mag_feat.flatten("F")
        mag_feat    = np.bincount(self.custom_mask, weights=np.array(mag_feat))
        return mag_feat

    def global_setup(self, all_raw_samples):
        # shape = all_raw_samples.shape
        # sample  = np.reshape(all_raw_samples, (shape[0], shape[1]*shape[2]))
        # #sample  = np.reshape(sample, (40*16))
        # pca = PCA()
        # z   = pca.fit(sample)
        pass