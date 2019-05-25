from ninaeval.models.model import ClassifierModel, FeatureExtractor, TorchModel, NUM_GESTURES
import numpy as np
from kymatio import Scattering1D
import torch
from sklearn.decomposition import PCA

import torch
import torch.nn.functional as F
import torch.nn as nn

import scipy.cluster.vq as sci_c


########################################################################################################################
########################################################################################################################
########################################################################################################################
#
# Advanced Classifiers
#
########################################################################################################################
########################################################################################################################
########################################################################################################################

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


# Performs well on own data
#
# class FullyConnectedNNV2(TorchModel):
#
#     def define_model(self, dim_in):
#         model = torch.nn.Sequential(
#             torch.nn.BatchNorm1d(dim_in),
#             torch.nn.Linear(dim_in, dim_in * 4),
#             torch.nn.LeakyReLU(),
#             torch.nn.Linear(dim_in * 4, NUM_GESTURES),
#             torch.nn.Softmax(dim=1)
#         )
#         return model
#
#     def forward_pass(self, sample):
#         targets     = torch.LongTensor(sample[1]).to(self.device)
#         predictions = self.model(sample[0].to(self.device))
#         return torch.nn.functional.cross_entropy(predictions, targets), predictions


# A variation of the above "FullyConnectedNNV2"
class FullyConnectedNNV2(TorchModel):

    def define_model(self, dim_in):
        model = torch.nn.Sequential(
            torch.nn.BatchNorm1d(dim_in),
            torch.nn.Linear(dim_in, dim_in * 2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(dim_in * 2, dim_in * 4),
            torch.nn.BatchNorm1d(dim_in * 4),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(dim_in * 4, dim_in * 2),
            torch.nn.BatchNorm1d(dim_in * 2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(dim_in * 2, dim_in),
            torch.nn.BatchNorm1d(dim_in),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(dim_in, NUM_GESTURES),
            torch.nn.Softmax(dim=1)
        )
        return model

    def forward_pass(self, sample):
        targets     = torch.LongTensor(sample[1]).to(self.device)
        predictions = self.model(sample[0].to(self.device))
        return torch.nn.functional.cross_entropy(predictions, targets), predictions


class CustomNet(torch.nn.Module):
    """
        The classifier with the best known performance on the NinaPro dataset thus far (using a variation of
            PaddedMultiRMS).
    """

    def __init__(self, dim_in):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(CustomNet, self).__init__()
        self.batchnorm_layer  = torch.nn.Sequential(torch.nn.BatchNorm1d(dim_in))
        self.layer_1   = torch.nn.Sequential(
                                torch.nn.Linear(dim_in, dim_in * 2),
                                torch.nn.LeakyReLU(),
                                )

        self.layer_2   = torch.nn.Sequential(
                                    torch.nn.Linear(dim_in * 2, dim_in * 4),
                                    torch.nn.BatchNorm1d(dim_in * 4),
                                    torch.nn.LeakyReLU())


        self.layer_3   = torch.nn.Sequential(
                                    torch.nn.Linear(dim_in * 4, dim_in * 8),
                                    torch.nn.BatchNorm1d(dim_in * 8),
                                    torch.nn.LeakyReLU())

        self.layer_4 = torch.nn.Sequential(
                                    torch.nn.Linear(dim_in * 8, dim_in * 4),
                                    torch.nn.BatchNorm1d(dim_in * 4),
                                    torch.nn.LeakyReLU())


        self.layer_5   = torch.nn.Sequential(
                                                torch.nn.Linear(2 * (dim_in * 4), dim_in * 2),
                                                torch.nn.BatchNorm1d(dim_in * 2),
                                                torch.nn.LeakyReLU())

        self.layer_6 = torch.nn.Sequential(
                                            torch.nn.Linear(2 * (dim_in * 2), dim_in),
                                            torch.nn.BatchNorm1d(dim_in),
                                            torch.nn.LeakyReLU())

        self.final_layer = torch.nn.Sequential(
                                                torch.nn.Linear(2 * (dim_in), NUM_GESTURES),
                                                torch.nn.Softmax(dim=1)
                                                )
    def forward(self, x):
        y0 = self.batchnorm_layer(x)
        y1 = self.layer_1(y0)
        y2 = self.layer_2(y1)
        y3 = self.layer_3(y2)
        y4 = self.layer_4(y3)
        y5 = self.layer_5(torch.cat((y4, y2), dim=1))
        y6 = self.layer_6(torch.cat((y5, y1), dim=1))
        y7 = self.final_layer(torch.cat((y6, y0), dim=1))
        return y7

#
# Yet another variation of FullyConnectedNNV2, leveraging the CustomNet module
#
# class FullyConnectedNNV2(TorchModel):
#     def define_model(self, dim_in):
#         model = CustomNet(dim_in)
#         return model
#
#     def forward_pass(self, sample):
#         targets     = torch.LongTensor(sample[1]).to(self.device)
#         predictions = self.model(sample[0].to(self.device))
#         return torch.nn.functional.cross_entropy(predictions, targets), predictions





########################################################################################################################
########################################################################################################################
########################################################################################################################
#
# Advanced Feature Extractors
#
########################################################################################################################
########################################################################################################################
########################################################################################################################

class ScatteringFeat1D(FeatureExtractor):
    """
        Refer to (as well papers citing this paper):
            "J. Bruna and S. Mallat. Invariant scattering convolution networks.IEEE Transactions on Pattern Analysis
             and Machine Intelligence, 2013."

        Computes scattering transform features, which have useful mathematical properties for classification
    """

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

class FourierBinning(FeatureExtractor):
    """
        Compute discrete Fourier transform, compute magnitude of returned Fourier coefficients, and compute sums
            of groups of coefficients.
    """

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
        pass


class VLAD(TorchModel):
    """
        Refer to:
            "A. Miech, I. Laptev, and J. Sivic.  Learnable pooling with context gating for video classification.CoRR,
             abs/1706.06905, 2017."

        Computes NetVLAD features, a variation of VLAD that is differentiable, and hence easy to learn
    """

    num_descriptors     = 4
    num_channels        = 16
    num_clusters        = 12

    # Passes training features for model initialization
    needs_train_feat = True

    class NetVLAD(nn.Module):
        """NetVLAD layer implementation"""

        def __init__(self, num_clusters, dim, feat_size, initial_clusters=None):
            """
            Args:
                num_clusters : int
                    The number of clusters
                dim : int
                    Dimension of descriptors
            """
            super().__init__()
            self.num_clusters   = num_clusters
            self.dim            = dim
            self.feat_size      = feat_size

            if initial_clusters is None:
                self.centroids = nn.Parameter(torch.rand(1, feat_size, num_clusters)) # 1 x L x K
            else:
                self.centroids = nn.Parameter(initial_clusters)

            self.activations_net = torch.nn.Sequential(
                                                        torch.nn.BatchNorm1d(self.feat_size),
                                                        torch.nn.Linear(self.feat_size, self.num_clusters),  # consider removing biases
                                                        torch.nn.BatchNorm1d(self.num_clusters),             # is this useful?
                                                        torch.nn.Softmax(dim=1)
                                                  )

        def forward(self, x):
            N, C, L = x.shape[:3]
            K       = self.num_clusters
            x_prime = torch.reshape(x, (N*C, L))

            # Similarity to each cluster: a_k(x_i) -> N x C x K
            #
            activations = self.activations_net.forward(x_prime)
            activations = torch.reshape(activations, (-1, C, K))

            # Strength per cluster:
            act_sum     = torch.unsqueeze(torch.sum(activations, dim=-2), dim=1)        # N x 1 x K
            a_clus      = act_sum * self.centroids                                      # N x L x K

            activations = activations.transpose(1, 2)                                   # N x K x C
            vlad        = torch.matmul(activations, x)                                  # N x K x L
            vlad        = vlad.transpose(1, 2)
            vlad        = vlad - a_clus                                                 # N x L x K

            # Normalize per cluster, and then globally
            vlad        = F.normalize(vlad, p=2, dim=1)                                 # N x L x K
            vlad        = torch.reshape(vlad, [-1, K * L])
            vlad        = F.normalize(vlad, p=2, dim=1)                                 # N x (L * K)

            return vlad

    def define_model(self, dim_in, all_train_features):
        """
            Computes K-means cluster as initial starting points for centroids

        :param dim_in: Expected dimension of each training feature sample
        :param all_train_features: All training features
        """

        feat_shape  = all_train_features.shape
        train_feat  = np.reshape(all_train_features, (feat_shape[0] * feat_shape[1], feat_shape[2]))

        initial_clusters    = sci_c.kmeans(train_feat, k_or_guess=self.num_clusters, iter=800)[0]
        initial_clusters    = np.expand_dims(np.transpose(initial_clusters), axis=0)
        initial_clusters    = torch.from_numpy(initial_clusters)

        model = torch.nn.Sequential(
            self.NetVLAD(self.num_clusters, self.num_descriptors, self.num_channels, initial_clusters),
            torch.nn.Linear(self.num_clusters * self.num_channels, NUM_GESTURES),
            torch.nn.Softmax(dim=1)
        )
        return model

    def forward_pass(self, sample):
        targets     = torch.LongTensor(sample[1]).to(self.device)
        predictions = self.model(sample[0].to(self.device))
        return torch.nn.functional.cross_entropy(predictions, targets), predictions


class KMeansRMS(FeatureExtractor):
    """
        A variation of NetVLAD, where clusters are computed, and the distance to the clusters are used as a
            similarity measure (instead of some complex function).
    """

    num_channels    = 16
    num_clusters    = 200
    window_size     = 200
    num_descriptors = 5

    requires_global_setup   = True

    def extract_feature_point(self, raw_samples):

        # Define data windows
        if raw_samples.shape[0] // self.window_size >= self.num_descriptors:
            start_indices = np.linspace(0, raw_samples.shape[0], num=self.num_descriptors, endpoint=False, dtype=int)
        else:
            num_start_indices = raw_samples.shape[0] // self.window_size
            start_indices = [x * self.window_size for x in range(num_start_indices)]
        emg_data = [raw_samples[y: y + self.window_size] for y in start_indices]

        # Create RMS descriptors
        descriptors = [np.sqrt(np.mean(np.square(data), axis=0)) for data in emg_data]
        descriptors = np.array(descriptors)
        descriptors = np.expand_dims(descriptors, axis=0)
        clust_dist  = descriptors - self.clusters
        dist_norm   = np.linalg.norm(clust_dist, axis=2)
        max_norm    = np.max(dist_norm, axis=0)
        dist_norm  /= max_norm
        dist_norm   = np.sum(dist_norm, axis=1)
        dist_norm  /= np.linalg.norm(dist_norm)
        return dist_norm

    def global_setup(self, all_raw_samples):

        all_descriptors = []

        for raw_samples in all_raw_samples:
            # Define data windows
            if raw_samples.shape[0] // self.window_size >= self.num_descriptors:
                start_indices = np.linspace(0, raw_samples.shape[0], num=self.num_descriptors, endpoint=False, dtype=int)
            else:
                num_start_indices = raw_samples.shape[0] // self.window_size
                start_indices = [x * self.window_size for x in range(num_start_indices)]
            emg_data = [raw_samples[y: y + self.window_size] for y in start_indices]

            # Create RMS descriptors
            all_descriptors.extend([np.sqrt(np.mean(np.square(data), axis=0)) for data in emg_data])

        all_descriptors = np.array(all_descriptors)
        self.clusters   = sci_c.kmeans(all_descriptors, k_or_guess=self.num_clusters, iter=800)[0]
        self.clusters   = np.expand_dims(self.clusters, axis=1)





########################################################################################################################
########################################################################################################################
########################################################################################################################
#
# (Advanced) Baseline-related Feature Extractors
#
########################################################################################################################
########################################################################################################################
########################################################################################################################

class MultipleRMS(FeatureExtractor):

    num_descriptors     = 4
    window_size         = 150
    num_channels        = 16

    def extract_feature_point(self, raw_samples):
        start_indices   = np.linspace(0, raw_samples.shape[0], num=self.num_descriptors, endpoint=False, dtype=int)
        emg_data        = [raw_samples[y:y+self.window_size] for y in start_indices]
        descriptors     = [np.sqrt(np.mean(np.square(data), axis=0)) for data in emg_data]
        descriptors     = np.array(descriptors)

        return descriptors.flatten()

    # Window based RMS
    #
    # def extract_feature_point(self, raw_samples):
    #     split_data      = np.split(raw_samples, indices_or_sections = 4, axis=0)
    #     mag_feat_list   = np.array([])
    #
    #     for data in split_data:
    #         mag_feat_list  = np.append(mag_feat_list, np.sqrt(np.mean(np.square(data), axis=0)))
    #
    #     return mag_feat_list

    def global_setup(self, all_raw_samples):
        pass



class PaddedMultiRMS(FeatureExtractor):

    num_descriptors     = 5
    window_size         = 200
    num_channels        = 16
    pca_dim             = 16

    requires_global_setup   = True

    ####################################################################################################################
    ####################################################################################################################
    #
    # Non-PCA, non-overlapping window version
    #
    ####################################################################################################################
    ####################################################################################################################
    # def extract_feature_point(self, raw_samples):
    #     # Define data windows
    #     if raw_samples.shape[0] // self.window_size >= self.num_descriptors:
    #         start_indices = np.linspace(0, raw_samples.shape[0], num=self.num_descriptors, endpoint=False, dtype=int)
    #     else:
    #         num_start_indices   = raw_samples.shape[0] // self.window_size
    #         start_indices       = [x * self.window_size for x in range(num_start_indices)]
    #     emg_data            = [raw_samples[y: y+self.window_size] for y in start_indices]
    #
    #     # Create RMS descriptors
    #     descriptors         = [np.sqrt(np.mean(np.square(data), axis=0)) for data in emg_data]
    #
    #     # Pad with zros
    #     num_miss            = self.num_descriptors - len(descriptors)
    #     descriptors        += [np.zeros(self.num_channels) for i in range(num_miss)]
    #     descriptors         = (np.array(descriptors)).flatten()
    #
    #     return descriptors
    #
    # def global_setup(self, all_raw_samples):
    #     pass
    ####################################################################################################################
    ####################################################################################################################


    ####################################################################################################################
    ####################################################################################################################
    #
    # PCA, overlapping window version
    #
    ####################################################################################################################
    ####################################################################################################################
    def extract_feature_point(self, raw_samples):
        # Define data windows
        start_indices = np.linspace(0, raw_samples.shape[0] - self.window_size, num=self.num_descriptors,
                                    endpoint=False,
                                    dtype=int)
        emg_data            = [raw_samples[y: y+self.window_size] for y in start_indices]

        # Create RMS descriptors
        descriptors = np.array([np.mean(np.abs(data), axis=0) for data in emg_data])
        norms       = np.linalg.norm(descriptors, axis=1)
        descriptors /= np.expand_dims(norms, axis=1)
        descriptors = self.pca.transform(descriptors)

        # Pad with zros
        num_miss        = self.num_descriptors - descriptors.shape[0]
        descriptors     = np.concatenate((descriptors, np.zeros((num_miss, self.pca_dim))), axis=0)
        descriptors     = descriptors.flatten()
        return descriptors

    def global_setup(self, all_raw_samples):

        all_descriptors = []

        for raw_samples in all_raw_samples:
            # Define data windows
            start_indices = np.linspace(0, raw_samples.shape[0] - self.window_size -1 , num=self.num_descriptors, endpoint=False,
                                        dtype=int)
            emg_data = [raw_samples[y: y + self.window_size] for y in start_indices]

            # Create RMS descriptors
            temp = np.array([np.mean(np.abs(data), axis=0) for data in emg_data])
            norm = np.linalg.norm(temp, axis=1)
            temp = temp / np.expand_dims(norm, axis=1)

            for i in range(temp.shape[0]):
                all_descriptors.append(temp[i])

        all_descriptors = np.array(all_descriptors)

        self.pca = PCA(n_components=self.pca_dim)
        self.pca.fit(all_descriptors)


    ####################################################################################################################
    ####################################################################################################################
    #
    # PCA, non-overlapping window version (ORIGINAL)
    #
    ####################################################################################################################
    ####################################################################################################################
    # def extract_feature_point(self, raw_samples):
    #     # Define data windows
    #     if raw_samples.shape[0] // self.window_size >= self.num_descriptors:
    #         start_indices = np.linspace(0, raw_samples.shape[0], num=self.num_descriptors, endpoint=False, dtype=int)
    #     else:
    #         num_start_indices   = raw_samples.shape[0] // self.window_size
    #         start_indices       = [x * self.window_size for x in range(num_start_indices)]
    #     emg_data            = [raw_samples[y: y+self.window_size] for y in start_indices]
    #
    #     # Create RMS descriptors
    #     descriptors = np.array([np.mean(np.abs(data), axis=0) for data in emg_data])
    #     norms       = np.linalg.norm(descriptors, axis=1)
    #     descriptors /= np.expand_dims(norms, axis=1)
    #     descriptors = self.pca.transform(descriptors)
    #
    #     # Pad with zros
    #     num_miss        = self.num_descriptors - descriptors.shape[0]
    #     descriptors     = np.concatenate((descriptors, np.zeros((num_miss, self.pca_dim))), axis=0)
    #     descriptors     = descriptors.flatten()
    #     return descriptors
    #
    # def global_setup(self, all_raw_samples):
    #
    #     all_descriptors     = []
    #
    #     for raw_samples in all_raw_samples:
    #         # Define data windows
    #         if raw_samples.shape[0] // self.window_size >= self.num_descriptors:
    #             start_indices = np.linspace(0, raw_samples.shape[0], num=self.num_descriptors, endpoint=False,
    #                                         dtype=int)
    #         else:
    #             num_start_indices = raw_samples.shape[0] // self.window_size
    #             start_indices = [x * self.window_size for x in range(num_start_indices)]
    #
    #         emg_data    = [raw_samples[y: y + self.window_size] for y in start_indices]
    #
    #         temp = np.array([np.mean(np.abs(data), axis=0) for data in emg_data])
    #         norm = np.linalg.norm(temp, axis=1)
    #         temp = temp / np.expand_dims(norm, axis=1)
    #         for i in range(temp.shape[0]):
    #             all_descriptors.append(temp[i])
    #
    #     all_descriptors = np.array(all_descriptors)
    #     self.pca = PCA(n_components=self.pca_dim)
    #     self.pca.fit(all_descriptors)
    ####################################################################################################################
    ####################################################################################################################


    ####################################################################################################################
    ####################################################################################################################
    #
    # Non-overlapping windows, PCA performed on zero padded vectors
    #
    ####################################################################################################################
    ####################################################################################################################
    # def extract_feature_point(self, raw_samples):
    #     # Define data windows
    #     if raw_samples.shape[0] // self.window_size >= self.num_descriptors:
    #         start_indices = np.linspace(0, raw_samples.shape[0], num=self.num_descriptors, endpoint=False, dtype=int)
    #     else:
    #         num_start_indices = raw_samples.shape[0] // self.window_size
    #         start_indices = [x * self.window_size for x in range(num_start_indices)]
    #     emg_data = [raw_samples[y: y + self.window_size] for y in start_indices]
    #
    #     # Create RMS descriptors
    #     descriptors = np.array([np.mean(np.abs(data), axis=0) for data in emg_data])
    #     norms = np.linalg.norm(descriptors, axis=1)
    #     descriptors /= np.expand_dims(norms, axis=1)
    #
    #     # Pad with zros
    #     num_miss = self.num_descriptors - descriptors.shape[0]
    #     descriptors = np.concatenate((descriptors, np.zeros((num_miss, self.num_channels))), axis=0)
    #     descriptors = descriptors.flatten()
    #     descriptors = np.squeeze(self.pca.transform(descriptors.reshape(1, -1)))
    #
    #     return descriptors
    #
    # def global_setup(self, all_raw_samples):
    #
    #     all_descriptors = []
    #
    #     for raw_samples in all_raw_samples:
    #         # Define data windows
    #         if raw_samples.shape[0] // self.window_size >= self.num_descriptors:
    #             start_indices = np.linspace(0, raw_samples.shape[0], num=self.num_descriptors, endpoint=False,
    #                                         dtype=int)
    #         else:
    #             num_start_indices = raw_samples.shape[0] // self.window_size
    #             start_indices = [x * self.window_size for x in range(num_start_indices)]
    #         emg_data = [raw_samples[y: y + self.window_size] for y in start_indices]
    #
    #         # Create RMS descriptors
    #         temp = np.array([np.mean(np.abs(data), axis=0) for data in emg_data])
    #         norm = np.linalg.norm(temp, axis=1)
    #         temp = temp / np.expand_dims(norm, axis=1)
    #
    #         num_miss = self.num_descriptors - temp.shape[0]
    #         descriptors = np.concatenate((temp, np.zeros((num_miss, self.num_channels))), axis=0)
    #         all_descriptors.append(descriptors.flatten())
    #
    #     all_descriptors = np.array(all_descriptors)
    #
    #     self.pca = PCA(n_components=self.pca_dim)
    #     self.pca.fit(all_descriptors)
    ####################################################################################################################
    ####################################################################################################################



class IMUPaddedMultiRMS(FeatureExtractor):

    num_descriptors     = 5
    window_size         = 200
    num_channels        = 16
    pca_dim             = 16
    acc_dim             = 6
    gyro_dim            = 6
    mag_dim             = 8
    #imu_dim            = 20 # acc: 6, gyro: 6, mag: 8
    imu_dim             = 12 # acc: 6, gyro: 6, mag: 8

    requires_global_setup  = True


    ####################################################################################################################
    ####################################################################################################################
    #
    # Original IMU enhanced PaddedMultiRMS
    #
    ####################################################################################################################
    ####################################################################################################################
    # def extract_feature_point(self, raw_samples):
    #     acc_samples     = raw_samples[1]
    #     gyro_samples    = raw_samples[2]
    #     mag_samples     = raw_samples[3]
    #     raw_samples     = raw_samples[0]
    #
    #     # Define data windows
    #     if raw_samples.shape[0] // self.window_size >= self.num_descriptors:
    #         start_indices = np.linspace(0, raw_samples.shape[0], num=self.num_descriptors, endpoint=False, dtype=int)
    #     else:
    #         num_start_indices = raw_samples.shape[0] // self.window_size
    #         start_indices = [x * self.window_size for x in range(num_start_indices)]
    #
    #     # Simple IMU mean features
    #     acc_data    = [acc_samples[y: y + self.window_size] for y in start_indices]
    #     acc_feat    = np.array([((np.mean(x) - self.acc_mean) / self.acc_std)  for x in acc_data])
    #
    #     gyro_data   = [gyro_samples[y: y + self.window_size] for y in start_indices]
    #     gyro_feat   = np.array([((np.mean(x) - self.gyro_mean) / self.gyro_std)  for x in gyro_data])
    #
    #     mag_data    = [mag_samples[y: y + self.window_size] for y in start_indices]
    #     mag_feat    = np.array([((np.mean(x) - self.mag_mean) / self.mag_std) for x in mag_data])
    #
    #     # Create RMS descriptors
    #     emg_data    = [raw_samples[y: y + self.window_size] for y in start_indices]
    #
    #     descriptors = np.array([np.mean(np.abs(data), axis=0) for data in emg_data])
    #     norms       = np.linalg.norm(descriptors, axis=1)
    #     descriptors /= np.expand_dims(norms, axis=1)
    #     descriptors = self.pca.transform(descriptors)
    #
    #     unpadded_feat = np.concatenate((descriptors, acc_feat, gyro_feat, mag_feat), axis=1)
    #
    #     # Pad with zros
    #     num_miss = self.num_descriptors - descriptors.shape[0]
    #     descriptors = np.concatenate((unpadded_feat, np.zeros((num_miss, self.pca_dim + self.imu_dim))), axis=0)
    #     descriptors = descriptors.flatten()
    #     return descriptors
    #
    # def global_setup(self, all_raw_samples):
    #     all_descriptors     = []
    #     all_acc_means       = []
    #     all_gyro_means      = []
    #     all_mag_means       = []
    #
    #     for combined_samples in all_raw_samples:
    #         acc_samples     = combined_samples[1]
    #         gyro_samples    = combined_samples[2]
    #         mag_samples     = combined_samples[3]
    #         raw_samples     = combined_samples[0]
    #
    #         # Define data windows
    #         if raw_samples.shape[0] // self.window_size >= self.num_descriptors:
    #             start_indices = np.linspace(0, raw_samples.shape[0], num=self.num_descriptors, endpoint=False,
    #                                         dtype=int)
    #         else:
    #             num_start_indices = raw_samples.shape[0] // self.window_size
    #             start_indices = [x * self.window_size for x in range(num_start_indices)]
    #
    #         emg_data    = [raw_samples[y: y + self.window_size] for y in start_indices]
    #         acc_data    = [acc_samples[y: y + self.window_size] for y in start_indices]
    #         gyro_data   = [gyro_samples[y: y + self.window_size] for y in start_indices]
    #         mag_data    = [mag_samples[y: y + self.window_size] for y in start_indices]
    #
    #         # (EMG) Create "RMS" descriptors
    #         temp = np.array([np.mean(np.abs(data), axis=0) for data in emg_data])
    #         norm = np.linalg.norm(temp, axis=1)
    #         temp = temp / np.expand_dims(norm, axis=1)
    #         for i in range(temp.shape[0]):
    #             all_descriptors.append(temp[i])
    #
    #         # (ACC)
    #         temp = np.array([np.mean(data, axis=0) for data in acc_data])
    #         for i in range(temp.shape[0]):
    #             all_acc_means.append(temp[i])
    #
    #         # (GYRO)
    #         temp = np.array([np.mean(data, axis=0) for data in gyro_data])
    #         for i in range(temp.shape[0]):
    #             all_gyro_means.append(temp[i])
    #
    #         # (MAG)
    #         temp = np.array([np.mean(data, axis=0) for data in mag_data])
    #         for i in range(temp.shape[0]):
    #             all_mag_means.append(temp[i])
    #
    #     # EMG
    #     all_descriptors = np.array(all_descriptors)
    #     self.pca = PCA(n_components=self.pca_dim)
    #     self.pca.fit(all_descriptors)
    #
    #     # ACC
    #     all_acc_means   = np.array(all_acc_means)
    #     self.acc_mean   = np.mean(all_acc_means, axis=0)
    #     self.acc_std    = np.std(all_acc_means, axis=0)
    #
    #     # GYRO
    #     all_gyro_means  = np.array(all_gyro_means)
    #     self.gyro_mean  = np.mean(all_gyro_means, axis=0)
    #     self.gyro_std   = np.std(all_gyro_means, axis=0)
    #
    #     # (MAG)
    #     all_mag_means = np.array(all_mag_means)
    #     self.mag_mean = np.mean(all_mag_means, axis=0)
    #     self.mag_std  = np.std(all_mag_means, axis=0)
    ####################################################################################################################
    ####################################################################################################################


    ####################################################################################################################
    ####################################################################################################################
    #
    # Best IMU enhanced PaddedMultiRMS (no use of magnetometer data, which leads to poor generalization)
    #
    ####################################################################################################################
    ####################################################################################################################
    def extract_feature_point(self, raw_samples):
        acc_samples     = raw_samples[1]
        gyro_samples    = raw_samples[2]
        #mag_samples     = raw_samples[3]
        raw_samples     = raw_samples[0]

        # Define data windows
        if raw_samples.shape[0] // self.window_size >= self.num_descriptors:
            start_indices = np.linspace(0, raw_samples.shape[0], num=self.num_descriptors, endpoint=False, dtype=int)
        else:
            num_start_indices = raw_samples.shape[0] // self.window_size
            start_indices = [x * self.window_size for x in range(num_start_indices)]

        # Simple IMU mean features
        acc_data    = [acc_samples[y: y + self.window_size] for y in start_indices]
        acc_feat    = np.array([((np.mean(x) - self.acc_mean) / self.acc_std)  for x in acc_data])

        gyro_data   = [gyro_samples[y: y + self.window_size] for y in start_indices]
        gyro_feat   = np.array([((np.mean(x) - self.gyro_mean) / self.gyro_std)  for x in gyro_data])

        # mag_data    = [mag_samples[y: y + self.window_size] for y in start_indices]
        # mag_feat    = np.array([((np.mean(x) - self.mag_mean) / self.mag_std) for x in mag_data])

        # Create RMS descriptors
        emg_data    = [raw_samples[y: y + self.window_size] for y in start_indices]

        descriptors = np.array([np.mean(np.abs(data), axis=0) for data in emg_data])
        norms       = np.linalg.norm(descriptors, axis=1)
        descriptors /= np.expand_dims(norms, axis=1)
        descriptors = self.pca.transform(descriptors)

        unpadded_feat = np.concatenate((descriptors, acc_feat, gyro_feat), axis=1)
        #unpadded_feat = descriptors

        # Pad with zros
        num_miss = self.num_descriptors - descriptors.shape[0]
        descriptors = np.concatenate((unpadded_feat, np.zeros((num_miss, self.pca_dim + self.imu_dim))), axis=0)
        descriptors = descriptors.flatten()
        return descriptors

    def global_setup(self, all_raw_samples):
        all_descriptors     = []
        all_acc_means       = []
        all_gyro_means      = []
        #all_mag_means       = []

        for combined_samples in all_raw_samples:
            acc_samples     = combined_samples[1]
            gyro_samples    = combined_samples[2]
            #mag_samples     = combined_samples[3]
            raw_samples     = combined_samples[0]

            # Define data windows
            if raw_samples.shape[0] // self.window_size >= self.num_descriptors:
                start_indices = np.linspace(0, raw_samples.shape[0], num=self.num_descriptors, endpoint=False,
                                            dtype=int)
            else:
                num_start_indices = raw_samples.shape[0] // self.window_size
                start_indices = [x * self.window_size for x in range(num_start_indices)]

            emg_data    = [raw_samples[y: y + self.window_size] for y in start_indices]
            acc_data    = [acc_samples[y: y + self.window_size] for y in start_indices]
            gyro_data   = [gyro_samples[y: y + self.window_size] for y in start_indices]
            #mag_data    = [mag_samples[y: y + self.window_size] for y in start_indices]

            # (EMG) Create "RMS" descriptors
            temp = np.array([np.mean(np.abs(data), axis=0) for data in emg_data])
            norm = np.linalg.norm(temp, axis=1)
            temp = temp / np.expand_dims(norm, axis=1)
            for i in range(temp.shape[0]):
                all_descriptors.append(temp[i])

            # (ACC)
            temp = np.array([np.mean(data, axis=0) for data in acc_data])
            for i in range(temp.shape[0]):
                all_acc_means.append(temp[i])

            # (GYRO)
            temp = np.array([np.mean(data, axis=0) for data in gyro_data])
            for i in range(temp.shape[0]):
                all_gyro_means.append(temp[i])

            # (MAG)
            # temp = np.array([np.mean(data, axis=0) for data in mag_data])
            # for i in range(temp.shape[0]):
            #     all_mag_means.append(temp[i])

        # EMG
        all_descriptors = np.array(all_descriptors)
        self.pca = PCA(n_components=self.pca_dim)
        self.pca.fit(all_descriptors)

        # ACC
        all_acc_means   = np.array(all_acc_means)
        self.acc_mean   = np.mean(all_acc_means, axis=0)
        self.acc_std    = np.std(all_acc_means, axis=0)

        # GYRO
        all_gyro_means  = np.array(all_gyro_means)
        self.gyro_mean  = np.mean(all_gyro_means, axis=0)
        self.gyro_std   = np.std(all_gyro_means, axis=0)

        # (MAG)
        # all_mag_means = np.array(all_mag_means)
        # self.mag_mean = np.mean(all_mag_means, axis=0)
        # self.mag_std  = np.std(all_mag_means, axis=0)
    ####################################################################################################################
    ####################################################################################################################


    ####################################################################################################################
    ####################################################################################################################
    #
    # IMU enhanced PaddedMultiRMS, with PCA transformation of IMU data
    #
    ####################################################################################################################
    ####################################################################################################################
    # def extract_feature_point(self, raw_samples):
    #     acc_samples     = raw_samples[1]
    #     gyro_samples    = raw_samples[2]
    #     mag_samples     = raw_samples[3]
    #     raw_samples     = raw_samples[0]
    #
    #     # Define data windows
    #     if raw_samples.shape[0] // self.window_size >= self.num_descriptors:
    #         start_indices = np.linspace(0, raw_samples.shape[0], num=self.num_descriptors, endpoint=False, dtype=int)
    #     else:
    #         num_start_indices = raw_samples.shape[0] // self.window_size
    #         start_indices = [x * self.window_size for x in range(num_start_indices)]
    #
    #     # Simple IMU mean features
    #     acc_data    = [acc_samples[y: y + self.window_size] for y in start_indices]
    #     acc_feat    = np.array([((np.mean(x) - self.acc_mean) / self.acc_std)  for x in acc_data])
    #     acc_feat    = self.acc_pca.transform(acc_feat)
    #
    #     gyro_data   = [gyro_samples[y: y + self.window_size] for y in start_indices]
    #     gyro_feat   = np.array([((np.mean(x) - self.gyro_mean) / self.gyro_std)  for x in gyro_data])
    #     gyro_feat   = self.gyro_pca.transform(gyro_feat)
    #
    #     mag_data    = [mag_samples[y: y + self.window_size] for y in start_indices]
    #     mag_feat    = np.array([((np.mean(x) - self.mag_mean) / self.mag_std) for x in mag_data])
    #     mag_feat    = self.mag_pca.transform(mag_feat)
    #
    #     # Create RMS descriptors
    #     emg_data    = [raw_samples[y: y + self.window_size] for y in start_indices]
    #
    #     descriptors = np.array([np.mean(np.abs(data), axis=0) for data in emg_data])
    #     norms       = np.linalg.norm(descriptors, axis=1)
    #     descriptors /= np.expand_dims(norms, axis=1)
    #     descriptors = self.pca.transform(descriptors)
    #
    #     unpadded_feat = np.concatenate((descriptors, acc_feat, gyro_feat, mag_feat), axis=1)
    #
    #     # Pad with zros
    #     num_miss = self.num_descriptors - descriptors.shape[0]
    #     descriptors = np.concatenate((unpadded_feat, np.zeros((num_miss, self.pca_dim + self.imu_dim))), axis=0)
    #     descriptors = descriptors.flatten()
    #     return descriptors
    #
    # def global_setup(self, all_raw_samples):
    #     all_descriptors     = []
    #     all_acc_means       = []
    #     all_gyro_means      = []
    #     all_mag_means       = []
    #
    #     for combined_samples in all_raw_samples:
    #         acc_samples     = combined_samples[1]
    #         gyro_samples    = combined_samples[2]
    #         mag_samples     = combined_samples[3]
    #         raw_samples     = combined_samples[0]
    #
    #         # Define data windows
    #         if raw_samples.shape[0] // self.window_size >= self.num_descriptors:
    #             start_indices = np.linspace(0, raw_samples.shape[0], num=self.num_descriptors, endpoint=False,
    #                                         dtype=int)
    #         else:
    #             num_start_indices = raw_samples.shape[0] // self.window_size
    #             start_indices = [x * self.window_size for x in range(num_start_indices)]
    #
    #         emg_data    = [raw_samples[y: y + self.window_size] for y in start_indices]
    #         acc_data    = [acc_samples[y: y + self.window_size] for y in start_indices]
    #         gyro_data   = [gyro_samples[y: y + self.window_size] for y in start_indices]
    #         mag_data    = [mag_samples[y: y + self.window_size] for y in start_indices]
    #
    #         # (EMG) Create "RMS" descriptors
    #         temp = np.array([np.mean(np.abs(data), axis=0) for data in emg_data])
    #         norm = np.linalg.norm(temp, axis=1)
    #         temp = temp / np.expand_dims(norm, axis=1)
    #         for i in range(temp.shape[0]):
    #             all_descriptors.append(temp[i])
    #
    #         # (ACC)
    #         temp = np.array([np.mean(data, axis=0) for data in acc_data])
    #         for i in range(temp.shape[0]):
    #             all_acc_means.append(temp[i])
    #
    #         # (GYRO)
    #         temp = np.array([np.mean(data, axis=0) for data in gyro_data])
    #         for i in range(temp.shape[0]):
    #             all_gyro_means.append(temp[i])
    #
    #         # (MAG)
    #         temp = np.array([np.mean(data, axis=0) for data in mag_data])
    #         for i in range(temp.shape[0]):
    #             all_mag_means.append(temp[i])
    #
    #     # EMG
    #     all_descriptors = np.array(all_descriptors)
    #     self.pca = PCA(n_components=self.pca_dim)
    #     self.pca.fit(all_descriptors)
    #
    #     # ACC
    #     all_acc_means   = np.array(all_acc_means)
    #     self.acc_mean   = np.mean(all_acc_means, axis=0)
    #     self.acc_std    = np.std(all_acc_means, axis=0)
    #     self.acc_pca    = PCA(n_components=self.acc_dim)
    #     self.acc_pca.fit((all_acc_means - self.acc_mean) / self.acc_std)
    #
    #     # GYRO
    #     all_gyro_means  = np.array(all_gyro_means)
    #     self.gyro_mean  = np.mean(all_gyro_means, axis=0)
    #     self.gyro_std   = np.std(all_gyro_means, axis=0)
    #     self.gyro_pca = PCA(n_components=self.gyro_dim)
    #     self.gyro_pca.fit((all_gyro_means - self.gyro_mean) / self.gyro_std)
    #
    #     # (MAG)
    #     all_mag_means = np.array(all_mag_means)
    #     self.mag_mean = np.mean(all_mag_means, axis=0)
    #     self.mag_std  = np.std(all_mag_means, axis=0)
    #     self.mag_pca  = PCA(n_components=self.mag_dim)
    #     self.mag_pca.fit((all_mag_means - self.mag_mean) / self.mag_std)
    ####################################################################################################################
    ####################################################################################################################



    ####################################################################################################################
    ####################################################################################################################
    #
    # Original IMU enhanced PaddedMultiRMS, with overlapping windows
    #
    ####################################################################################################################
    ####################################################################################################################
    # def extract_feature_point(self, raw_samples):
    #     acc_samples     = raw_samples[1]
    #     gyro_samples    = raw_samples[2]
    #     mag_samples     = raw_samples[3]
    #     raw_samples     = raw_samples[0]
    #
    #     # Define data windows
    #     start_indices = np.linspace(0, raw_samples.shape[0] - self.window_size, num=self.num_descriptors,
    #                                     endpoint=False, dtype=int)
    #
    #     # Simple IMU mean features
    #     acc_data    = [acc_samples[y: y + self.window_size] for y in start_indices]
    #     acc_feat    = np.array([((np.mean(x) - self.acc_mean) / self.acc_std)  for x in acc_data])
    #
    #     gyro_data   = [gyro_samples[y: y + self.window_size] for y in start_indices]
    #     gyro_feat   = np.array([((np.mean(x) - self.gyro_mean) / self.gyro_std)  for x in gyro_data])
    #
    #     mag_data    = [mag_samples[y: y + self.window_size] for y in start_indices]
    #     mag_feat    = np.array([((np.mean(x) - self.mag_mean) / self.mag_std) for x in mag_data])
    #
    #     # Create RMS descriptors
    #     emg_data    = [raw_samples[y: y + self.window_size] for y in start_indices]
    #
    #     descriptors = np.array([np.mean(np.abs(data), axis=0) for data in emg_data])
    #     norms       = np.linalg.norm(descriptors, axis=1)
    #     descriptors /= np.expand_dims(norms, axis=1)
    #     descriptors = self.pca.transform(descriptors)
    #
    #     unpadded_feat = np.concatenate((descriptors, acc_feat, gyro_feat, mag_feat), axis=1)
    #
    #     # Pad with zros
    #     num_miss = self.num_descriptors - descriptors.shape[0]
    #     descriptors = np.concatenate((unpadded_feat, np.zeros((num_miss, self.pca_dim + self.imu_dim))), axis=0)
    #     descriptors = descriptors.flatten()
    #     return descriptors
    #
    # def global_setup(self, all_raw_samples):
    #     all_descriptors     = []
    #     all_acc_means       = []
    #     all_gyro_means      = []
    #     all_mag_means       = []
    #
    #     for combined_samples in all_raw_samples:
    #         acc_samples     = combined_samples[1]
    #         gyro_samples    = combined_samples[2]
    #         mag_samples     = combined_samples[3]
    #         raw_samples     = combined_samples[0]
    #
    #         # Define data windows
    #         start_indices = np.linspace(0, raw_samples.shape[0] - self.window_size, num=self.num_descriptors,
    #                                     endpoint=False, dtype=int)
    #
    #         emg_data    = [raw_samples[y: y + self.window_size] for y in start_indices]
    #         acc_data    = [acc_samples[y: y + self.window_size] for y in start_indices]
    #         gyro_data   = [gyro_samples[y: y + self.window_size] for y in start_indices]
    #         mag_data    = [mag_samples[y: y + self.window_size] for y in start_indices]
    #
    #         # (EMG) Create "RMS" descriptors
    #         temp = np.array([np.mean(np.abs(data), axis=0) for data in emg_data])
    #         norm = np.linalg.norm(temp, axis=1)
    #         temp = temp / np.expand_dims(norm, axis=1)
    #         for i in range(temp.shape[0]):
    #             all_descriptors.append(temp[i])
    #
    #         # (ACC)
    #         temp = np.array([np.mean(data, axis=0) for data in acc_data])
    #         for i in range(temp.shape[0]):
    #             all_acc_means.append(temp[i])
    #
    #         # (GYRO)
    #         temp = np.array([np.mean(data, axis=0) for data in gyro_data])
    #         for i in range(temp.shape[0]):
    #             all_gyro_means.append(temp[i])
    #
    #         # (MAG)
    #         temp = np.array([np.mean(data, axis=0) for data in mag_data])
    #         for i in range(temp.shape[0]):
    #             all_mag_means.append(temp[i])
    #
    #     # EMG
    #     all_descriptors = np.array(all_descriptors)
    #     self.pca = PCA(n_components=self.pca_dim)
    #     self.pca.fit(all_descriptors)
    #
    #     # ACC
    #     all_acc_means   = np.array(all_acc_means)
    #     self.acc_mean   = np.mean(all_acc_means, axis=0)
    #     self.acc_std    = np.std(all_acc_means, axis=0)
    #
    #     # GYRO
    #     all_gyro_means  = np.array(all_gyro_means)
    #     self.gyro_mean  = np.mean(all_gyro_means, axis=0)
    #     self.gyro_std   = np.std(all_gyro_means, axis=0)
    #
    #     # (MAG)
    #     all_mag_means = np.array(all_mag_means)
    #     self.mag_mean = np.mean(all_mag_means, axis=0)
    #     self.mag_std  = np.std(all_mag_means, axis=0)
    ####################################################################################################################
    ####################################################################################################################