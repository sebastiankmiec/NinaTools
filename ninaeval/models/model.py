from abc import ABC, abstractmethod
import torch
import torchnet
from tqdm import tqdm
import os
import glob
import matplotlib.pyplot as plt

try:
    import cPickle as pickle
except:
    import pickle

# For all torch models
torch.manual_seed(13)

NUM_GESTURES = 53 # 52 + 1 (rest)


class ClassifierModel(ABC):
    #
    # All classifier models perform this basic functionality
    #

    def __init__(self, models_path, feat_extractor):
        """
        :param models_path: Path to store model
        :param feat_extractor: FeatureExtractor object, turning raw EMG/IMU samples into a feature vector
        """
        self.models_path    = models_path
        self.feat_extractor = feat_extractor
        self.num_samples    = None

    @abstractmethod
    def train_model(self, train_features, train_labels, valid_features = None, valid_labels = None):
        """
        :param train_features: A numpy array, a member of a Dataset object, containing features for training
        :param train_labels:  A numpy array, a member of a Dataset object, containing label for training
        :param valid_features: A numpy array, a member of a Dataset object, containing features for validation
        :param valid_labels:  A numpy array, a member of a Dataset object, containing label for validation
        """
        pass

    def perform_inference(self, test_features, test_labels):
        """
            Given test features and labels, compute predictions and classifier accuracy,

        :param test_features:Features from the test split.
        :param test_labels: Labels from the test split.
        :return: * (test_labels != None) Classifier accuracy from 0 ~ 1.0.
                 * (test_labels == None) Predictions
        """
        predictions = self.perform_inference(test_features)

        if test_labels is None:
            return predictions
        else:
            return self.classifier_accuracy(predictions, test_labels)

    @abstractmethod
    def perform_inference(self, test_features):
        """
            Helper function for above
        """
        pass

    @abstractmethod
    def save_figure(self, path):
        pass

    def save_model(self, path = None):
        """
            Serializes (this) object for future loading and use

        :param path: Path to save this object
        """
        if path is None:
            path = os.path.join(self.models_path,  self.__class__.__name__ + "_" +
                                    self.feat_extractor.__class__.__name__ )

        with open(path, 'wb') as f:
            pickle.dump(self, f, 2)

    def load_model(path):
        """
            De-serializes a pickled (serialized) ClassifierModel object

        :param: Path of pickled (serialized) object

        :return A ClassifierModel object
        """
        if not (os.path.exists(path)):
            return

        with open(path, 'rb') as f:
            object = pickle.load(f)
            return object

    def classifier_accuracy(self, predictions, test_labels):
        """
        :param predictions: Predictions made on a set of test_features
        :param test_labels: Ground truth labels associated with test features

        :return: Classifier accuracy from 0 ~ 1.0.
        """
        errors      = predictions == test_labels
        acc_rate    = len([x for x in errors if (x == True)]) / len(errors)
        return acc_rate

    def per_class_accuracy(self, test_features, test_labels):
        """
        :param test_features: Features used to generate predictions
        :param test_labels: Ground truth labels associated with test features

        :return: [dict] : Each key refers to a per class accuracy
        """

        pred_labels = self.perform_inference(test_features)

        test_labels_found = {}
        for x in test_labels:
            if x not in test_labels_found:
                test_labels_found[x] = []

        for i in range(test_labels.shape[0]):
            cur_label   = test_labels[i]
            cur_pred    = pred_labels[i]
            test_labels_found[cur_label].append(cur_label == cur_pred)

        for y in test_labels_found.keys():
            sum = 0
            for j in test_labels_found[y]:
                sum += j

            test_labels_found[y] = sum / len(test_labels_found[y])

        return test_labels_found


#
# All Torch classifiers implement this interface.
#
class TorchModel(ClassifierModel):

    # Defaults
    weight_decay    = 1e-5
    learning_rate   = 1e-3
    batch_size      = 32
    num_workers     = 1
    num_epoch       = 200
    chkpt_period    = 1
    valid_period    = 1

    # Configurable
    checkpoint_ext   = ".chkpt"
    needs_train_feat = False

    # Do not modify (states)
    start_epoch     = 0
    train_accs      = []
    test_accs       = []
    valid_features  = None
    valid_labels    = None

    def __init__(self, models_path, feat_extractor):
        super().__init__(models_path, feat_extractor)

        # Engine for training/testing model
        self.engine = torchnet.engine.Engine()
        self.engine.hooks['on_start_epoch'] = self.on_start_epoch
        self.engine.hooks['on_end_epoch']   = self.on_end_epoch
        self.engine.hooks['on_end']         = self.on_end
        self.engine.hooks['on_sample']      = self.on_sample
        self.engine.hooks['on_forward']     = self.on_forward

        self.meter_loss = torchnet.meter.AverageValueMeter()
        self.classerr   = torchnet.meter.ClassErrorMeter(accuracy=True)

        # Note: Use GPU by default
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model  = None

    def train_model(self, train_features, train_labels, valid_features = None, valid_labels = None):
        """
        :param train_features: A numpy array, a member of a Dataset object, containing features for training
        :param train_labels:  A numpy array, a member of a Dataset object, containing label for training
        :param valid_features: A numpy array, a member of a Dataset object, containing features for validation
        :param valid_labels:  A numpy array, a member of a Dataset object, containing label for validation
        """
        self.valid_features = valid_features
        self.valid_labels   = valid_labels

        dim_in = train_features[0].shape[0]

        if self.needs_train_feat:
            self.model  = self.define_model(dim_in, train_features)
        else:
            self.model  = self.define_model(dim_in)

        # Use model for training
        self.model.to(self.device)
        self.model.train()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        self.load_checkpoint()

        if self.start_epoch < self.num_epoch:

            # First convert to one-hot enccoding (using broadcasting)
            #
            # train_labels    = torch.from_numpy(train_labels).reshape(len(train_labels), 1)
            # train_labels    = (
            #                        train_labels == torch.arange(NUM_GESTURES).reshape(1, NUM_GESTURES)
            #                ).type(torch.LongTensor)

            torch_dataset = torchnet.dataset.TensorDataset([
                (torch.from_numpy(train_features)).float(),
                torch.from_numpy(train_labels)
            ])
            iterator = torch_dataset.parallel(
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True
            )
            self.engine.train(self.forward_pass, iterator, maxepoch=self.num_epoch - self.start_epoch, optimizer=self.optimizer)

        print("Training complete ({} epochs)...".format(self.num_epoch))


    def perform_inference(self, test_features):
        """
            Helper function for ClassifierModel's perform_inference(self, test_features, test_labels)
        """

        # Use inference model
        self.model.eval()

        predictions = (self.model((torch.from_numpy(test_features).float()).to(self.device))).cpu().detach()
        predictions = (torch.argmax(predictions, 1)).numpy()

        return predictions

    def perform_testing(self, test_features, test_labels):
        """
            Performs a forward pass on each batch in the validation split

        :param test_features: A numpy array, a member of a Dataset object, containing features for validation
        :param test_labels: A numpy array, a member of a Dataset object, containing labels for validation
        """
        # Use inference model
        self.model.eval()
        self.reset_meters()

        torch_dataset = torchnet.dataset.TensorDataset([
            (torch.from_numpy(test_features)).float(),
            torch.from_numpy(test_labels)
        ])
        iterator = torch_dataset.parallel(
            batch_size  = self.batch_size,
            num_workers = self.num_workers,
            shuffle     = False
        )
        self.engine.test(self.forward_pass, iterator)

    def on_sample(self, state):
        """
            For each batch of training/testing samples in the training/testing split, this function is called to
                determine if the sample is for training/testing (by other functions used by Torch engine)

        :param state: A state object used by the Torch engine
        """
        state['sample'].append(state['train'])

    def on_forward(self, state):
        """
            For each batch of training/testing samples in the training/testing split, this function is called to
                append classification accuracy and loss

        :param state: A state object used by the Torch engine
        """
        self.classerr.add(state['output'].data, torch.LongTensor(state['sample'][1]))
        self.meter_loss.add(state['loss'].item())

    def reset_meters(self):
        """
            Resets meters used for computing classification accuracy
        """
        self.classerr.reset()
        self.meter_loss.reset()

    def on_start_epoch(self, state):
        """
            On the start of a training/testing epoch, this function is called, to reset meters (used for class. acc.)

        :param state: A state object used by the Torch engine
        """
        self.reset_meters()

        # Buggy (to track the progress of an epoch)
        #state['iterator'] = tqdm(state['iterator'])

    def on_end_epoch(self, state):
        """
            On the end of a training/testing epoch, this function is called, to save a checkpoint and append
                training/testing accuracy

        :param state: A state object used by the Torch engine
        """
        print('Training Accuracy (epoch', int(state["epoch"]) + self.start_epoch, '): ', self.classerr.value())
        self.train_accs.append(self.classerr.value())

        if int(int(state["epoch"]) + self.start_epoch) % self.chkpt_period == 0:
            self.save_checkpoint(int(state["epoch"]) + self.start_epoch, state["loss"])

        if (
                (int(int(state["epoch"]) + self.start_epoch) % self.valid_period == 0) and
                ((self.valid_labels is not None) and (self.valid_features is not None)) and
                (state["epoch"] != self.num_epoch)
            ):
            self.reset_meters()
            self.perform_testing(self.valid_features, self.valid_labels)
            self.model.train()

    def on_end(self, state):
        """
            On the end of training/testing, this function is called, to append training/test accuracy

        :param state: A state object used by the Torch engine
        """
        print('Training' if state['train'] else 'Testing', 'accuracy')
        print(self.classerr.value())

        if not state["train"]:
            self.test_accs.append(self.classerr.value())
        self.reset_meters()

    def save_checkpoint(self, epoch, loss):
        """
            Saves a partially/completely trained Torch model

        :param epoch: Epoch number associated with this checkpoint
        :param loss: Current loss associated with this checkpoint
        :return:
        """
        if self.model is not None:
            print("Saving model checkpoint...")

            full_model_path = os.path.join(self.models_path,
                                                self.__class__.__name__ + "_" + self.feat_extractor.__class__.__name__)
            if not os.path.exists(full_model_path):
                os.makedirs(full_model_path)

            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss,
            }, os.path.join(full_model_path, str(epoch) + self.checkpoint_ext))

    def load_checkpoint(self):
        """
            Checks the model path provided initially for a partially/completely trained model
        """
        full_model_path = os.path.join(self.models_path,
                                            self.__class__.__name__ + "_" + self.feat_extractor.__class__.__name__)

        if os.path.exists(full_model_path):
            if len(os.listdir(full_model_path)) > 0:

                print("Loading model checkpoint...")
                max         = -1
                checkpoints = glob.glob(os.path.join(full_model_path, "*" + self.checkpoint_ext))
                latest      = None

                # Look for latest checkpoint
                for checkpoint in checkpoints:
                    chkpt_filename  = os.path.split(checkpoint)[-1]
                    if not self.checkpoint_ext in chkpt_filename:
                        continue

                    chkpt_filename  = chkpt_filename.replace(self.checkpoint_ext, "")
                    chkpt_num       = int(chkpt_filename)

                    if chkpt_num > max:
                        max     = chkpt_num
                        latest  = checkpoint

                if latest is None:
                    return

                checkpoint = torch.load(latest)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = int(checkpoint['epoch'])
                print("Succesfully loaded checkpoint (epoch {})...".format(self.start_epoch))

    def save_figure(self, path):
        """
            Saves a training\validation classification accuracy curve
                > This function is to be called after performing training

        :param path: Path to save figure
        """

        if len(self.train_accs) > 0:
            print("Saving accuracy plot...")
            figure = plt.figure()
            plt.plot(list(range(self.start_epoch, self.start_epoch + len(self.train_accs))), self.train_accs, '.r-')
            plt.xlabel('Epoch')
            plt.ylabel('Average Accuracy [%]')

            if len(self.test_accs) > 0:
                figure.suptitle('Training\\Testing Accuracy')
                plt.plot(list(range(self.start_epoch + self.valid_period,
                                    self.start_epoch + self.valid_period * (1 + len(self.test_accs)),
                                    self.valid_period)
                              ),
                            self.test_accs, '.b-')
            else:
                figure.suptitle('Training Accuracy')

            plt.show()
            figure.savefig(path)

    @abstractmethod
    def forward_pass(self, sample):
        """
        :param sample: Tuple containing 0: Labels
                                        1: Input features

        :return: A loss object that is used for optimizer updates (e.g. backward pass)
        """
        pass

    @abstractmethod
    def define_model(self, dim_in, train_features=None):
        """
            Defines a network's architecture, used by forward_pass

        :param dim_in: Expected dimension of inputs features

        :return: A Torch "torch.nn.Module" object
        """
        pass



class FeatureExtractor(ABC):
    """
        Transforms raw EMG/IMU samples into a feature vector used for classification
    """

    requires_global_setup   = False

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