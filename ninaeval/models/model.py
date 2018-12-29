from abc import ABC, abstractmethod
import torch
import torchnet
from tqdm import tqdm
import os
import glob
import matplotlib.pyplot as plt

# For all torch models
torch.manual_seed(13)

NUM_GESTURES = 53 # 52 + 1 (rest)

#
# All classifiers perform this basic functionality
#
class ClassifierModel(ABC):

    def __init__(self, models_path, feat_extractor):
        self.models_path    = models_path
        self.feat_extractor = feat_extractor

    @abstractmethod
    def train_model(self, train_features, train_labels, valid_features = None, valid_labels = None):
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

    @abstractmethod
    def save_figure(self, path):
        pass

    def classifier_accuracy(self, predictions, test_labels):
        errors      = predictions == test_labels
        acc_rate    = len([x for x in errors if (x == True)]) / len(errors)
        return acc_rate

#
# All Torch classifiers implement this interface.
#
class TorchModel(ClassifierModel):

    # Defaults
    weight_decay    = 1e-6
    learning_rate   = 1e-3
    batch_size      = 32
    num_workers     = 1
    num_epoch       = 55
    chkpt_period    = 10
    valid_period    = 1

    checkpoint_ext  = ".chkpt"

    # Do not modify
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
        self.device     = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.model = None

    def train_model(self, train_features, train_labels, valid_features = None, valid_labels = None):
        self.valid_features = valid_features
        self.valid_labels   = valid_labels

        dim_in      = train_features[0].shape[0]
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
            # train_labels    = torch.from_numpy(train_labels).reshape(len(train_labels), 1)
            # train_labels    = (
            #                        train_labels == torch.arange(NUM_GESTURES).reshape(1, NUM_GESTURES)
            #                ).type(torch.LongTensor)

            torch_dataset = torchnet.dataset.TensorDataset([
                torch.from_numpy(train_features),
                torch.from_numpy(train_labels)
            ])
            iterator = torch_dataset.parallel(
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True
            )
            self.engine.train(self.forward_pass, iterator, maxepoch=self.num_epoch - self.start_epoch, optimizer=self.optimizer)

        print("Training complete ({} epochs)...".format(self.num_epoch))

    def perform_inference(self, test_features, test_labels):
        # Use inference model
        self.model.eval()
        self.classerr.reset()

        torch_dataset = torchnet.dataset.TensorDataset([
            torch.from_numpy(test_features),
            torch.from_numpy(test_labels)
        ])
        iterator = torch_dataset.parallel(
            batch_size  = self.batch_size,
            num_workers = self.num_workers,
            shuffle     = False
        )
        self.engine.test(self.forward_pass, iterator)

    def on_sample(self, state):
        state['sample'].append(state['train'])

    def on_forward(self, state):
        self.classerr.add(state['output'].data, torch.LongTensor(state['sample'][1]))
        self.meter_loss.add(state['loss'].item())

    def reset_meters(self):
        self.classerr.reset()
        self.meter_loss.reset()

    def on_start_epoch(self, state):
        self.reset_meters()
        # Buggy...
        #state['iterator'] = tqdm(state['iterator'])

    def on_end_epoch(self, state):
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
            self.perform_inference(self.valid_features, self.valid_labels)
            self.model.train()

    def on_end(self, state):
        print('Training' if state['train'] else 'Testing', 'accuracy')
        print(self.classerr.value())

        if not state["train"]:
            self.test_accs.append(self.classerr.value())
        self.reset_meters()

    def save_checkpoint(self, epoch, loss):
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
        pass

    @abstractmethod
    def define_model(self, dim_in):
        pass


class FeatureExtractor(ABC):

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