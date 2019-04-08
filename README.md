# *ninaeval* Python Package

NinaTools, and the PythonMyoLinux repository, constitute the work of my final year capstone project (poster below).
![image](https://drive.google.com/uc?export=view&id=1tMUZbA5dVqKowB9Ruh7uCwTAUb2KQM0H)



**The *ninaeval* package can:**

1. Download, process and format the NinaPro dataset (as well as newly created data).
2. Extract features to create training/validation/testing sets.
3. Compare the performance of various classifier models.

**NinaPro dataset source (DB5):**
```
@article{
    author = {Pizzolato, Stefano and Tagliapietra, Luca and Cognolato, Matteo and Reggiani, Monica and M{\"{u}}ller, Henning and Atzori, Manfredo},
     title = {Comparison of Six Electromyography Acquisition Setups on Hand Movement Classification Tasks},
   journal = {Plos One},
      year = {2017}
}
```
&nbsp;


## Example Usage: *main.py*
```

from ninaeval.config import config_parser, config_setup
from ninaeval.utils.nina_data import NinaDataParser, BaselineDataset, LogicalDatasetV1

DATA_PATH = "all_data/"
MODEL_PATH = "all_models/"

def main():

    # Reads JSON file via --json, or command line arguments:
    config_param = config_parser.parse_config()

    # Basic setup:
    feat_extractor  = config_setup.get_feat_extract(config_param.features)()
    classifier      = config_setup.get_model(config_param.model)(MODEL_PATH, feat_extractor)

    # Generate a dataset, if necessary:
    print("Checking for existing features extracted...")
    dataset = LogicalDatasetV1(DATA_PATH, feat_extractor, False)

    if not dataset.load_dataset():
        data_parser = NinaDataParser(DATA_PATH)
        print("Loading Ninapro data from processed directory...")
        loaded_nina = data_parser.load_processed_data()

        print("Extracting dataset features for training, and testing...")
        dataset.create_dataset(loaded_nina)

    # Train on the training dataset:
    print("Training classifier on training dataset...")
    classifier.train_model(dataset.train_features, dataset.train_labels, dataset.test_features, dataset.test_labels)
    classifier.save_figure("Training_Accuracy.png")

    # Classify the testing dataset:
    print("Testing classifier on testing dataset...")
    print(classifier.perform_inference(dataset.test_features, dataset.test_labels))

if __name__ == "__main__":
    main()
```
