from ninaeval.config import config_parser, config_setup
from ninaeval.utils.nina_data import NinaDataParser, BaselineDataset

DATA_PATH = "all_data/"

def main():

    # Reads JSON file via --json, or command line arguments:
    config_param = config_parser.parse_config()

    # Basic setup:
    classifier      = config_setup.get_model(config_param.model)()
    feat_extractor  = config_setup.get_feat_extract(config_param.features)()

    # Generate a dataset, if necessary:
    print("Checking for existing features extracted...")
    dataset = BaselineDataset(DATA_PATH, feat_extractor)

    if not dataset.load_dataset():
        data_parser = NinaDataParser(DATA_PATH)
        print("Loading Ninapro data from processed directory...")
        loaded_nina = data_parser.load_processed_data()

        print("Extracting dataset features for training, and testing...")
        dataset.create_dataset(loaded_nina)

    # Train on the training dataset:
    print("Training classifier on training dataset...")
    classifier.train_model(dataset.train_features, dataset.train_labels)

    # Classify the testing dataset:
    print("Testing classifier on testing dataset...")
    print(classifier.perform_inference(dataset.test_features, dataset.test_labels))

if __name__ == "__main__":
    main()