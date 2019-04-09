from ninaeval.config import config_parser, config_setup
from ninaeval.utils.nina_data import NinaDataParser

DATA_PATH = "all_data/"
MODEL_PATH = "all_models/"

def main():

    # Reads JSON file via --json, or command line arguments:
    config_param = config_parser.parse_config()

    feat_extractor  = config_setup.get_feat_extract(config_param.features)()
    classifier      = config_setup.get_model(config_param.model)(MODEL_PATH, feat_extractor)
    dataset         = config_setup.get_dataset(config_param.data)(DATA_PATH, feat_extractor, False)

    if not dataset.load_dataset():
        data_parser = NinaDataParser(DATA_PATH)

        print("Loading Ninapro data from processed directory...")
        loaded_nina = data_parser.load_processed_data()

        print("Extracting dataset features for training, and testing...")
        dataset.create_dataset(loaded_nina)

    print("Training classifier on training dataset...")
    classifier.train_model(dataset.train_features, dataset.train_labels, dataset.test_features, dataset.test_labels)

    print("Testing classifier on testing dataset...")
    print(classifier.perform_inference(dataset.test_features, dataset.test_labels))


if __name__ == "__main__":
    main()