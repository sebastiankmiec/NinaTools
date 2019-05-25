from ninaeval.config import config_parser, config_setup
from ninaeval.utils.data_extract import extract_myo_all_csv

DATA_PATH   = "all_data/"
MODEL_PATH  = "all_models/"

def main():

    # Reads JSON file via --json, or command line arguments:
    config_param = config_parser.parse_config()

    feat_extractor  = config_setup.get_feat_extract(config_param.features)()
    classifier      = config_setup.get_model(config_param.model)(MODEL_PATH, feat_extractor)
    dataset         = config_setup.get_dataset(config_param.data)(DATA_PATH, feat_extractor, False)

    #
    # Use "extract_myo_all_csv" to load myo_all_data.csv data (obtained from our data collecton GUI) into "new_data".
    #
    if not dataset.load_dataset():

        new_data = {}

        #
        # Your own data paths....
        #

        # extract_myo_all_csv('/home/skmiec/Documents/ex5/a/myo_all_data.csv', new_data, "s11", "E1")
        # extract_myo_all_csv('/home/skmiec/Documents/ex5/b/myo_all_data.csv', new_data, "s11", "E2")
        # extract_myo_all_csv('/home/skmiec/Documents/ex5/c/myo_all_data.csv', new_data, "s11", "E3")

        # extract_myo_all_csv('/home/skmiec/Documents/ex6/a/myo_all_data.csv', new_data, "s12", "E1")
        # extract_myo_all_csv('/home/skmiec/Documents/ex6/b/myo_all_data.csv', new_data, "s12", "E2")
        # extract_myo_all_csv('/home/skmiec/Documents/ex6/c/myo_all_data.csv', new_data, "s12", "E3")

        print("Extracting dataset features for training, and testing...")
        dataset.create_dataset(new_data, False)

    print("Training classifier on training dataset...")
    classifier.train_model(dataset.train_features, dataset.train_labels, dataset.test_features, dataset.test_labels)

    print("Testing classifier on testing dataset...")
    print(classifier.perform_inference(dataset.test_features, dataset.test_labels))


if __name__ == "__main__":
    main()