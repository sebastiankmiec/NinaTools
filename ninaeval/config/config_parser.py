from optparse import OptionParser
from argparse import ArgumentParser
import json

#
# Model/Feature abbreviations and choices.
#
model_choices   = {"rf": "RandomForest", "svm": "SupportVectorMachine", "fcn": "FullyConnectedNN",
                    "fcn2": "FullyConnectedNNV2", "vlad": "VLAD", "rnn": "RecurrentNN"}

feature_choices = {"rms": "RMS", "ts": "TimeStatistics", "mdwt": "MarginalDiscreteWaveletTransform",
                    "hist": "HistogramBins", "all": "AllFeatures", "scat1d": "ScatteringFeat1D",
                    "none": "EmptyFeature", "multirms": "MultipleRMS", "pmrms" :"PaddedMultiRMS",
                    "kmrms": "KMeansRMS", "imuprms": "IMUPaddedMultiRMS"}

dataset_choices = {"base": "BaselineDataset", "v1":"LogicalDatasetV1", "basevar": "BaselineVariableWindowDataset",
                    "v1var": "LogicalVariableWindowDataset", "basevarimu":"IMUVariableWindowDataset"}

########################################################################################################################
########################################################################################################################


def init_parser():
    '''
        Create a parser to read command line or JSON file configuration parameters.

        :return: ArgumentParser
    '''

    parser = OptionParser()

    #
    # Model choices
    #
    parser.add_option('--model', action='store', type='choice', default='rf',
                     choices=list(model_choices.keys()))

    parser.add_option('--features', action='store', type='choice', default='rms',
                      choices=list(feature_choices.keys()))

    #
    # Dataset choices
    #
    parser.add_option('--data', action='store', type='choice', default='v1',
                      choices=list(dataset_choices.keys()))

    return parser


def json_to_string(json_path):
    """
        Helper function (parse_config): Parses a JSON file.

    :param json_path: Path to a JSON file.
    :return: list of JSON file arguments, in command-line format
    """

    if not json_path.endswith(".json"):
        raise ValueError('Expecting a .json file passed to --json argument.')

    try:
        f           = open(json_path)
        raw_json    = f.read().replace("\n", "")
        json_dict   = json.loads(raw_json)
    except Exception as e:
        print("Invalid JSON file passed to --json.")
        print(e)
        exit()

    # Convert JSON dict to a list of command line arguments:
    command_list    = []
    json_keys       = list(json_dict.keys())

    for key in json_keys:

        cur_command = ""
        if not "--" in key:
            cur_command += "--"

        cur_command += key + "=" + json_dict[key]
        command_list.append(cur_command)

    return command_list



def parse_config():
    '''
        Checks for --json for a JSON file, otherwise, reads command line arguments.

    :return: A dictionary of configuration options
    '''

    # Check for "--json" argument first;
    json_parser   = ArgumentParser()
    json_parser.add_argument("--json", action="store", dest="JSON_PATH")
    json_path, _  = json_parser.parse_known_args()
    json_path     = json_path.JSON_PATH

    if json_path:
        argparser       = init_parser()
        command_list    = json_to_string(json_path)
        options, _  = argparser.parse_args(args=command_list)

    else:
        argparser  = init_parser()
        options, _ = argparser.parse_args()

    return options