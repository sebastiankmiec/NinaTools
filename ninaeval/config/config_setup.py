from ninaeval.models import baseline_model, advanced_model
from ninaeval.config.config_parser import model_choices, feature_choices, dataset_choices
from ninaeval.utils import nina_data

def get_model(model_abbrev):
    """
    :param model_abbrev: An abbreviated model name (config_parser.py).
    :return: ClassifierModel object
    """

    model = None
    try:
        model = getattr(baseline_model, model_choices[model_abbrev])
    except AttributeError:
        try:
            model = getattr(advanced_model, model_choices[model_abbrev])
        except AttributeError:
            pass

    return model

def get_feat_extract(features_abbrev):
    """
    :param features_abbrev: An abbreviated model name (config_parser.py).
    :return: FeatureExtractor object
    """

    feat_ext = None
    try:
        feat_ext = getattr(baseline_model, feature_choices[features_abbrev])
    except AttributeError:
        try:
            feat_ext = getattr(advanced_model, feature_choices[features_abbrev])
        except AttributeError:
            pass

    return feat_ext


def get_dataset(dataset_abbrev):
    """
    :param dataset_abbrev: An abbreviated dataset name (config_parser.py).
    :return: Dataset object
    """

    dataset = None
    try:
        dataset = getattr(nina_data, dataset_choices[dataset_abbrev])
    except AttributeError:
        pass

    return dataset