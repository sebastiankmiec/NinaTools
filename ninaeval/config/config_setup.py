from ninaeval.models import baseline_model
from ninaeval.config.config_parser import model_choices, feature_choices

def get_model(model_abbrev):
    """
    :param model_abbrev: An abbreviated model name (config_parser.py).
    :return: ClassifierModel
    """

    model = None
    try:
        model = getattr(baseline_model, model_choices[model_abbrev])
    except AttributeError:
        pass

    return model

def get_feat_extract(features_abbrev):
    """
    :param features_abbrev: An abbreviated model name (config_parser.py).
    :return: FeatureExtractor
    """

    feat_ext = None
    try:
        feat_ext = getattr(baseline_model, feature_choices[features_abbrev])
    except AttributeError:
        pass

    return feat_ext