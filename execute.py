# standard imports
from pathlib import PurePath
import logging

# package imports
from src.search import PipelineSearch
from src.helper.settings import TARGETS_BINARY
from src.helper.utils import _SMOTE, _RFE
from src.helper.classification_grid import MODELS

# surpress FutureWarnings
import warnings

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

#warnings.simplefilter(action="ignore", category=FutureWarning)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(filename)s:%(lineno)s - %(funcName)20s() ] - %(message)s",
    )
    logging.info("Start the run!")
    target = TARGETS_BINARY[2]

    results_filepath = PurePath("results/"+str(target))  # TODO: Define the input/output paths
    input_file_path = PurePath("input/20210403_input_file_new.xlsx")

    task_type = "classification"
    no_features_selected = 7

    #initiate pipeline object
    first_search = PipelineSearch(
        file_path=input_file_path,
        target=target,
        task_type=task_type,
        results_filepath=results_filepath,
        no_features_selected=no_features_selected
    )

    #for key in MODELS.keys():
    first_search.search('LR', sampler=_SMOTE, feature_selector=_RFE, scoring_function='accuracy')  # , PCA=None, seed=SEED


if __name__ == "__main__":
    main()
