# standard imports
from pathlib import PurePath
import logging

# package imports
from src.search import PipelineSearch
from src.helper.settings import TARGETS_BINARY
from src.helper.utils import _SMOTE

# surpress FutureWarnings
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - [%(filename)s:%(lineno)s - %(funcName)20s() ] - %(message)s"
    )
    logging.info("Start the run!")
    results_filepath = PurePath("results")  # TODO: Define the input/output paths
    input_file_path = PurePath("input/input_file.xlsx")
    target = TARGETS_BINARY[0]
    task_type = "Binary_Classification"
    first_search = PipelineSearch(
        file_path=input_file_path,
        target=target,
        task_type=task_type,
        results_filepath=results_filepath,
    )

    first_search.search("DTR", sampler=_SMOTE)


if __name__ == "__main__":
    main()
