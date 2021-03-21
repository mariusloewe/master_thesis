# standard imports
from pathlib import PurePath
import logging


# package imports
from src.search import PipelineSearch
from src.settings import TARGETS_BINARY, TARGETS_REGRESSION, TARGETS_MULTICLASS


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - [%(filename)s:%(lineno)s - %(funcName)20s() ] - %(message)s"
    )
    logging.info("Start the run!")
    results_filepath = PurePath("results")  # TODO: Define the input/output paths
    input_file_path = PurePath("input/input_file.xlsx")
    target = TARGETS_BINARY[0]
    task_type = "Binary_Classification"
    dataframe = PipelineSearch(
        file_path=input_file_path,
        target=target,
        task_type=task_type,
        results_filepath=results_filepath,
    )


if __name__ == "__main__":
    main()
