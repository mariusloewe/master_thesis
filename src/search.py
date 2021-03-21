# standard imports
import logging
from pathlib import PurePath
import os
import shutil
from datetime import datetime, date

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# project imports
from src.settings import FEATURE_LIST, SEED
from src.grid import PARAMETERS, MODELS
from src.utils import df_to_csv

import sys
import pandas as pd
import numpy as np
from datetime import date
from datetime import datetime
from sklearn.model_selection import train_test_split

# from data_loader import Dataset
# from data_preprocessing import Processor
# from feature_engineering import FeatureEngineer
# from model import grid_search_MLP, assess_generalization_auprc, grid_search_RF
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Lasso

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve

from imblearn.pipeline import Pipeline

from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet
from sklearn.dummy import DummyClassifier

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score

# import scikitplot as skplt
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree
from scipy.stats import mstats


from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# import gplearn
import warnings
from sklearn.exceptions import DataConversionWarning

# from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# import graphviz
# import pydotplus
from IPython.display import Image
from sklearn.tree import export_graphviz

# from sklearn.externals.six import StringIO
from scipy.stats import mannwhitneyu
from sklearn.metrics import classification_report
import re
from math import nan

# import torch
# import torchvision as tv
# import torchvision.transforms as transforms
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable
from sklearn.ensemble import (
    GradientBoostingRegressor,
    AdaBoostRegressor,
    RandomForestRegressor,
)

# from gplearn.genetic import SymbolicRegressor
# from gplearn.genetic import SymbolicClassifier#
from sklearn.neighbors import KNeighborsClassifier
import math

# import featuretools as ft
# from torchvision.utils import save_image


class PipelineSearch:
    def __init__(
        self, file_path, target, task_type, results_filepath, test_size=0.30, seed=None
    ):

        self.target = target
        self.task_type = task_type
        self.raw_data = self._import_data(file_path)
        self.results_filepath = results_filepath
        self.seed = seed if seed is not None else SEED
        self.test_size = test_size
        ##PRE-PROCESS
        self.processed_data = None
        ##SPLIT
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        ## Scaled Data
        self.x_train_scaled = None
        self.x_test_scaled = None
        self.y_train_scaled = None
        self.y_test_scaled = None

        # Fill above defined attributes
        self._preprocessing()
        self._split_data()

    def search(self, model_class, sampler=None, scoring_function=mean_squared_error, n_jobs=3):
        """
        Public Function that carries out the Pipeline Search.
        """
        self._GSCV_REG(model_class, sampler=sampler, scoring_function=scoring_function, n_jobs=n_jobs)

    def _pick_features(self, df):
        """
        This Function filters the imported dataframe by the FEATURE_LIST provided in settings.py.
        :param df:
        :return: filtered df
        """
        # append the target to the feature list
        feature_list_extended = FEATURE_LIST.copy()
        feature_list_extended.append(self.target)
        df = df.filter(items=feature_list_extended)
        logging.info(
            "The imported file has shape {} and contains following columns {}.".format(
                df.shape, df.columns.values
            )
        )
        return df

    def _import_data(self, file_path):
        """
        Internal Helper Function to check whether a pickle already exists or not.
        """
        # set up the file paths for importing and exporting the pickled df
        cwd = os.getcwd()
        export_file_path = PurePath(cwd).joinpath("tmp")
        export_file_name = str(self.target) + "_" + str(date.today()) + ".p"
        full_path = export_file_path.joinpath(export_file_name)

        try:
            df = pd.read_pickle(full_path)
            logging.info("Raw_data pickle imported.!")
        except FileNotFoundError:
            df = self._pick_features(pd.read_excel(file_path))

            try:
                os.mkdir(export_file_path)
                df.to_pickle(full_path)
                logging.info("File dumped as {}".format(full_path))
            except:
                logging.info("File dump failed. Path: {}".format(full_path))

        return df

    def _load_processed_date(self, directory, file_name):
        """
        Loads processed data from csv file and sets as self.processed_data. Returns False if file not found.
        """
        try:
            file_path = PurePath(directory).joinpath(file_name)
            self.processed_data = pd.read_csv(file_path)
            logging.info(
                "Imported processed data from {} with shape {}.".format(
                    file_path, self.processed_data.shape
                )
            )
            return True
        except FileNotFoundError:
            logging.info(
                "Processed Data file not found - continuing in preprocessing.!"
            )
            return False

    def _preprocessing(self):
        """
        This internal function contains as a wrapper for the preprocessing steps, which are:
        i.) Remove Nan's with SOMA Patients
        ii.) One-Hot Encode Gender, Primary Tumor, First met Organ Site
        iii.) Get the current age of patients
        iv.) Sets the result as self.processed_date and dumps it on disk.
        """

        # TODO: Marius: Are you sure you want the age "today" and not the age of the first examination?
        # Also, think about reproducability - at least I would have the date fixed
        def calculate_age(born, ref_date=None):
            """
            Helper Function to calculate the age.
            """
            born = pd.to_datetime(born)
            today = (
                pd.to_datetime(ref_date)
                if ref_date is not None
                else pd.to_datetime("today")
            )
            return (today - born) / np.timedelta64(1, "Y")

        # first check if the file already exists
        file_name = "preprocessed_data_{}_target.csv".format(self.target)
        directory = PurePath("input").joinpath("processed_data")

        # check if file is already present and returns if its the case
        if self._load_processed_date(directory, file_name):
            return

        num_valid_entries = 175
        self.processed_data = self.raw_data[0:num_valid_entries].copy()

        # TODO: Marius validate this part, it seems very odd to me how you remove nan's
        # filter out all not SOMA patients
        temp = self.processed_data.loc[
            self.processed_data["SOMA            1=Y 0=N"] == 0
        ]
        # replace all nans with 0
        temp = temp.replace(np.nan, 0)
        self.processed_data = self.processed_data.loc[
            self.processed_data["SOMA            1=Y 0=N"] == 1
        ]
        self.processed_data = self.processed_data.append(temp)
        self.processed_data = self.processed_data.dropna()

        self.processed_data["Gender"] = self.processed_data["Gender"].apply(
            lambda x: str(x).lower().strip()
        )
        self.processed_data["Primary Tumour"] = self.processed_data[
            "Primary Tumour"
        ].apply(lambda x: str(x).lower().strip())
        self.processed_data["First Met Organ Site"] = self.processed_data[
            "First Met Organ Site"
        ].apply(lambda x: str(x).lower().strip())

        # one hot encode categorical columns
        self.processed_data = pd.concat(
            [
                self.processed_data,
                pd.get_dummies(
                    self.processed_data["Primary Tumour"], prefix="Primary Tumour"
                ),
            ],
            axis=1,
        )
        self.processed_data = pd.concat(
            [
                self.processed_data,
                pd.get_dummies(self.processed_data["Gender"], prefix="Gender"),
            ],
            axis=1,
        )
        self.processed_data = pd.concat(
            [
                self.processed_data,
                pd.get_dummies(
                    self.processed_data["First Met Organ Site"],
                    prefix="First Met Organ Site",
                ),
            ],
            axis=1,
        )
        logging.info(
            "After one-hot-encoding following columns are present {}".format(
                self.processed_data.columns.values
            )
        )

        # get the current age of patients
        self.processed_data["age"] = self.processed_data["DoB"].apply(calculate_age)

        # dropping the columns - why do we drop gender_m?
        self.processed_data = self.processed_data.drop(
            columns=[
                "DoB",
                "Gender",
                "Gender_m",
                "Primary Tumour",
                "First Met Organ Site",
            ]
        )

        # Remove rows with NaN values
        self.processed_data = self.processed_data.dropna()
        logging.info(
            "Dataframe after preprocessing has shape {}. {} Patients were removed from dataset.".format(
                self.processed_data.shape,
                (num_valid_entries - self.processed_data.shape[0]),
            )
        )

        # dump to disc
        df_to_csv(self.processed_data, directory, file_name)

    def _split_data(self, scaled=True):
        """
        Splitting the processed data into X and y, train and test.
        """
        X_ = self.processed_data.drop([self.target], axis=1)
        y_ = self.processed_data[self.target]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            X_, y_, test_size=self.test_size, random_state=self.seed
        )
        if scaled:
            self._scale_split_date()

        logging.info("after split shape, x: {}".format(self.x_train.shape))
        logging.info("after split cols: {}".format(self.x_train.columns))
        logging.info("after split shape, y: {}".format(self.y_train.shape))
        logging.info(
            "after split x dtypes: {}".format(self.x_train.dtypes.value_counts())
        )
        logging.info("after split y dtypes: {}".format(self.y_train.dtypes))

    def _scale_split_date(self):
        """
        MinMaxScales the data.
        """

        def _scale_floats(df):
            # hacky way to get all columns that contain only floats
            tmp_cols = [key for key, value in df.dtypes.items() if value.kind is "f"]
            tmp_df = df[tmp_cols]
            scaler = MinMaxScaler()
            df[tmp_cols] = scaler.fit_transform(tmp_df)
            return df

        def _scale_y(s):
            scaler = MinMaxScaler()
            return scaler.fit_transform(s)

        self.x_train_scaled = _scale_floats(self.x_train)
        self.x_test_scaled = _scale_floats(self.x_test)
        self.y_train_scaled = (
            _scale_y(self.y_train) if self.task_type is "Regression" else self.y_train
        )
        self.y_test_scaled = (
            _scale_y(self.y_test) if self.task_type is "Regression" else self.y_test
        )

        logging.info(
            "Scaled all datasets - Checking if shapes are the same X_test: {}, X_train: {}, y_test: {}, y_train: {}".format(
                self.x_train_scaled.shape == self.x_train.shape,
                self.x_test_scaled.shape == self.x_test.shape,
                self.y_train_scaled.shape == self.y_train.shape,
                self.y_test_scaled.shape == self.y_test.shape,
            )
        )

    def _GSCV_REG(self, model_class, sampler: callable, scoring_function, n_jobs):

        if sampler is not None:
            X_train_sampled, y_train_sampled = sampler(self.seed, self.x_train_scaled, self.y_train_scaled, self.target)
        else:
            X_train_sampled, y_train_sampled = self.x_train_scaled, self.y_train_scaled

        model = MODELS[model_class]
        model_instance = model()
        parameter = PARAMETERS[model_class]

        gscv = GridSearchCV(
            estimator=model_instance,
            param_grid=parameter,
            cv=3,
            scoring=make_scorer(scoring_function),
            n_jobs=n_jobs,
            verbose=1
        )
        gscv.fit(X_train_sampled, y_train_sampled)


        print(gscv.cv_results_)
        print("best estimator is: {}".format(gscv.best_estimator_))
        print("best score are: {}".format(gscv.best_score_))
        print("best parameters are: {}".format(gscv.best_params_))
        # self.best_classifier[name] = gscv

        best_estimator = gscv.best_estimator_


