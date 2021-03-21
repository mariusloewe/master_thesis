# standard imports
import logging
from pathlib import PurePath
import os
import shutil
from datetime import datetime, date

# project imports
from src.settings import FEATURE_LIST, SEED
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

# from imblearn.pipeline import Pipeline

from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR

from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet
from sklearn.dummy import DummyClassifier

from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFE
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


from sklearn.preprocessing import LabelEncoder

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

# configure the logging
LOGGER = logging.getLogger("LOGGER")
# TODO: Fix logging
# LOGGER.setFormatter(logging.Formatter(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s'))


class PipelineSearch:
    def __init__(
        self, file_path, target, task_type, results_filepath, test_size=0.30, seed=None
    ):

        self.target = target
        self.task_type = task_type
        self.raw_data = self._import_data(file_path)
        self.processed_data = None
        self.results_filepath = results_filepath
        self.seed = seed if seed is not None else SEED
        self.test_size = test_size

        ##PRE-PROCESS
        self._preprocessing()

        ##SPLIT
        self.x_train = pd.DataFrame()
        self.y_train = pd.DataFrame()
        self.x_test = pd.DataFrame()
        self.y_test = pd.DataFrame()
        # self._split_data()

        ##DATA_PREP
        # self._Scaling_Features()
        self.best_classifier = {}
        self.X_train_smote = np.zeros(shape=(2636, 6))
        self.y_train_smote = np.zeros(shape=(2636,))
        # self._SMOTE()
        # self._SMOTE_Boarder()
        # self._SMOTE_SVM()
        # self._SMOTETomek()
        # IMPLEMENT: '''SMOTEENN'''
        # self._ANASYN()
        # self._Scaling_Features()
        self.feat_scores = pd.DataFrame()
        # self.select_KBest()
        self.boruta_selection()

        ##MODEL
        self.scoring_metric = make_scorer(f1_score)  #'recall' # make_scorer(f1_score)
        # self._ADABoost()
        self._GSCV_REG()

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
        LOGGER.info(
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
            logging.info("Imported processed data from {}".format(file_path))
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
        LOGGER.info(
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
        LOGGER.info(
            "Dataframe after preprocessing has shape {}. {} Patients were removed from dataset.".format(
                self.processed_data.shape,
                (num_valid_entries - self.processed_data.shape[0]),
            )
        )

        # dump to disc
        df_to_csv(self.processed_data, directory, file_name)

    def _split_data(self):
        X_ = self.raw_data.drop([self.target], axis=1)
        y_ = self.raw_data[self.target]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            X_, y_, test_size=self.test_size, random_state=self.seed
        )
        # self.x_train = self.x_train.astype('float').reset_index(drop=True)
        # self.y_train = self.y_train.astype('float').reset_index(drop=True)
        print("after split shape, x", self.x_train.shape)
        print("after split cols", self.x_train.columns)
        print("after split shape, y", self.y_train.shape)
        print("after split types", self.x_train.dtypes.value_counts())
        print(self.y_train.dtypes)

    def _GSCV_REG(self):

        for i in range(5):
            self.seed = i

            """Train-Test Split"""
            X_ = self.raw_data.drop([self.target], axis=1)
            y_ = self.raw_data[self.target]
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                X_, y_, test_size=0.30, random_state=self.seed
            )
            # self.x_train = self.x_train.astype('float').reset_index(drop=True)
            # self.y_train = self.y_train.astype('float').reset_index(drop=True)
            print("after split shape, x", self.x_train.shape)
            print("after split cols", self.x_train.columns)
            print("after split shape, y", self.y_train.shape)
            print("after split types", self.x_train.dtypes.value_counts())
            print(self.y_train.dtypes)

            """Scaling Features"""
            num_cols = self.x_train.columns  # .select_dtypes(include=['float']).columns
            # num_cols = self.x_train.columns[self.x_train.dtypes.apply(lambda c: np.issubdtype(c, np.number))]
            # print('Numerical Columns: \n', num_cols)

            # Only scale continuous values
            scaler = MinMaxScaler()
            # if Bucketing - every feature is dtype object - so no need to specify to scale only numerical features

            self.x_train[num_cols] = scaler.fit_transform(self.x_train[num_cols])
            print(len(self.x_train))

            self.x_test[num_cols] = scaler.fit_transform(self.x_test[num_cols])
            print(len(self.x_test))

            for s in range(len(selection)):

                for m in range(len(models)):
                    model = Pipeline(
                        [
                            ("sampling", selection[s]),
                            # ('PCA', PCA()),
                            (models[m]),
                        ]
                    )

                    gscv = GridSearchCV(
                        estimator=model,
                        param_grid=params[m],
                        cv=3,
                        scoring=make_scorer(mean_squared_error),
                        n_jobs=6,
                    )
                    gscv.fit(self.x_train, self.y_train)
                    print(gscv.cv_results_)
                    print("best estimator is: {}".format(gscv.best_estimator_))
                    print("best score are: {}".format(gscv.best_score_))
                    print("best parameters are: {}".format(gscv.best_params_))
                    # self.best_classifier[name] = gscv

                    best_estimator = gscv.best_estimator_

                    best = best_estimator
                    best.fit(self.x_train, self.y_train)

                    y_pred_ = best.predict(self.x_test)

                    aggregate_results = list()
                    median_results = list()

                    for i in range(5):
                        self.seed = i
                        X_ = self.raw_data.drop([self.target], axis=1)
                        y_ = self.raw_data[self.target]
                        (
                            self.x_train,
                            self.x_test,
                            self.y_train,
                            self.y_test,
                        ) = train_test_split(
                            X_, y_, test_size=0.30, random_state=self.seed
                        )

                        num_cols = self.x_train.columns
                        scaler = MinMaxScaler()

                        self.x_train[num_cols] = scaler.fit_transform(
                            self.x_train[num_cols]
                        )
                        print(len(self.x_train))

                        self.x_test[num_cols] = scaler.fit_transform(
                            self.x_test[num_cols]
                        )
                        print(len(self.x_test))

                        best_estimator = gscv.best_estimator_

                        best_estimator.fit(self.x_train, self.y_train)

                        y_pred_ = best_estimator.predict(self.x_test)

                        print(gscv.best_score_)

                        results_string = (
                            str(
                                [
                                    self.seed,
                                    str(gscv.best_score_),
                                    str(mean_absolute_error(self.y_test, y_pred_)),
                                    str(
                                        math.sqrt(
                                            mean_squared_error(self.y_test, y_pred_)
                                        )
                                    ),
                                    str(pearsonr(self.y_test, y_pred_)),
                                    str(gscv.best_estimator_),
                                    str(
                                        gscv.best_params_
                                    ),  # die letzten 2 überprüfen was die genau bedeuten, MSE und RMSE ergänzen
                                    str(selection[s]),
                                    str(gscv.best_params_),
                                ]
                            )
                            + "\n"
                        )
                        print(results_string)
                        with open(self.results_filepath, "a") as f:
                            f.write(results_string)

                        aggregate_results.append(
                            [
                                self.seed,
                                round(gscv.best_score_, 3),
                                mean_absolute_error(self.y_test, y_pred_),
                                math.sqrt(mean_squared_error(self.y_test, y_pred_)),
                                pearsonr(self.y_test, y_pred_)[0],
                                pearsonr(self.y_test, y_pred_)[1],
                            ]
                        )

                        print(aggregate_results)
                        median_results.append(
                            [round(mean_absolute_error(self.y_test, y_pred_), 3)]
                        )

                median = np.median(median_results)
                aggregate_results_mean = np.around(
                    np.mean(aggregate_results, axis=0), decimals=3
                )

                # print(aggregate_results_mean)
                results_str = (
                    str(
                        [
                            aggregate_results_mean[0],
                            aggregate_results_mean[1],
                            aggregate_results_mean[2],
                            median,
                            aggregate_results_mean[3],
                            aggregate_results_mean[4],
                            aggregate_results_mean[5],
                            str(selection[s]),
                            str(gscv.best_estimator_),
                            str(gscv.best_params_),
                        ]
                    )
                    + "\n"
                )
                with open(
                    "C:\\Users\\mariu\\OneDrive\\Dokumente\\NOVA\\Thesis\\20191015_new_data\\Final Results\\AVG\\DTR\\_results_"
                    + str(self.target)
                    + ".csv",
                    "a",
                ) as f:
                    f.write(results_str)
