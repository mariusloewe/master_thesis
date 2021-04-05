# standard imports
import logging
from pathlib import PurePath
import os
import json

# project imports
from src.helper.settings import FEATURE_LIST, SEED
from src.helper.utils import df_to_csv

# DS imports
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import date

# sklearn imports
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler


class PipelineSearch:
    def __init__(
        self, file_path, target, task_type, results_filepath, test_size=0.30, seed=None, no_features_selected=10,
    ):

        self.target = target
        self.task_type = task_type
        self.raw_data = self._import_data(file_path)
        self.results_filepath = results_filepath
        self.seed = seed if seed is not None else SEED
        self.test_size = test_size
        self.no_features_selected = no_features_selected
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

    def search(
        self, model_class, sampler=None, feature_selector=None, scoring_function = mean_squared_error, n_jobs=3
    ):
        """
        Public Function that carries out the Pipeline Search.
        """
        self._GSCV_REG(
            model_class,
            sampler=sampler,
            scoring_function=scoring_function,
            feature_selector=feature_selector,
            n_jobs=n_jobs,
        )

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

        # first check if the file already exists
        file_name = "preprocessed_data_{}_target.csv".format(self.target)
        directory = PurePath("input").joinpath("processed_data")

        # check if file is already present and returns if its the case
        if self._load_processed_date(directory, file_name):
            return

        self.processed_data = self.raw_data.copy()

        def calculate_age(born, ref_date='2019-09-14'):
            """
            Helper Function to calculate the age.
            """
            born = pd.to_datetime(born)
            today = (
                datetime.strptime(ref_date, '%Y-%m-%d')
                if ref_date is not None
                else pd.to_datetime("today")
            )
            return (today - born) / np.timedelta64(1, "Y")

        #calulate the age from the DoB
        self.processed_data['age'] = self.processed_data['DoB'].apply(calculate_age)
        self.processed_data = self.processed_data.drop(['DoB'], axis=1)

        #encode Gender
        self.processed_data.Gender = self.processed_data.Gender.str.rstrip()
        self.processed_data['Gender'] = self.processed_data['Gender'].replace({'M': '0', 'F': '1'}).astype('int32')

        #strip spacings and lower case for uniform categoricals
        self.processed_data['Primary Tumour'] = self.processed_data['Primary Tumour'].str.lower().str.rstrip()
        self.processed_data['First Met Organ Site'] = self.processed_data['First Met Organ Site'].str.lower().str.rstrip()

        # fill NAN where possible and sensible
        self.processed_data['SOMA > 1st OM  0=N 1=Y'] = self.processed_data['SOMA > 1st OM  0=N 1=Y'].replace(('NA ', '0')).fillna(0)
        self.processed_data['N of targets    I SOMA'] = self.processed_data['N of targets    I SOMA'].fillna(0).astype('int32')
        self.processed_data['Largest single SOMA burden'] = self.processed_data['Largest single SOMA burden'].fillna(0).astype('int32')
        self.processed_data['Tumour burden         I SOMA'] = self.processed_data['Tumour burden         I SOMA'].fillna(0).astype('int32')

        #ToDo: Marius check logic if they are usable by any means, probably, yes
        # remove unusable columns

        self.processed_data = self.processed_data.drop([
            'Interval between ablations',
               'Min %Δ Tumour burden',
               'Max %Δ Tumour burden',
               'Mean %Δ Tumour burden',
               'Min SOMA interval',
               'Max SOMA Interval',
               'Average SOMA Interval',
               ], axis=1)



        # Encode categorical features to dummy variables
        self.processed_data = pd.get_dummies(data=self.processed_data,columns=['Primary Tumour', 'First Met Organ Site','Same organ (0:Y 1:N 2:Both)'])

        logging.info(
            "Dataframe after preprocessing has shape {}. {} Patients were removed from dataset.".format(
                self.processed_data.shape, self.processed_data.isnull().sum(axis=0).sum()
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
            self._scale_split_data()

        logging.info("after split shape, x: {}".format(self.x_train.shape))
        logging.info("after split cols: {}".format(self.x_train.columns))
        logging.info("after split shape, y: {}".format(self.y_train.shape))
        logging.info(
            "after split x dtypes: {}".format(self.x_train.dtypes.value_counts())
        )
        logging.info("after split y dtypes: {}".format(self.y_train.dtypes))

    def _scale_split_data(self):
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

    def _GSCV_REG(
        self,
        model_class,
        sampler: callable,
        scoring_function,
        n_jobs,
        feature_selector: callable,
        pca=None,
    ):

        features = False

        # hacky way to distinguish between regression and classification grid
        if self.task_type == "regression":
            from src.helper.regression_grid import PARAMETERS, MODELS

        elif self.task_type == "classification":
            from src.helper.classification_grid import PARAMETERS, MODELS

        if feature_selector is not None:
            if not callable(feature_selector):
                raise ReferenceError(
                    "Feature Selector {} is not a callable.".format(feature_selector)
                )
            else:
                features = feature_selector(
                    self.x_train_scaled, self.y_train_scaled, self.no_features_selected
                )
                self.x_train_scaled = self.x_train_scaled.loc[:, features]
                self.x_test_scaled = self.x_test_scaled.loc[:, features]

                feature_names = list(self.x_train_scaled.columns)
                logging.info("after feature_selection shape, x: {}".format(self.x_train_scaled.shape))

        if sampler is not None:
            X_train_sampled, y_train_sampled = sampler(
                self.seed, self.x_train_scaled, self.y_train_scaled, self.target
            )
        else:
            X_train_sampled, y_train_sampled = self.x_train_scaled, self.y_train_scaled

        if pca is not None:
            # assumption is, that the PCA object got instantiated already
            X_train_sampled, y_train_sampled = pca.fit_transform(
                x=np.array(X_train_sampled), y=np.array(y_train_sampled)
            )

        model = MODELS[model_class]
        model_instance = model()
        parameter = PARAMETERS[model_class]

        possible_parameter = set(model_instance.get_params().keys())
        not_possible_parameter = set(parameter.keys()) - possible_parameter
        logging.info("Not functioniing parameters: {}".format(not_possible_parameter))

        gscv = GridSearchCV(
            estimator=model_instance,
            param_grid=parameter,
            cv=5,
            scoring=scoring_function,
            n_jobs=n_jobs,
            verbose=0,
        )
        gscv.fit(X_train_sampled, y_train_sampled)

        logging.info("GSCV Results:", gscv.cv_results_)

        output_dict = {
            "PCA": str(pca),
            "feature_selector": str(feature_selector),
            "sampler": str(sampler),
            "model_class": model_class,
            "params": gscv.best_params_,
            "seed": self.seed,
            "features": feature_names if feature_names else "ALL",
        }
        output_file_name = str(self.target) + "_" + str(model_class) + ".json"
        json.dump(
            output_dict,
            open(self.results_filepath.joinpath(output_file_name), "w"),
            indent=4,
        )
        logging.info("best estimator is: {}".format(gscv.best_estimator_))
        logging.info("best training acc. is: {}".format(gscv.best_estimator_.score(X_train_sampled, y_train_sampled)))
        logging.info("best score are: {}".format(gscv.best_score_))
        logging.info("best parameters are: {}".format(gscv.best_params_))

        best_estimator = gscv.best_estimator_
