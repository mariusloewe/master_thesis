# standard imports
import logging

# project imports
from src.settings import FEATURE_LIST
from src.utils import df_to_csv

import sys
import pandas as pd
import numpy as np
from datetime import date
from datetime import datetime
from sklearn.model_selection import train_test_split
#from data_loader import Dataset
#from data_preprocessing import Processor
#from feature_engineering import FeatureEngineer
#from model import grid_search_MLP, assess_generalization_auprc, grid_search_RF
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
#import imblearn.under_sampling as uns
#from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN
#from imblearn.combine import SMOTETomek, SMOTEENN
#from imblearn.pipeline import Pipeline

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
#import scikitplot as skplt
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree
from scipy.stats import mstats
from datetime import datetime, date

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder
# from boruta import BorutaPy
# import gplearn
import warnings
from sklearn.exceptions import DataConversionWarning
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
# import graphviz
# import pydotplus
from IPython.display import Image
from sklearn.tree import export_graphviz
#from sklearn.externals.six import StringIO
from scipy.stats import mannwhitneyu
from sklearn.metrics import classification_report
import re
from math import nan

# import torch
#import torchvision as tv
#import torchvision.transforms as transforms
#import torch.nn as nn
#import torch.nn.functional as F
#from torch.autograd import Variable
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor
#from gplearn.genetic import SymbolicRegressor
#from gplearn.genetic import SymbolicClassifier#
from sklearn.neighbors import KNeighborsClassifier
import math
#import featuretools as ft
#from torchvision.utils import save_image

# configure the logging
LOGGER = logging.getLogger("LOGGER")
# TODO: Fix logging
# LOGGER.setFormatter(logging.Formatter(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s'))


class PipelineSearch:

    def __init__(self, file_path, target, task_type, results_filepath):

        self.target = target
        self.task_type = task_type
        self.raw_data = self._pick_features(pd.read_excel(file_path))

        self.results_filepath = results_filepath

        self.seed = 0 # TODO: check if the seed makes any sense here

        ##PRE-PROCESS
        self.preprocessing()  #-> Fraction split needs to be set here
        #self._binning()
        #self.calc_correlations()

        ##SPLIT
        self.x_train = pd.DataFrame()
        self.y_train = pd.DataFrame()
        self.x_test = pd.DataFrame()
        self.y_test = pd.DataFrame()
        #self._split_data()

        ##DATA_PREP
        #self._Scaling_Features()
        self.best_classifier = {}
        self.X_train_smote = np.zeros(shape=(2636, 6))
        self.y_train_smote = np.zeros(shape=(2636,))
        #self._SMOTE()
        #self._SMOTE_Boarder()
        #self._SMOTE_SVM()
        #self._SMOTETomek()
        #IMPLEMENT: '''SMOTEENN'''
        #self._ANASYN()
        #self._Scaling_Features()
        self.feat_scores = pd.DataFrame()
        #self.select_KBest()
        self.boruta_selection()

        ##MODEL
        self.scoring_metric = make_scorer(f1_score)#'recall' # make_scorer(f1_score)
        #self._ADABoost()
        self._GSCV_REG()
        self._algorithm_opt()

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
        LOGGER.info("The imported file has shape {} and contains following columns {}.".format(df.shape, df.columns.values))
        return df

    def preprocessing(self):
        """
        # TODO: write docstring
        :return:
        """
        # TODO: Marius: Are you sure you want the age "today" and not the age of the first examination?
        # Also, think about reproducability - at least I would have the date fixed
        def calculate_age(born, ref_date=None):
            """
            Helper Function to calculate the age.
            """
            born = pd.to_datetime(born)
            today = pd.to_datetime(ref_date) if ref_date is not None else date.today()
            return (today - born).dt.years

        num_valid_entries = 175
        self.raw_data = self.raw_data[0:num_valid_entries]

        # TODO: Marius validate this part, it seems very odd to me how you remove nan's
        # filter out all not SOMA patients
        temp = self.raw_data.loc[self.raw_data['SOMA            1=Y 0=N'] == 0]
        # replace all nans with 0
        temp = temp.replace(np.nan, 0)
        self.raw_data = self.raw_data.loc[self.raw_data['SOMA            1=Y 0=N'] == 1]
        self.raw_data = self.raw_data.append(temp)
        self.raw_data = self.raw_data.dropna()

        self.raw_data['Gender'] = self.raw_data['Gender'].apply(lambda x: str(x).lower().strip())
        self.raw_data['Primary Tumour'] = self.raw_data['Primary Tumour'].apply(lambda x: str(x).lower().strip())
        self.raw_data['First Met Organ Site'] = self.raw_data['First Met Organ Site'].apply(lambda x: str(x).lower().strip())

        # TODO: Marius: I think with your implementation you don't drop the original column, please check what is valid
        # one hot encode categorical columns and drop the original column
        self.raw_data = pd.concat([self.raw_data, pd.get_dummies(self.raw_data['Primary Tumour'], prefix ='Primary Tumour')], axis=1)
        self.raw_data = pd.concat([self.raw_data, pd.get_dummies(self.raw_data['Gender'], prefix ='Gender')], axis=1)
        self.raw_data = pd.concat([self.raw_data, pd.get_dummies(self.raw_data['First Met Organ Site'], prefix='First Met Organ Site')], axis=1)
        LOGGER.info("After one-hot-encoding following columns are present {}".format(self.raw_data.columns.values))

        self.raw_data['age'] = self.raw_data['DoB'].apply(calculate_age)

        # ahhhh there we go with dropping the columns - why do we drop gender_m?
        self.raw_data = self.raw_data.drop(columns=['DoB', 'Gender', 'Gender_m', 'Primary Tumour', 'First Met Organ Site'])

        # Remove rows with NaN values
        self.raw_data = self.raw_data.dropna()
        LOGGER.info('Dataframe after preprocessing has shape {}. {} Patients were removed from dataset.'.format(self.raw_data.shape, (num_valid_entries-self.raw_data.shape[0])))

        file_name = "preprocessed_data_{}_target.csv".format(self.target)
        directory = "input"
        df_to_csv(self.raw_data, directory, file_name)


    def _split_data(self):
        X_ = self.raw_data.drop([self.target], axis=1)
        y_ = self.raw_data[self.target]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X_, y_, test_size=0.30, random_state=self.seed)
        #self.x_train = self.x_train.astype('float').reset_index(drop=True)
        #self.y_train = self.y_train.astype('float').reset_index(drop=True)
        print('after split shape, x',self.x_train.shape)
        print('after split cols', self.x_train.columns)
        print('after split shape, y', self.y_train.shape)
        print('after split types', self.x_train.dtypes.value_counts())
        print(self.y_train.dtypes)

    def _SMOTE(self):
        # Oversampling - SMOTE - Synthetic Minority Over-sampling Technique

        smote = SMOTE(k_neighbors=3, random_state=self.seed) #sampling_strategy=0.8
        self.X_train_smote, self.y_train_smote = smote.fit_sample(self.x_train, self.y_train)
        print('X_train_SMOTE:\n', self.X_train_smote[1])

        self.x_train = pd.DataFrame(self.X_train_smote, columns = self.x_train.columns)
        self.y_train = pd.DataFrame(self.y_train_smote, columns = ['Local Relapse Y(1) /N(0)'])

        print('len smote: \n', len(self.X_train_smote))
        print('len new x_train: \n', len(self.x_train))

        number_pos_x = self.y_train.loc[self.y_train['Local Relapse Y(1) /N(0)'] == 1]
        print('number positive responses y_train:\n', len(number_pos_x))


    def _SMOTE_Boarder(self):
        # Oversampling - SMOTE - Synthetic Minority Over-sampling Technique

        print('before SMOTE df', self.x_train.shape)
        smote = BorderlineSMOTE(k_neighbors=5,m_neighbors=5, random_state=self.seed) #sampling_strategy=0.8
        self.X_train_smote, self.y_train_smote = smote.fit_sample(self.x_train, self.y_train)
        print('X_train_SMOTE:\n', self.X_train_smote[1])

        self.x_train = pd.DataFrame(self.X_train_smote, columns = self.x_train.columns)
        self.y_train = pd.DataFrame(self.y_train_smote, columns = ['Local Relapse Y(1) /N(0)'])

        print('len smote: \n', len(self.X_train_smote))
        print('len new x_train: \n', len(self.x_train))

        number_pos_x = self.y_train.loc[self.y_train['Local Relapse Y(1) /N(0)'] == 1]
        print('number positive responses y_train:\n', len(number_pos_x))


    def _SMOTE_SVM(self):
        # Oversampling - SMOTE - Synthetic Minority Over-sampling Technique
        #print('before SMOTE df', self.x_train)
        print('before SMOTE df', self.x_train.shape)
        smote = SVMSMOTE(k_neighbors=5,m_neighbors=5, random_state=self.seed) #sampling_strategy=0.8
        self.X_train_smote, self.y_train_smote = smote.fit_sample(self.x_train, self.y_train)
        print('X_train_SMOTE:\n', self.X_train_smote[1])

        self.x_train = pd.DataFrame(self.X_train_smote, columns = self.x_train.columns)
        self.y_train = pd.DataFrame(self.y_train_smote, columns = ['Local Relapse Y(1) /N(0)'])

        #print('len smote: \n', len(self.X_train_smote))
        print('len new x_train after smote: \n', len(self.x_train))

        number_pos_x = self.y_train.loc[self.y_train['Local Relapse Y(1) /N(0)'] == 1]
        print('number positive responses y_train:\n', len(number_pos_x))



    def _SMOTETomek(self):
        '''Tomek links can be used as an under-sampling method or as a data cleaning method.
        Tomek links to the over-sampled training set as a data cleaning method.
        Thus, instead of removing only the majority class examples that form Tomek links, examples from both classes are removed'''
        smt = SMOTETomek(random_state=self.seed)
        self.X_train_smote, self.y_train_smote = smt.fit_sample(self.x_train, self.y_train)
        print('X_train_SMOTE:\n',self.X_train_smote[1])

        self.x_train = pd.DataFrame(self.X_train_smote, columns=self.x_train.columns)
        self.y_train = pd.DataFrame(self.y_train_smote, columns=['Local Relapse Y(1) /N(0)'])

        print('len smote: \n', len(self.X_train_smote))
        print('len new x_train: \n', len(self.x_train))

        number_pos_x = self.y_train.loc[self.y_train['Local Relapse Y(1) /N(0)'] == 1]
        print('number positive responses y_train:\n', len(number_pos_x))

    def _ANASYN(self):
        '''ADAptive SYNthetic (ADASYN) is based on the idea of
        adaptively generating minority data samples according to their distributions using K nearest neighbor.
        The algorithm adaptively updates the distribution and
        there are no assumptions made for the underlying distribution of the data.'''
        print('before: ', len(self.x_train))
        resampler = uns.InstanceHardnessThreshold(sampling_strategy=0.2,random_state=self.seed)
        self.X_train_smote2, self.y_train_smote2 = resampler.fit_resample(self.x_train, self.y_train)
        self.x_train = pd.DataFrame(self.X_train_smote2, columns=self.x_train.columns)
        self.y_train = pd.DataFrame(self.y_train_smote2, columns=['Local Relapse Y(1) /N(0)'])
        print('after: ', len(self.x_train))

        adasyn = ADASYN(random_state=self.seed)
        self.X_train_smote, self.y_train_smote = adasyn.fit_sample(self.x_train, self.y_train)
        print('X_train_SMOTE:\n', self.X_train_smote[1])

        self.x_train = pd.DataFrame(self.X_train_smote, columns=self.x_train.columns)
        self.y_train = pd.DataFrame(self.y_train_smote, columns=['Local Relapse Y(1) /N(0)'])

        print('len smote: \n', len(self.X_train_smote))
        print('len new x_train: \n', len(self.x_train))

        number_pos_x = self.y_train.loc[self.y_train['Local Relapse Y(1) /N(0)'] == 1]
        print('number positive responses y_train:\n', len(number_pos_x))



    def select_KBest(self, score_func=f_regression, k = 10  ):

        print('Pre feature selection shape X_Train', self.x_train.shape)
        print('Pre feature selection shape Y_Train',self.y_train.shape)
        feat_selector = SelectKBest(score_func=score_func, k=k)
        selector = feat_selector.fit(np.asarray(self.x_train), np.asarray(self.y_train.values))


        #feat_scores = pd.DataFrame()
        self.feat_scores["Score"] = selector.scores_
        self.feat_scores["Pvalue"] = selector.pvalues_
        self.feat_scores["Support"] = selector.get_support()
        self.feat_scores["Attribute"] = self.x_train.columns

        self.feat_scores = self.feat_scores.sort_values('Score', ascending=False, axis=0)
        #print(sorted_df)
        self.feat_scores = self.feat_scores.iloc[:k, :]  # get selected number of rows from ranking
        sorted_columns = self.feat_scores['Attribute'].values  # get column names

        print("Ranked input features:\n", self.feat_scores)

        #self.new_df = self.df[sorted_columns] #create new DF with selected rows - created in def __init__()
        #self.new_df[target] = self.df[target].values # add 'Response' column to new dataframe
        #print("New DataFrame Columns:\n", self.new_df.columns)
        #print('Length of new_df: \n', len(self.new_df))
        #print(self.new_df.dtypes)


        self.x_train = self.x_train[sorted_columns]
        self.x_test = self.x_test[sorted_columns]

        print("New Columns:\n", self.x_train.columns)
        #print("New x_test Columns:\n", self.x_train.columns)

        print('Length of x_train: \n', len(self.x_train))
        print('Length of x_test: \n', len(self.x_test))




    def boruta_selection(self):
        '''    Kursa, M., Rudnicki, W., “Feature, Selection
    with the Boruta Package” Journal of Statistical Software, Vol.36, Issue 11, Sep 2010'''


        # define random forest classifier, with utilising all cores and
        # sampling in proportion to y labels
        rf = RandomForestRegressor(n_jobs=-1, oob_score=True)

        feat_selector = BorutaPy(rf, n_estimators='auto',max_iter=100, alpha=0.05 ,verbose=2, random_state=self.seed)

        # find all relevant features - 5 features should be selected
        print(self.x_train.head())
        print(self.y_train.head())
        selector = feat_selector.fit(np.asarray(self.x_train), np.asarray(self.y_train))
        print('selector', selector)
        print(selector.support_)
        print(selector.ranking_)
        self.x_train = self.x_train.loc[:, feat_selector.support_].astype('float')
        self.x_test = self.x_test.loc[:, feat_selector.support_].astype('float')

       # self.features[target] = self.y_train['Response']
        print('New DF Shape: ', self.x_train.shape)
        #print(self.x_train)
       # self.x_train = self.features
        print("New DataFrame Columns:\n", self.x_train.columns)
        #print('Length of new_df: \n', len(self.x_train))
        print(self.x_train.dtypes)


    def _ADABoost(self):

        boost = AdaBoostRegressor(base_estimator=None, learning_rate=0.05, loss='square',
                                  n_estimators=50, random_state=self.seed)

        boost.fit(self.x_train, self.y_train)

        # Use the forest's predict method on the test data
        predictions = boost.predict(self.x_test)
        # Calculate the absolute errors
        errors = abs(predictions - self.y_test)
        # Print out the mean absolute error (mae)
        print('Mean Absolute Error:', round(np.mean(errors), 2))


    def _GSCV_REG(self):

        for i in range(5):
            self.seed = i

            '''Train-Test Split'''
            X_ = self.raw_data.drop([self.target], axis=1)
            y_ = self.raw_data[self.target]
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X_, y_, test_size=0.30,
                                                                                    random_state=self.seed)
            # self.x_train = self.x_train.astype('float').reset_index(drop=True)
            # self.y_train = self.y_train.astype('float').reset_index(drop=True)
            print('after split shape, x', self.x_train.shape)
            print('after split cols', self.x_train.columns)
            print('after split shape, y', self.y_train.shape)
            print('after split types', self.x_train.dtypes.value_counts())
            print(self.y_train.dtypes)

            '''Scaling Features'''
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


            selection = [PCA(n_components=3),
                         PCA(n_components=5),
                         RFE(SVR(kernel='linear'), 9, step=1)]
            models = [
            ('DTR', DecisionTreeRegressor()),
            #('SymbolicRegressor',SymbolicRegressor()),
            #('RandomForestRegressor', RandomForestRegressor()),
            #('AdaBoostRegressor', AdaBoostRegressor()),
            #('MLPRegressor', MLPRegressor()),
            #('Lasso', Lasso()),
            #('XGBRegressor', XGBRegressor()),
            #('SVR', SVR()),
            #('KNeighborsRegressor', KNeighborsRegressor()),
            #('ElasticNet', ElasticNet()),
            #('LinearRegression', LinearRegression()),



                 ]
            params = [
                # DecisionTreeRegressor
                {
                    'DTR__criterion': ['mse', 'mae'],
                    'DTR__max_depth': [3, 4, 6],
                    'DTR__max_leaf_nodes': [3, 5, 9, 15],
                    'DTR__min_impurity_split': [0,1e-07, 1e-08, 1e-06],
                    'DTR__min_samples_leaf': [2, 3],
                    'DTR__min_samples_split': [3, 5, 8],
                    'DTR__random_state': [self.seed],
                    #'DTR__ccp_alpha': [0.005, 0.015]
                },
                #SymbolicRegressor
                {
                    'SymbolicRegressor__population_size' : [500,200],
                    'SymbolicRegressor__generations' : [50],
                    'SymbolicRegressor__stopping_criteria' : [0.01],
                    'SymbolicRegressor__p_crossover' : [0.7],
                    'SymbolicRegressor__p_subtree_mutation' : [0.1],
                    'SymbolicRegressor__p_hoist_mutation' : [0.05],
                    'SymbolicRegressor__p_point_mutation' : [0.1],
                    'SymbolicRegressor__max_samples' : [0.9],
                    'SymbolicRegressor__verbose' : [0],
                    'SymbolicRegressor__parsimony_coefficient' : [0.01],
                    'SymbolicRegressor__random_state' : [0]
                },
                ##'RandomForestRegressor'
                 {
                    'RandomForestRegressor__bootstrap': [True],
                    'RandomForestRegressor__max_depth': [80, 90, 100, 110],
                    'RandomForestRegressor__max_features': [2, 3],
                    'RandomForestRegressor__min_samples_leaf': [3, 4, 5],
                    'RandomForestRegressor__min_samples_split': [8, 10, 12],
                    'RandomForestRegressor__n_estimators': [100, 200, 300, 1000],
                    'RandomForestRegressor__random_state': [self.seed]
                 },
                #'AdaBoostRegressor':
                 {
                    'AdaBoostRegressor__n_estimators': [20,50, 100],
                    'AdaBoostRegressor__learning_rate': [0.01, 0.05, 0.1, 0.3, 1],
                    'AdaBoostRegressor__loss': ['linear', 'square', 'exponential'],
                    'AdaBoostRegressor__random_state': [self.seed]
                 },
                #MLP Regressor
                {
                    'MLPRegressor__hidden_layer_sizes': [(8,),(10,),(10,10),(15,),(7,7),(10,10,5),(10,10,5,5)],
                    'MLPRegressor__alpha': [0.01, 0.05, 0.1, 0.3, 1],
                    'MLPRegressor__activation': ['tanh', 'relu', 'logistic'],
                    'MLPRegressor__random_state': [self.seed]
                },
                #Lasso Regression
                {
                    'Lasso__alpha': [0.3, 0.5,0.8,1, 1.2, 1.5],
                    'Lasso__selection': ['cyclic', 'random'],
                    'Lasso__random_state': [self.seed]
                },
                #'XGBRegressor':
                {
                    'XGBRegressor__min_child_weight': [1, 0.8, 1.2],
                    'XGBRegressor__learning_rate': [0.1, 0.05, 0.02, 0.3, 0.4],
                    'XGBRegressor__max_depth': [6, 4, 8, 3],
                    #'XGBRegressor__min_samples_leaf': [3, 5, 9, 17],
                    'XGBRegressor__reg_lambda': [1.0, 0.5, 1.5],
                    'XGBRegressor__reg_alpha': [1.0, 0.5, 1.5],
                    'XGBRegressor__n_jobs': [6],
                    #'XGBRegressor__eval_metric': ['rmse','mae'],
                    'XGBRegressor__scale_pos_weight': [1.0, 0.5, 1.5],
                    'XGBRegressor__random_state': [self.seed]
                },
                #SVR
                {
                    'SVR__kernel' :['rbf','linear','poly'],
                    'SVR__C' :[50,100,200],
                    'SVR__gamma' : [0.1,'auto',0.2],
                    'SVR__epsilon' :[.1]
                },

                #KNR
                {"KNeighborsRegressor__weights": ['uniform', 'distance'],
                 # "KNeighborsRegressor__random_state": [self.seed],
                 "KNeighborsRegressor__n_neighbors": [5, 2, 3, 7],
                 "KNeighborsRegressor__algorithm": ['auto', 'ball_tree', 'kd_tree'],
                 #"KNeighborsRegressor__p": [1,2],
                 },


                ]



            for s in range(len(selection)):


                for m in range(len(models)):
                    model = Pipeline([
                        ('sampling', selection[s]),
                        #('PCA', PCA()),
                        (models[m])
                    ])


                    gscv = GridSearchCV(estimator=model, param_grid=params[m], cv=3, scoring=make_scorer(mean_squared_error),n_jobs=6)
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
                        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X_, y_, test_size=0.30,
                                                                                                random_state=self.seed)

                        num_cols = self.x_train.columns
                        scaler = MinMaxScaler()

                        self.x_train[num_cols] = scaler.fit_transform(self.x_train[num_cols])
                        print(len(self.x_train))

                        self.x_test[num_cols] = scaler.fit_transform(self.x_test[num_cols])
                        print(len(self.x_test))

                        best_estimator = gscv.best_estimator_

                        best_estimator.fit(self.x_train, self.y_train)

                        y_pred_ = best_estimator.predict(self.x_test)

                        print(gscv.best_score_)

                        results_string = str([self.seed, str(gscv.best_score_), str(mean_absolute_error(self.y_test, y_pred_))
                                                 ,str(math.sqrt(mean_squared_error(self.y_test, y_pred_))), str(pearsonr(self.y_test, y_pred_)),
                                              str(gscv.best_estimator_),str(gscv.best_params_), # die letzten 2 überprüfen was die genau bedeuten, MSE und RMSE ergänzen
                                              str(selection[s]), str(gscv.best_params_)]) + '\n'
                        print(results_string)
                        with open(self.results_filepath, 'a')as f:
                            f.write(results_string)

                        aggregate_results.append(
                            [self.seed, round(gscv.best_score_, 3), mean_absolute_error(self.y_test, y_pred_), math.sqrt(mean_squared_error(self.y_test, y_pred_)),pearsonr(self.y_test, y_pred_)[0],pearsonr(self.y_test, y_pred_)[1]])

                        print(aggregate_results)
                        median_results.append([round(mean_absolute_error(self.y_test, y_pred_), 3)])

                median = np.median(median_results)
                aggregate_results_mean = np.around(np.mean(aggregate_results, axis=0), decimals=3)

                # print(aggregate_results_mean)
                results_str = str(
                    [aggregate_results_mean[0], aggregate_results_mean[1], aggregate_results_mean[2], median,
                     aggregate_results_mean[3], aggregate_results_mean[4], aggregate_results_mean[5],
                     str(selection[s]), str(gscv.best_estimator_),str(gscv.best_params_)]) + '\n'
                with open(
                        'C:\\Users\\mariu\\OneDrive\\Dokumente\\NOVA\\Thesis\\20191015_new_data\\Final Results\\AVG\\DTR\\_results_' + str(
                                self.target) + '.csv', 'a')as f:
                    f.write(results_str)


    def _algorithm_opt(self):

        for i in range(5):
            self.seed  = i

            '''Train-Test Split'''
            X_ = self.raw_data.drop([self.target], axis=1)
            y_ = self.raw_data[self.target]
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X_, y_, test_size=0.30,
                                                                                    random_state=self.seed)
            # self.x_train = self.x_train.astype('float').reset_index(drop=True)
            # self.y_train = self.y_train.astype('float').reset_index(drop=True)
            print('after split shape, x', self.x_train.shape)
            print('after split cols', self.x_train.columns)
            print('after split shape, y', self.y_train.shape)
            print('after split types', self.x_train.dtypes.value_counts())
            print(self.y_train.dtypes)

            num_cols = self.x_train.columns  # .select_dtypes(include=['float']).columns
            scaler = MinMaxScaler()

            self.x_train[num_cols] = scaler.fit_transform(self.x_train[num_cols])
            print(len(self.x_train))

            self.x_test[num_cols] = scaler.fit_transform(self.x_test[num_cols])
            print(len(self.x_test))


            oversamplings = [SVMSMOTE(random_state=self.seed), SMOTETomek(random_state=self.seed),SMOTE(random_state=self.seed), BorderlineSMOTE(random_state=self.seed)]#, ADASYN(random_state=self.seed)]
            models = [#('SGD',SGDClassifier(random_state=self.seed)),
                      #('DTC', DecisionTreeClassifier()),
                      #('SVC',SVC()),
                      #('MLP',MLPClassifier()),
                      #('ABC',AdaBoostClassifier()),
                      #('XGB', XGBClassifier()),
                      #('RandomForestClassifier', RandomForestClassifier()),
                      #('KNC',KNeighborsClassifier()),
                      #('SymbolicClassifier', SymbolicClassifier()),
                      #('LR', LogisticRegression())


                       ]

            parameters_list = [
                #{'sampling__random_state':[self.seed],
                 #'SGD__alpha': [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0],
                 #'SGD__penalty': ['l2','l1','elasticnet'],
                 #'SGD__loss':['log','perceptron','hinge'],
                 #'SGD__n_jobs': [-1],
                 #'SGD__random_state': [self.seed]},

                #{'sampling__random_state':[self.seed],
                 #'DTC__class_weight':[None],
                 #'DTC__criterion':['gini', 'entropy'],
                 #'DTC__max_depth':[3,4,6],
                 #'DTC__max_features':[,None],
                 #'DTC__max_leaf_nodes':[2,3,5,9,15],
                 #'DTC__min_impurity_split':[1e-07, 1e-08, 1e-06],
                 #'DTC__min_samples_leaf':[2,3,3],
                 #'DTC__min_samples_split':[3,5,8,6],
                 #'DTC__min_weight_fraction_leaf':[0.1, 0.3],
                 #'DTC__presort':[False],
                 #'DTC__random_state':[self.seed],
                 #'DTC__splitter':['best']},

                #{'sampling__random_state' : [self.seed],
                # 'SVC__C':[0.4, 0.3,0.8],
                # 'SVC__decision_function_shape':['ovr'],
                # 'SVC__degree':[3,5,7],
                # 'SVC__gamma':[0.3,0.5,0.8],
                ## 'SVC__kernel':['poly', 'rbf'],
                 #'SVC__max_iter':[-1],
                 #'SVC__probability':[True],
                 #'SVC__random_state':[self.seed],
                 #'SVC__tol':[0.001, 0.0005, 0.002],
                 #'SVC__verbose':[False]},

                #{'sampling__random_state' : [self.seed],
                #'MLP__solver': ['adam', 'sgd', 'lbfgs'],
                #'MLP__learning_rate': ['constant', 'invscaling', 'adaptive'],
                #'MLP__activation': ['logistic', 'tanh', 'relu'],
                #'MLP__hidden_layer_sizes' : [(10,10),(10,5),(15,10,10,5,5),(10,10,5,5)
                ##    #(10,10,5,5,5)
                #     ],
                # 'MLP__random_state' : [self.seed]},

                #{"ABC__learning_rate": [1,0.8,0.6,1.2],
                # "ABC__random_state": [self.seed],
                # "ABC__n_estimators": [50, 20,80,60,40]
                # },

                #{"XGB__learning_rate": [1,0.8,0.6,1.2],
                # "XGB__random_state": [self.seed],
                # "XGB__n_estimators": [50, 20,100,200],
                # "XGB__max_depth": [5, 3, 10],
                # "XGB__gamma": [0, 0.1, 0.2],
                #  "XGB__subsample": [0.8, 0.5, 0.9],
                #  "XGB__min_child_weight": [1,2, 5],
                # },
                #{
                #    'RandomForestClassifier__bootstrap': [True],
                #    'RandomForestClassifier__max_depth': [2, 3, 4, 6],
                #    #'RandomForestClassifier__max_features': [2, 3],
                #    'RandomForestClassifier__min_samples_leaf': [3,4,6],
                 #   'RandomForestClassifier__min_samples_split': [2, 3, 5, 8],
                #    'RandomForestClassifier__n_estimators': [30, 80, 200, 300],
                #    'RandomForestClassifier__random_state': [self.seed]
                #},
                 #{"KNC__weights": ['distance'], #'uniform',
                 #"KNC__random_state": [self.seed],
                 # "KNC__n_neighbors": [5,2,3,7],
                 # "KNC__algorithm": ['auto', 'ball_tree','kd_tree'],
                  #"KNC__n_jobs": [5],
                 #   },
           # {
           #     'SymbolicClassifier__population_size': [100,50],
           #     'SymbolicClassifier__generations': [50,100],
           #     'SymbolicClassifier__stopping_criteria': [0.01,0.1,0.001],
           #     'SymbolicClassifier__p_crossover': [0.5,0.2],
           #     'SymbolicClassifier__p_subtree_mutation': [0.10,0.2],
           #     'SymbolicClassifier__p_hoist_mutation': [0.05,0.1],
           #     'SymbolicClassifier__p_point_mutation': [0.1,0.2],
           #     'SymbolicClassifier__max_samples': [0.9],
           #     'SymbolicClassifier__verbose': [1],
           #     'SymbolicClassifier__parsimony_coefficient': [0.01,0.1],
           #     'SymbolicClassifier__random_state': [self.seed]
           # },
                 #{'LR__penalty': ['l1', 'l2'],
                 #'LR__random_state' : [self.seed],
                 #'LR__solver' : ['liblinear','saga'],
                 #'LR__multi_class' : ['ovr', 'auto'],
                 #'LR__C': [0.5, 0.8,0.7,0.9],
                 #'LR__max_iter':[80,100,200]
                 # },

                {'sampling__random_state': [self.seed],
                 'DTR__criterion': ['mse', 'mae'],
                 'DTR__max_depth': [3, 4, 6],
                 'DTR__max_leaf_nodes': [2, 3, 5, 9, 15],
                 'DTR__min_impurity_split': [0, 1e-07, 1e-08, 1e-06],
                 'DTR__min_samples_leaf': [2, 3, 3],
                 'DTR__min_samples_split': [2, 3, 5, 8, 6],
                 'DTR__ccp_alpha': [0,0.1, 0.3],
                 'DTR__random_state': [self.seed],
                 'DTR__splitter': ['best']},
                               ]
            selection = [
                ('RFE', RFE(SVR(kernel='linear'), 9, step=1)),
                ('PCA', PCA(n_components=9)),
                ('PCA', PCA(n_components=5))
            ]

            for s in range(len(selection)):

                for o in range(len(oversamplings)):

                    for m in range(len(models)):
                        model = Pipeline([
                            ('sampling', oversamplings[o]),
                            selection[s],
                            models[m]
                        ])


                        gscv = GridSearchCV(estimator=model, param_grid=parameters_list[m], cv=5, scoring='f1_macro', n_jobs=6)
                        gscv.fit(self.x_train, self.y_train)
                        results_df = pd.DataFrame(gscv.cv_results_)
                        results_df = results_df.loc[results_df['rank_test_score']==1]
                        print(results_df)
                        print("best estimator is: {}".format(gscv.best_estimator_))
                        print("best score are: {}".format(gscv.best_score_))
                        print("best parameters are: {}".format(gscv.best_params_))
                        # self.best_classifier[name] = gscv

                        aggregate_results = list()
                        median_results = list()

                        for i in range(5):
                            self.seed = i
                            X_ = self.raw_data.drop([self.target], axis=1)
                            y_ = self.raw_data[self.target]
                            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X_, y_, test_size=0.30,
                                                                                                    random_state=self.seed)

                            num_cols = self.x_train.columns
                            scaler = MinMaxScaler()

                            self.x_train[num_cols] = scaler.fit_transform(self.x_train[num_cols])
                            print(len(self.x_train))

                            self.x_test[num_cols] = scaler.fit_transform(self.x_test[num_cols])
                            print(len(self.x_test))

                            best_estimator = gscv.best_estimator_

                            best_estimator.fit(self.x_train, self.y_train)

                            y_pred_ = best_estimator.predict(self.x_test)

                            #  print('The accuracy score of Random Forest is : %s ' % (round(forest.score(self.y_test_, y_pred_), 3)))

                            print('ConfMatrix: \n', confusion_matrix(self.y_test, y_pred_))
                            # print(classification_report(y_test, y_pred))
                            print('Accuracy Score :\n', accuracy_score(self.y_test, y_pred_))

                            prec = precision_score(self.y_test, y_pred_, average='micro')
                            rec = recall_score(self.y_test, y_pred_,average='micro')
                            print('Precision is %s and Recall is %s' % (round(prec, 3), round(rec, 3)))

                            self.cm = confusion_matrix(self.y_test, y_pred_)

                            #Positive_repsonses = self.y_test.loc[self.y_test == 1].count()

                            y_score = best_estimator.fit(self.x_train,self.y_train).predict(self.x_test)

                            fpr = dict()
                            tpr = dict()
                            roc_auc = dict()

                            fpr, tpr, _ = roc_curve(self.y_test, y_score,pos_label=1)
                            roc_auc = auc(fpr, tpr)

                            plt.figure()
                            lw = 1
                            plt.plot(fpr, tpr, color='darkorange',
                                     lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
                            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
                            plt.xlim([0.0, 1.0])
                            plt.ylim([0.0, 1.05])
                            plt.xlabel('False Positive Rate')
                            plt.ylabel('True Positive Rate')
                            plt.title('Receiver operating characteristic example')
                            plt.legend(loc="lower right")
                            plt.savefig('C:\\Users\\mariu\\OneDrive\\Dokumente\\NOVA\\Thesis\\20191015_new_data\\Final Results\\Graphs\\_results_' + str(self.target) +'_'+ str(models[m])[:10]  +'_'+ str(selection[s])[:10] +'_'+ str(oversamplings[o])[:10]  +'_'+ str(self.seed)+'.png')

                            results_string = str(
                                [self.seed, round(gscv.best_score_,3), confusion_matrix(self.y_test, y_pred_)[0][0],confusion_matrix(self.y_test, y_pred_)[1][0],confusion_matrix(self.y_test, y_pred_)[0][1],confusion_matrix(self.y_test, y_pred_)[1][1], round(accuracy_score(self.y_test, y_pred_),3),
                                 round(prec, 3), round(rec, 3), round(f1_score(self.y_test, y_pred_, average='micro'),3)#,round(roc_auc_score(self.y_test, y_pred_),3)
                                    ,str(self.scoring_metric),str(gscv.best_estimator_),
                                 str(selection[s]), str(oversamplings[o]),str(gscv.best_params_)]) + '\n'
                            with open(self.results_filepath, 'a')as f:
                                f.write(results_string)

                            aggregate_results.append([self.seed, round(gscv.best_score_,3), round(accuracy_score(self.y_test, y_pred_),3), confusion_matrix(self.y_test, y_pred_)[0][0],confusion_matrix(self.y_test, y_pred_)[1][0],confusion_matrix(self.y_test, y_pred_)[0][1],
                                                      confusion_matrix(self.y_test, y_pred_)[1][1], round(prec, 3), round(rec, 3),
                                                      round(f1_score(self.y_test, y_pred_, average='micro'),3)#,round(roc_auc_score(self.y_test, y_pred_),3)
                                                       ])
                            median_results.append([round(accuracy_score(self.y_test, y_pred_),3)])


                        #print(aggregate_results)
                        median = np.median(median_results)
                        aggregate_results_mean = np.around(np.mean(aggregate_results,axis = 0),decimals=3)

                        #print(aggregate_results_mean)
                        results_str = str([aggregate_results_mean[0],aggregate_results_mean[1],aggregate_results_mean[2],median, aggregate_results_mean[3],aggregate_results_mean[4],aggregate_results_mean[5],
                                           aggregate_results_mean[6],aggregate_results_mean[7],aggregate_results_mean[8],aggregate_results_mean[9],#aggregate_results_mean[10],
                                          str(selection[s]), str(oversamplings[o]),str(gscv.best_estimator_)])+ '\n'
                        with open('C:\\Users\\mariu\\OneDrive\\Dokumente\\NOVA\\Thesis\\20191015_new_data\\Final Results\\AVG\\DTR\\_results_' + str(self.target) +'.csv', 'a')as f:
                            f.write(results_str)
