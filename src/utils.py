# standard imports
from pathlib import PurePath
import logging

# DS imports
import pandas as pd
import numpy as np

# Third-Party imports
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import MinMaxScaler
import imblearn.under_sampling as uns

# imblearn imports
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN
from imblearn.combine import SMOTETomek, SMOTEENN

# feature selectors
from boruta import BorutaPy
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import chi2

# models
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier

### A package that carry miscellaenous utilities


def calc_correlations(
    df, target, output_filepath="results/correlations", verbose=False
):
    """
    Calculates Pearson and Spearman correlations per target and dumps it in a csv file.
    :param df: Df with data
    :param target: target variable
    :param output_filepath: output file path
    :param verbose: verbosity of function
    """
    corr_pear = []
    corr_spear = []
    for i in range(len(df.columns)):
        pear, _ = pearsonr(df[df.columns[i]], df[df.columns[0]])
        corr_pear.append(pear)
        spear, _ = spearmanr(df[df.columns[i]], df[df.columns[0]])
        corr_spear.append(spear)
    df_corr = pd.DataFrame(
        list(zip(df.columns, corr_pear, corr_spear)),
        columns=["col", "pearson_corr", "spearman_corr"],
    )
    if verbose:
        print(df_corr.sort_values("pearson_corr", ascending=False))
        print(df_corr.sort_values("spearman_corr", ascending=False))

    file_name = "_Correlations" + str(target) + ".csv"
    file_path = PurePath(output_filepath)
    df_corr.to_csv(file_path.joinpath(file_name))


def binning(df, col, bins=None, labels=None, verbose=False):
    """
    Bins a given column from a dataframe
    :param df: Datframe
    :param col: column to be binned
    :param labels: label of bins
    :param bins: bins
    :param verbose: verbosity
    :return:
    """
    if labels is None:
        labels = [0, 1, 2]
    if bins is None:
        bins = [0, 3, 10, 12]
    df[col] = pd.cut(x=df[col], bins=bins, labels=labels)
    if verbose:
        print(df["col"].value_counts())
    return df


def df_to_csv(df, directory, file_name):
    """
    Dumps a pandas DF to csv. Only works if directory is present in the current working directory.
    :param df: DF to be dumped
    :param directory: directory to dump in to
    :param file_name: file name
    """
    directory = PurePath(directory)
    df.to_csv(directory.joinpath(file_name))


def scaling_features(df, features_to_scale=None):
    """
    Scales features with MinMaxScaler of a provided df.
    :param df:
    :param features_to_scale:
    :return:
    """
    num_cols = df.columns if features_to_scale is None else features_to_scale

    # .select_dtypes(include=['float']).columns
    # num_cols = self.x_train.columns[self.x_train.dtypes.apply(lambda c: np.issubdtype(c, np.number))]
    # print('Numerical Columns: \n', num_cols)

    # Only scale continuous values
    scaler = MinMaxScaler()
    # if Bucketing - every feature is dtype object - so no need to specify to scale only numerical features

    df[num_cols] = scaler.fit_transform(df[num_cols])
    logging.info("Scaled {} columns.".format(num_cols))

    return df


# Sampling Methods
# TODO: Finish refactoring sampling methods
def _SMOTE(seed, X, y, target, k_neighbors=3):
    """
    Oversampling - SMOTE - Synthetic Minority Over-sampling Technique
    :param seed:
    :param X:
    :param y:
    :param target:
    :param k_neighbors:
    :return:
    """

    # TODO: Marius: Should the original frame be concatenated with the new frames?
    smote = SMOTE(k_neighbors=k_neighbors, random_state=seed) #sampling_strategy=0.8
    X_smote, y_smote = smote.fit_sample(X, y)

    X_smote = pd.DataFrame(X_smote, columns=X.columns)
    y_smote = pd.DataFrame(y_smote, columns=[target])

    pos_before = y.loc[y[target] == 1].shape[0]
    pos_after = y_smote.loc[y_smote[target] == 1].shape[0]

    logging.info("Observations before applying SMOTE: {} and after applying SMOTE {}".format(X.shape[0], X_smote.shape[0]))
    logging.info("Number positives before/after SMOTE: {}/{}.".format(pos_before, pos_after))

    return X_smote, y_smote


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


### Feature Selectors

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