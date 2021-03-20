# standard imports
from pathlib import PurePath

# DS imports
import pandas as pd

# Third-Party imports
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import MinMaxScaler

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
    LOGGER.info("Scaled {} columns.".format(num_cols))

    return df

