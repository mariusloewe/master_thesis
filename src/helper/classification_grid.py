"""
This file serves to create the grid-search's grid of classification tasks.
"""
# project imports
from src.helper.settings import SEED
from src.helper.utils import _SMOTE, _SMOTETomek, _SMOTE_Border, _SMOTE_SVM, _RFE

# sklearn feature selection
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.svm import SVR

# sklearn models
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

# boosting algos
from xgboost import XGBClassifier

# genetic programming
from gplearn.genetic import SymbolicClassifier


SELECTION = {
    "RFE": _RFE,
    "PCA_9": PCA(n_components=9),
    "PCA_5": PCA(n_components=5),
}

OVERSAMPLING = {
    "SMOTETomek": _SMOTETomek,
    "SMOTE": _SMOTE,
    "BorderlineSMOTE": _SMOTE_Border,
    "SVMSMOTE": _SMOTE_SVM,
}

MODELS = {
    "SGD": SGDClassifier,
    "DTC": DecisionTreeClassifier,
    "SVC": SVC,
    "MLP": MLPClassifier,
    "ABC": AdaBoostClassifier,
    "XGB": XGBClassifier,
    "RandomForestClassifier": RandomForestClassifier,
    "KNC": KNeighborsClassifier,
    "SymbolicClassifier": SymbolicClassifier,
    "LR": LogisticRegression,
}

PARAMETERS = {
    "SGD": {
        "alpha": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0],
        "penalty": ["l2", "l1", "elasticnet"],
        "loss": ["log", "perceptron", "hinge"],
        "n_jobs": [-1],
        "random_state": [SEED],
    },
    "DTC": {
        "class_weight": [None],
        "criterion": ["gini", "entropy"],
        "max_depth": [2, 3, 4, 6, 8],
        "max_features": [None, 5, 10],
        "max_leaf_nodes": [2, 3, 5, 9, 15],
        "min_samples_leaf": [2, 3, 3],
        "min_samples_split": [3, 5, 8, 6],
        "min_weight_fraction_leaf": [0.1, 0.3],
        "random_state": [SEED],
        "splitter": ["best"],
    },
    "SVC": {
        "C": [0.4, 0.3, 0.8],
        "decision_function_shape": ["ovr"],
        "degree": [3, 5, 7],
        "gamma": [0.3, 0.5, 0.8],
        # 'kernel':['poly', 'rbf'],
        "max_iter": [-1],
        "probability": [True],
        "random_state": [SEED],
        "tol": [0.001, 0.0005, 0.002],
        "verbose": [False],
    },
    "MLP": {
        "solver": ["adam", "sgd", "lbfgs"],
        "learning_rate": ["constant", "invscaling", "adaptive"],
        "activation": ["logistic", "tanh", "relu"],
        "hidden_layer_sizes": [(10, 10), (10, 5), (15, 10, 10, 5, 5), (10, 10, 5, 5)],
        "random_state": [SEED],
    },
    "ABC": {
        "learning_rate": [1, 0.8, 0.6, 1.2, 0.1, 0.01],
        "random_state": [SEED],
        "n_estimators": [50, 20, 80, 60, 40],
    },
    "XGB": {
        "learning_rate": [1, 0.6, 0.4,  0.2, 0.1, 0.01],
        "random_state": [SEED],
        "n_estimators": [50, 20, 100, 200],
        "max_depth": [5, 3, 10],
        "gamma": [0, 0.1, 0.2],
        "subsample": [0.8, 0.5, 0.9],
        "min_child_weight": [1, 2, 5],
    },
    "RandomForestClassifier": {
        "bootstrap": [True],
        "max_depth": [2, 3, 4, 6],
        "max_features": [2, 3],
        "min_samples_leaf": [3, 4, 6],
        "min_samples_split": [2, 3, 5, 8],
        "n_estimators": [30, 80, 200, 300],
        "random_state": [SEED],
    },
    "KNC": {
        "weights": ["distance"],  #'uniform',
        "n_neighbors": [5, 2, 3, 7],
        "algorithm": ["auto", "ball_tree", "kd_tree"],
        "n_jobs": [5],
    },
    "SymbolicClassifier": {
        "population_size": [100, 50],
        "generations": [50, 100],
        "stopping_criteria": [0.01, 0.1, 0.001],
        "p_crossover": [0.5, 0.2],
        "p_subtree_mutation": [0.10, 0.2],
        "p_hoist_mutation": [0.05, 0.1],
        "p_point_mutation": [0.1, 0.2],
        "max_samples": [0.9],
        "verbose": [0],
        "parsimony_coefficient": [0.01, 0.1],
        "random_state": [SEED],
    },
    "LR": {
        "penalty": ["l1", "l2"],
        "random_state": [SEED],
        "solver": ["liblinear", "saga", 'newton-cg'],
        "multi_class": ["ovr", "auto"],
        "C": [0.5, 0.8, 0.7, 0.9, 0.2, 0.3],
        "max_iter": [80, 100, 200],
    },
    "DTR": {
        "criterion": ["mse", "mae"],
        "max_depth": [3, 4, 6],
        "max_leaf_nodes": [2, 3, 5, 9, 15],
        "min_impurity_split": [0, 1e-07, 1e-08, 1e-06],
        "min_samples_leaf": [2, 3, 3],
        "min_samples_split": [2, 3, 5, 8, 6],
        "ccp_alpha": [0, 0.1, 0.3],
        "random_state": [SEED],
        "splitter": ["best"],
    },
}
