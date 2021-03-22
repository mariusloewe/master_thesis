"""
This file serves to create the grid for the regressions for the grid search.
"""
# project imports
from src.settings import SEED

# sklearn feature selection
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA

# sklearn Regressors
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import (
    GradientBoostingRegressor,
    AdaBoostRegressor,
    RandomForestRegressor,
)
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Lasso, ElasticNet, LinearRegression
from sklearn.neighbors import KNeighborsRegressor

# Boosting algorithms
from xgboost import XGBRegressor

# genetic programming
from gplearn.genetic import SymbolicRegressor


MODELS = {
    "DTR": DecisionTreeRegressor,
    "SymbolicRegressor": SymbolicRegressor,
    "RandomForestRegressor": RandomForestRegressor,
    "AdaBoostRegressor": AdaBoostRegressor,
    "MLPRegressor": MLPRegressor,
    "Lasso": Lasso,
    "XGBRegressor": XGBRegressor,
    "SVR": SVR,
    "KNeighborsRegressor": KNeighborsRegressor,
    "ElasticNet": ElasticNet,
    "LinearRegression": LinearRegression,
}


SELECTION = [
    PCA(n_components=3),
    PCA(n_components=5),
    RFE(SVR(kernel="linear"), 9, step=1),
]


PARAMETERS = {
    "DTR": {
        "criterion": ["mse", "mae"],
        "max_depth": [3, 4, 6],
        "max_leaf_nodes": [3, 5, 9, 15],
        # "min_impurity_split": [0, 1e-07, 1e-08, 1e-06], deprecated
        "min_samples_leaf": [2, 3],
        "min_samples_split": [3, 5, 8],
        "random_state": [SEED],
        #'DTR__ccp_alpha': [0.005, 0.015]
    },
    "SymbolicRegressor": {
        "population_size": [500, 200],
        "generations": [50],
        "stopping_criteria": [0.01],
        "p_crossover": [0.7],
        "p_subtree_mutation": [0.1],
        "p_hoist_mutation": [0.05],
        "p_point_mutation": [0.1],
        "max_samples": [0.9],
        "verbose": [0],
        "parsimony_coefficient": [0.01],
        "random_state": [0],
    },
    "RandomForestRegressor": {
        "bootstrap": [True],
        "max_depth": [80, 90, 100, 110],
        "max_features": [2, 3],
        "min_samples_leaf": [3, 4, 5],
        "min_samples_split": [8, 10, 12],
        "n_estimators": [100, 200, 300, 1000],
        "random_state": [SEED],
    },
    "AdaBoostRegressor": {
        "n_estimators": [20, 50, 100],
        "learning_rate": [0.01, 0.05, 0.1, 0.3, 1],
        "loss": ["linear", "square", "exponential"],
        "random_state": [SEED],
    },
    "MLP Regressor": {
        "hidden_layer_sizes": [
            (8,),
            (10,),
            (10, 10),
            (15,),
            (7, 7),
            (10, 10, 5),
            (10, 10, 5, 5),
        ],
        "alpha": [0.01, 0.05, 0.1, 0.3, 1],
        "activation": ["tanh", "relu", "logistic"],
        "random_state": [SEED],
    },
    "Lasso Regression": {
        "alpha": [0.3, 0.5, 0.8, 1, 1.2, 1.5],
        "selection": ["cyclic", "random"],
        "random_state": [SEED],
    },
    "XGBRegressor": {
        "min_child_weight": [1, 0.8, 1.2],
        "learning_rate": [0.1, 0.05, 0.02, 0.3, 0.4],
        "max_depth": [6, 4, 8, 3],
        #'min_samples_leaf': [3, 5, 9, 17],
        "reg_lambda": [1.0, 0.5, 1.5],
        "reg_alpha": [1.0, 0.5, 1.5],
        "n_jobs": [6],
        #'eval_metric': ['rmse','mae'],
        "scale_pos_weight": [1.0, 0.5, 1.5],
        "random_state": [SEED],
    },
    "SVR": {
        "kernel": ["rbf", "linear", "poly"],
        "C": [50, 100, 200],
        "gamma": [0.1, "auto", 0.2],
        "epsilon": [0.1],
    },
    "KNR": {
        "weights": ["uniform", "distance"],
        # "random_state": [SEED],
        "n_neighbors": [5, 2, 3, 7],
        "algorithm": ["auto", "ball_tree", "kd_tree"]
        # "p": [1,2],
    },
}
