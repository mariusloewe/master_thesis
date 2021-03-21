# This file serves to create the grid for the grid search
# project imports
from src.settings import SEED

# imports

# sklearn
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import RFE
from sklearn.svm import SVR

MODELS = {
    "DTR": DecisionTreeRegressor,
    # ('SymbolicRegressor',SymbolicRegressor()),
    # ('RandomForestRegressor', RandomForestRegressor()),
    # ('AdaBoostRegressor', AdaBoostRegressor()),
    # ('MLPRegressor', MLPRegressor()),
    # ('Lasso', Lasso()),
    # ('XGBRegressor', XGBRegressor()),
    # ('SVR', SVR()),
    # ('KNeighborsRegressor', KNeighborsRegressor()),
    # ('ElasticNet', ElasticNet()),
    # ('LinearRegression', LinearRegression())
}


SELECTION = [
#    PCA(n_components=3),
#    PCA(n_components=5),
   RFE(SVR(kernel="linear"), 9, step=1),
]


PARAMETERS = {
    # DecisionTreeRegressor
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
    "SymbolicRegressor":
    {
        "SymbolicRegressor__population_size": [500, 200],
        "SymbolicRegressor__generations": [50],
        "SymbolicRegressor__stopping_criteria": [0.01],
        "SymbolicRegressor__p_crossover": [0.7],
        "SymbolicRegressor__p_subtree_mutation": [0.1],
        "SymbolicRegressor__p_hoist_mutation": [0.05],
        "SymbolicRegressor__p_point_mutation": [0.1],
        "SymbolicRegressor__max_samples": [0.9],
        "SymbolicRegressor__verbose": [0],
        "SymbolicRegressor__parsimony_coefficient": [0.01],
        "SymbolicRegressor__random_state": [0],
    },
    'RandomForestRegressor':
    {
        "RandomForestRegressor__bootstrap": [True],
        "RandomForestRegressor__max_depth": [80, 90, 100, 110],
        "RandomForestRegressor__max_features": [2, 3],
        "RandomForestRegressor__min_samples_leaf": [3, 4, 5],
        "RandomForestRegressor__min_samples_split": [8, 10, 12],
        "RandomForestRegressor__n_estimators": [100, 200, 300, 1000],
        "RandomForestRegressor__random_state": [SEED],
    },
    'AdaBoostRegressor':
    {
        "AdaBoostRegressor__n_estimators": [20, 50, 100],
        "AdaBoostRegressor__learning_rate": [0.01, 0.05, 0.1, 0.3, 1],
        "AdaBoostRegressor__loss": ["linear", "square", "exponential"],
        "AdaBoostRegressor__random_state": [SEED],
    },
    "MLP Regressor":
    {
        "MLPRegressor__hidden_layer_sizes": [
            (8,),
            (10,),
            (10, 10),
            (15,),
            (7, 7),
            (10, 10, 5),
            (10, 10, 5, 5),
        ],
        "MLPRegressor__alpha": [0.01, 0.05, 0.1, 0.3, 1],
        "MLPRegressor__activation": ["tanh", "relu", "logistic"],
        "MLPRegressor__random_state": [SEED],
    },
    "Lasso Regression":
    {
        "Lasso__alpha": [0.3, 0.5, 0.8, 1, 1.2, 1.5],
        "Lasso__selection": ["cyclic", "random"],
        "Lasso__random_state": [SEED],
    },
    'XGBRegressor':
    {
        "XGBRegressor__min_child_weight": [1, 0.8, 1.2],
        "XGBRegressor__learning_rate": [0.1, 0.05, 0.02, 0.3, 0.4],
        "XGBRegressor__max_depth": [6, 4, 8, 3],
        #'XGBRegressor__min_samples_leaf': [3, 5, 9, 17],
        "XGBRegressor__reg_lambda": [1.0, 0.5, 1.5],
        "XGBRegressor__reg_alpha": [1.0, 0.5, 1.5],
        "XGBRegressor__n_jobs": [6],
        #'XGBRegressor__eval_metric': ['rmse','mae'],
        "XGBRegressor__scale_pos_weight": [1.0, 0.5, 1.5],
        "XGBRegressor__random_state": [SEED],
    },
    "SVR":
    {
        "SVR__kernel": ["rbf", "linear", "poly"],
        "SVR__C": [50, 100, 200],
        "SVR__gamma": [0.1, "auto", 0.2],
        "SVR__epsilon": [0.1],
    },
    "KNR":
    {
        "KNeighborsRegressor__weights": ["uniform", "distance"],
        # "KNeighborsRegressor__random_state": [SEED],
        "KNeighborsRegressor__n_neighbors": [5, 2, 3, 7],
        "KNeighborsRegressor__algorithm": ["auto", "ball_tree", "kd_tree"]
        # "KNeighborsRegressor__p": [1,2],
    }
}
