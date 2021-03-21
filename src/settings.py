# this file serves as a dump for global constants

# Global SEED for search
SEED = 0

# lists of target variables
TARGETS_REGRESSION = ["OS months", "PMFS Time to endpoint  PM or no PM"]
TARGETS_MULTICLASS = ["If OM_number of lesions at last FU"]
TARGETS_BINARY = [
    "PM definition  ≥4 lesions",
    "PMFS Oligo Status (≤5) mantained until  last FU 0=Y",
    "≤10",
    "By feasibility",
]


# the features used in the project - by commenting them in and out you can set which ones to use
# TODO: Marius, please double check this list is correct
FEATURE_LIST = [
    "DoB",
    "Gender",
    "Primary Tumour",
    "First Met Organ Site",
    "CTV cc",
    "SUVmax Baseline PET-CT",
    # 'Local Relapse Y(1) /N(0)',
    # 'LRFS Months',
    # 'Progression Elsewhere(Y:1 / N: 0)',
    "Same organ (0:Y 1:N 2:Both)",
    "Systemic Tx (0 =no or pre, 1= combination with during or post)",
    "First lesion(s) SDRT Only (1=Y)",
    "SOMA            1=Y 0=N",
    "Number of targets at 1st Tx",
    "Overall Regimen",
    "N LRR",
    "Patient with LF with rescue",
    "Tumour Burden 1st Tx cc",
    "Tumour burden         I SOMA",
    "Largest single OM burden",
    #'SOMA > 1st OM  0=N 1=Y',
    "DFS months between repeat Tx",
    "Highest SUVmax at 1st Tx",
    "N. of  target organs at 1st Tx"
    # 'N of targets    I SOMA',
    # 'Interval between ablations',
    # 'Δ Tumour burden	Min Burden',
    # 'Max Burden',
    # 'Average burden',
    # 'Min %Δ Tumour burden',
    # 'Max %Δ Tumour burden',
    # 'Mean %Δ Tumour burden',
    # 'Min N',
    # 'Max N',
    # 'Min SOMA interval',
    # 'Max SOMA Interval',
    # 'Average SOMA Interval',
    # 'Cumulative Tumour burden',
    # 'Largest single SOMA burden',
    # 'Highest SUVmax ever',
    # 'N. of  target organs involved in total',
    # 'Repeat TX',
    # 'Total number of targets',
    # 'Total SOMA lesions'
]

SELECTION = [
#    PCA(n_components=3),
#    PCA(n_components=5),
#    RFE(SVR(kernel="linear"), 9, step=1),
]

MODELS = [
#    ("DTR", DecisionTreeRegressor()),
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
]

PARAMETERS = [
    # DecisionTreeRegressor
    {
        "DTR__criterion": ["mse", "mae"],
        "DTR__max_depth": [3, 4, 6],
        "DTR__max_leaf_nodes": [3, 5, 9, 15],
        "DTR__min_impurity_split": [0, 1e-07, 1e-08, 1e-06],
        "DTR__min_samples_leaf": [2, 3],
        "DTR__min_samples_split": [3, 5, 8],
        "DTR__random_state": [SEED],
        #'DTR__ccp_alpha': [0.005, 0.015]
    },
    # SymbolicRegressor
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
    ##'RandomForestRegressor'
    {
        "RandomForestRegressor__bootstrap": [True],
        "RandomForestRegressor__max_depth": [80, 90, 100, 110],
        "RandomForestRegressor__max_features": [2, 3],
        "RandomForestRegressor__min_samples_leaf": [3, 4, 5],
        "RandomForestRegressor__min_samples_split": [8, 10, 12],
        "RandomForestRegressor__n_estimators": [100, 200, 300, 1000],
        "RandomForestRegressor__random_state": [SEED],
    },
    #'AdaBoostRegressor':
    {
        "AdaBoostRegressor__n_estimators": [20, 50, 100],
        "AdaBoostRegressor__learning_rate": [0.01, 0.05, 0.1, 0.3, 1],
        "AdaBoostRegressor__loss": ["linear", "square", "exponential"],
        "AdaBoostRegressor__random_state": [SEED],
    },
    # MLP Regressor
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
    # Lasso Regression
    {
        "Lasso__alpha": [0.3, 0.5, 0.8, 1, 1.2, 1.5],
        "Lasso__selection": ["cyclic", "random"],
        "Lasso__random_state": [SEED],
    },
    #'XGBRegressor':
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
    # SVR
    {
        "SVR__kernel": ["rbf", "linear", "poly"],
        "SVR__C": [50, 100, 200],
        "SVR__gamma": [0.1, "auto", 0.2],
        "SVR__epsilon": [0.1],
    },
    # KNR
    {
        "KNeighborsRegressor__weights": ["uniform", "distance"],
        # "KNeighborsRegressor__random_state": [SEED],
        "KNeighborsRegressor__n_neighbors": [5, 2, 3, 7],
        "KNeighborsRegressor__algorithm": ["auto", "ball_tree", "kd_tree"],
        # "KNeighborsRegressor__p": [1,2],
    },
]
