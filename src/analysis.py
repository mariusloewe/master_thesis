class PipelineAnalysis:
    def __init__(self, pipeline_path):
        self.pipeline_path = pipeline_path
        self.raw_pipeline = self._load_pipeline()

    def _load_pipeline(self):
        pass


# TODO: Refactor this part
"""    
    def _algorithm_opt(self):

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

            num_cols = self.x_train.columns  # .select_dtypes(include=['float']).columns
            scaler = MinMaxScaler()

            self.x_train[num_cols] = scaler.fit_transform(self.x_train[num_cols])
            print(len(self.x_train))

            self.x_test[num_cols] = scaler.fit_transform(self.x_test[num_cols])
            print(len(self.x_test))

            oversamplings = [SVMSMOTE(random_state=self.seed), SMOTETomek(random_state=self.seed),
                             SMOTE(random_state=self.seed),
                             BorderlineSMOTE(random_state=self.seed)]  # , ADASYN(random_state=self.seed)]
            models = [  # ('SGD',SGDClassifier(random_state=self.seed)),
                # ('DTC', DecisionTreeClassifier()),
                # ('SVC',SVC()),
                # ('MLP',MLPClassifier()),
                # ('ABC',AdaBoostClassifier()),
                # ('XGB', XGBClassifier()),
                # ('RandomForestClassifier', RandomForestClassifier()),
                # ('KNC',KNeighborsClassifier()),
                # ('SymbolicClassifier', SymbolicClassifier()),
                # ('LR', LogisticRegression())

            ]

            parameters_list = [
                # {'sampling__random_state':[self.seed],
                # 'SGD__alpha': [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0],
                # 'SGD__penalty': ['l2','l1','elasticnet'],
                # 'SGD__loss':['log','perceptron','hinge'],
                # 'SGD__n_jobs': [-1],
                # 'SGD__random_state': [self.seed]},

                # {'sampling__random_state':[self.seed],
                # 'DTC__class_weight':[None],
                # 'DTC__criterion':['gini', 'entropy'],
                # 'DTC__max_depth':[3,4,6],
                # 'DTC__max_features':[,None],
                # 'DTC__max_leaf_nodes':[2,3,5,9,15],
                # 'DTC__min_impurity_split':[1e-07, 1e-08, 1e-06],
                # 'DTC__min_samples_leaf':[2,3,3],
                # 'DTC__min_samples_split':[3,5,8,6],
                # 'DTC__min_weight_fraction_leaf':[0.1, 0.3],
                # 'DTC__presort':[False],
                # 'DTC__random_state':[self.seed],
                # 'DTC__splitter':['best']},

                # {'sampling__random_state' : [self.seed],
                # 'SVC__C':[0.4, 0.3,0.8],
                # 'SVC__decision_function_shape':['ovr'],
                # 'SVC__degree':[3,5,7],
                # 'SVC__gamma':[0.3,0.5,0.8],
                ## 'SVC__kernel':['poly', 'rbf'],
                # 'SVC__max_iter':[-1],
                # 'SVC__probability':[True],
                # 'SVC__random_state':[self.seed],
                # 'SVC__tol':[0.001, 0.0005, 0.002],
                # 'SVC__verbose':[False]},

                # {'sampling__random_state' : [self.seed],
                # 'MLP__solver': ['adam', 'sgd', 'lbfgs'],
                # 'MLP__learning_rate': ['constant', 'invscaling', 'adaptive'],
                # 'MLP__activation': ['logistic', 'tanh', 'relu'],
                # 'MLP__hidden_layer_sizes' : [(10,10),(10,5),(15,10,10,5,5),(10,10,5,5)
                ##    #(10,10,5,5,5)
                #     ],
                # 'MLP__random_state' : [self.seed]},

                # {"ABC__learning_rate": [1,0.8,0.6,1.2],
                # "ABC__random_state": [self.seed],
                # "ABC__n_estimators": [50, 20,80,60,40]
                # },

                # {"XGB__learning_rate": [1,0.8,0.6,1.2],
                # "XGB__random_state": [self.seed],
                # "XGB__n_estimators": [50, 20,100,200],
                # "XGB__max_depth": [5, 3, 10],
                # "XGB__gamma": [0, 0.1, 0.2],
                #  "XGB__subsample": [0.8, 0.5, 0.9],
                #  "XGB__min_child_weight": [1,2, 5],
                # },
                # {
                #    'RandomForestClassifier__bootstrap': [True],
                #    'RandomForestClassifier__max_depth': [2, 3, 4, 6],
                #    #'RandomForestClassifier__max_features': [2, 3],
                #    'RandomForestClassifier__min_samples_leaf': [3,4,6],
                #   'RandomForestClassifier__min_samples_split': [2, 3, 5, 8],
                #    'RandomForestClassifier__n_estimators': [30, 80, 200, 300],
                #    'RandomForestClassifier__random_state': [self.seed]
                # },
                # {"KNC__weights": ['distance'], #'uniform',
                # "KNC__random_state": [self.seed],
                # "KNC__n_neighbors": [5,2,3,7],
                # "KNC__algorithm": ['auto', 'ball_tree','kd_tree'],
                # "KNC__n_jobs": [5],
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
                # {'LR__penalty': ['l1', 'l2'],
                # 'LR__random_state' : [self.seed],
                # 'LR__solver' : ['liblinear','saga'],
                # 'LR__multi_class' : ['ovr', 'auto'],
                # 'LR__C': [0.5, 0.8,0.7,0.9],
                # 'LR__max_iter':[80,100,200]
                # },

                {'sampling__random_state': [self.seed],
                 'DTR__criterion': ['mse', 'mae'],
                 'DTR__max_depth': [3, 4, 6],
                 'DTR__max_leaf_nodes': [2, 3, 5, 9, 15],
                 'DTR__min_impurity_split': [0, 1e-07, 1e-08, 1e-06],
                 'DTR__min_samples_leaf': [2, 3, 3],
                 'DTR__min_samples_split': [2, 3, 5, 8, 6],
                 'DTR__ccp_alpha': [0, 0.1, 0.3],
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

                        gscv = GridSearchCV(estimator=model, param_grid=parameters_list[m], cv=5, scoring='f1_macro',
                                            n_jobs=6)
                        gscv.fit(self.x_train, self.y_train)
                        results_df = pd.DataFrame(gscv.cv_results_)
                        results_df = results_df.loc[results_df['rank_test_score'] == 1]
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
                            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X_, y_,
                                                                                                    test_size=0.30,
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
                            rec = recall_score(self.y_test, y_pred_, average='micro')
                            print('Precision is %s and Recall is %s' % (round(prec, 3), round(rec, 3)))

                            self.cm = confusion_matrix(self.y_test, y_pred_)

                            # Positive_repsonses = self.y_test.loc[self.y_test == 1].count()

                            y_score = best_estimator.fit(self.x_train, self.y_train).predict(self.x_test)

                            fpr = dict()
                            tpr = dict()
                            roc_auc = dict()

                            fpr, tpr, _ = roc_curve(self.y_test, y_score, pos_label=1)
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
                            plt.savefig(
                                'C:\\Users\\mariu\\OneDrive\\Dokumente\\NOVA\\Thesis\\20191015_new_data\\Final Results\\Graphs\\_results_' + str(
                                    self.target) + '_' + str(models[m])[:10] + '_' + str(selection[s])[:10] + '_' + str(
                                    oversamplings[o])[:10] + '_' + str(self.seed) + '.png')

                            results_string = str(
                                [self.seed, round(gscv.best_score_, 3), confusion_matrix(self.y_test, y_pred_)[0][0],
                                 confusion_matrix(self.y_test, y_pred_)[1][0],
                                 confusion_matrix(self.y_test, y_pred_)[0][1],
                                 confusion_matrix(self.y_test, y_pred_)[1][1],
                                 round(accuracy_score(self.y_test, y_pred_), 3),
                                 round(prec, 3), round(rec, 3),
                                 round(f1_score(self.y_test, y_pred_, average='micro'), 3)
                                 # ,round(roc_auc_score(self.y_test, y_pred_),3)
                                    , str(self.scoring_metric), str(gscv.best_estimator_),
                                 str(selection[s]), str(oversamplings[o]), str(gscv.best_params_)]) + '\n'
                            with open(self.results_filepath, 'a')as f:
                                f.write(results_string)

                            aggregate_results.append(
                                [self.seed, round(gscv.best_score_, 3), round(accuracy_score(self.y_test, y_pred_), 3),
                                 confusion_matrix(self.y_test, y_pred_)[0][0],
                                 confusion_matrix(self.y_test, y_pred_)[1][0],
                                 confusion_matrix(self.y_test, y_pred_)[0][1],
                                 confusion_matrix(self.y_test, y_pred_)[1][1], round(prec, 3), round(rec, 3),
                                 round(f1_score(self.y_test, y_pred_, average='micro'), 3)
                                 # ,round(roc_auc_score(self.y_test, y_pred_),3)
                                 ])
                            median_results.append([round(accuracy_score(self.y_test, y_pred_), 3)])

                        # print(aggregate_results)
                        median = np.median(median_results)
                        aggregate_results_mean = np.around(np.mean(aggregate_results, axis=0), decimals=3)

                        # print(aggregate_results_mean)
                        results_str = str(
                            [aggregate_results_mean[0], aggregate_results_mean[1], aggregate_results_mean[2], median,
                             aggregate_results_mean[3], aggregate_results_mean[4], aggregate_results_mean[5],
                             aggregate_results_mean[6], aggregate_results_mean[7], aggregate_results_mean[8],
                             aggregate_results_mean[9],  # aggregate_results_mean[10],
                             str(selection[s]), str(oversamplings[o]), str(gscv.best_estimator_)]) + '\n'
                        with open(
                                'C:\\Users\\mariu\\OneDrive\\Dokumente\\NOVA\\Thesis\\20191015_new_data\\Final Results\\AVG\\DTR\\_results_' + str(
                                        self.target) + '.csv', 'a')as f:
                            f.write(results_str)
                            
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
            ) = train_test_split(X_, y_, test_size=0.30, random_state=self.seed)

            num_cols = self.x_train.columns
            scaler = MinMaxScaler()

            self.x_train[num_cols] = scaler.fit_transform(
                self.x_train[num_cols]
            )
            print(len(self.x_train))

            self.x_test[num_cols] = scaler.fit_transform(self.x_test[num_cols])
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
                            math.sqrt(mean_squared_error(self.y_test, y_pred_))
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
"""
