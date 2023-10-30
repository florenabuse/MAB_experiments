#codingL utf8

import math
import pandas as pd
import numpy as np
import xgboost
from sklearn.metrics import mean_squared_error, roc_auc_score
from autogluon.tabular import TabularPredictor


class AutoFeature_env(object):
    """
    The env of AutoFeature
    """
    def __init__(self, folder, base_name, base_train_path, base_test_path, repo_train_path_list, repo_test_path_list, tables, connections, index_col, target_col, target, model, max_try_num = 5, top_l_feature = 1):
        self.base_train_table = pd.read_csv(base_train_path)
        self.base_test_table = pd.read_csv(base_test_path)

        self.repo_train_table_list = []
        for repo_train_path in repo_train_path_list:
            self.repo_train_table_list.append(pd.read_csv(repo_train_path))

        self.repo_test_table_list = []
        for repo_test_path in repo_test_path_list:
            self.repo_test_table_list.append(pd.read_csv(repo_test_path))

        self.connections = pd.read_csv(connections)
        self.connections = self.connections.loc[(self.connections['pk_column'] == index_col) & (self.connections['fk_column'] == index_col)]
        self.tables = tables

        self.agg = base_name
        self.base_name = base_name
        self.folder = folder

        self.index_col = index_col
        self.target_col = target_col

        self.target = target        # AUC or MSE

        self.current_training_set = None
        self.current_test_set = None

        self.selected_features_dict = None
        self.current_model = None


        self.original_score = None
        self.cur_score = None
        self.prev_score = None


        self.max_try_num = max_try_num
        self.try_num = 0

        self.l = top_l_feature

        self.action_space = [i for i in range(len(self.repo_train_table_list))]
        self.action_valid = []

        self.model = model
        self.init_env()


    def init_env(self):
        # Init training set
        self.current_training_set = self.base_train_table.copy()
        self.current_test_set = self.base_test_table.copy()

        self.selected_features_dict = dict()
        self.selected_features_dict[self.base_name + '.csv'] = list(self.current_test_set.columns)

        # Init the model
        X_train, Y_train = self.get_training_dataset()
        self.current_model = self.model_training(X_train, Y_train)

        print('-' * 20 + "Init:" + '-' * 20)
        train_auc = self.model_test_rmse(X_train, Y_train)
        print(f"Train RMSE score: {train_auc}")

        X_test, Y_test = self.get_test_dataset()
        test_auc = self.model_test_rmse(X_test, Y_test)
        print(f"Test RMSE Score: {test_auc}")

        self.cur_score = test_auc
        self.original_score = test_auc

        # Check the valid action
        self.action_valid = []
        for i in range(len(self.repo_train_table_list)):
            if len(list(self.repo_train_table_list[i])) >= self.l:
                self.action_valid.append(i)


    def reset(self):
        self.try_num = 0

        X_train, Y_train = self.get_training_dataset()
        self.current_model = self.model_training(X_train, Y_train)

        X_test, Y_test = self.get_test_dataset()
        test_auc = self.model_test_rmse(X_test, Y_test)

        print('-' * 20 + "Reset:" + '_' * 20)
        print(f"Test RMSE score: {test_auc}")

        self.cur_score = test_auc
        self.original_score = test_auc

        self.action_valid = []
        for i in range(len(self.repo_train_table_list)):
            if len(list(self.repo_train_table_list[i])) >= self.l:
                self.action_valid.append(i)


    def step(self, action):
        """
        Execute the action
        :param action: the action selected by the agent
        :return: reward, done or not
        """
        print(f"Action: {action}")

        if action == -1:
            # Update the model on current training set
            X_train, Y_train = self.get_training_dataset()
            self.current_model = self.model_training(X_train, Y_train)

            # Test the performance of the model
            X_test, Y_test = self.get_test_dataset()
            test_auc = self.model_test_rmse(X_test, Y_test)

            # Update the reward and the valid action
            self.try_num += 1

            self.prev_score = self.cur_score
            self.cur_score = test_auc
            done = True
            print(list(self.current_training_set.columns))

            copy = self.get_feature_importances(self.current_model, X_test, Y_test)
            return self.cur_score - self.prev_score, test_auc, done, copy

        # Select the table from the pool
        foreign_key = None
        primary_key = None

        chosen_table = self.tables[action] + '.csv'
        connections_to_chosen_table = self.connections.loc[self.connections['fk_table'] == chosen_table]

        # selected_features_dict = self.selected_features_dict
        # for index, row in connections_to_chosen_table.iterrows():
        #     if row['pk_table'] in selected_features_dict.keys():
        #         if row['pk_column'] in selected_features_dict[row['pk_table']]:
        #             foreign_key = row['fk_column']
        #             primary_key = row['pk_column']
        #
        # if foreign_key is None:
        #     # Update the model on current training set
        #     X_train, Y_train = self.get_training_dataset()
        #     self.current_model = self.model_training(X_train, Y_train)
        #
        #     # Test the performance of the model
        #     X_test, Y_test = self.get_test_dataset()
        #     test_auc = self.model_test_rmse(X_test, Y_test)
        #
        #     # Update the reward and the valid action
        #     self.try_num += 1
        #
        #     self.prev_score = self.cur_score
        #     self.cur_score = test_auc
        #     done = False
        #     print(list(self.current_training_set.columns))
        #
        #     copy = self.get_feature_importances(self.current_model, X_test, Y_test)
        #     return self.cur_score - self.prev_score, test_auc, done, copy

        ## Join base and candidate table
        tmp_joined_train_table = pd.merge(self.current_training_set, self.repo_train_table_list[action], how='left', on=self.index_col)
        tmp_joined_test_table = pd.merge(self.current_test_set, self.repo_test_table_list[action], how='left', on=self.index_col)

        tmp_org_base_cols = list(self.current_training_set.columns)
        # tmp_org_base_cols.remove(self.index_col)
        tmp_org_base_cols.remove(self.target_col)

        # tmp_repo_cols = list(self.repo_train_table_list[action].columns)
        # tmp_repo_cols.remove(self.index_col)

        ## Train a model to rank the feature
        X_train = tmp_joined_train_table.drop(self.target_col, axis = 1)
        Y_train = tmp_joined_train_table[self.target_col]
        tmp_model = self.model_training(X_train, Y_train)
        feature_importance_dict = self.get_feature_importances(tmp_model, X_train, Y_train)

        copy = feature_importance_dict

        feature_importance_dict = feature_importance_dict['importance'].to_dict()
        for col in tmp_org_base_cols:
            if col in feature_importance_dict.keys():
                del feature_importance_dict[col]

        candidate_feature_rank_list = sorted(feature_importance_dict.items(), key=lambda x:x[1], reverse = True)

        ## Select top-l features
        new_feature_list = []
        repo_other_col_list = []

        if len(candidate_feature_rank_list) > self.l:
            for i in range(self.l):
                new_feature_list.append(candidate_feature_rank_list[i][0])
                if chosen_table not in self.selected_features_dict:
                    self.selected_features_dict[chosen_table] = [candidate_feature_rank_list[i][0]]
                else:
                    self.selected_features_dict[chosen_table].append(candidate_feature_rank_list[i][0])

            for i in range(self.l, len(candidate_feature_rank_list)):
                repo_other_col_list.append(candidate_feature_rank_list[i][0])
        else:
            for i in range(len(candidate_feature_rank_list)):
                new_feature_list.append(candidate_feature_rank_list[i][0])
                if chosen_table not in self.selected_features_dict:
                    self.selected_features_dict[chosen_table] = [candidate_feature_rank_list[i][0]]
                else:
                    self.selected_features_dict[chosen_table].append(candidate_feature_rank_list[i][0])

            self.action_valid.remove(action)

        if len(candidate_feature_rank_list) > 0:
            self.agg = self.compute_join_name(self.agg, self.index_col, chosen_table)

        self.current_training_set = tmp_joined_train_table[list(self.current_training_set.columns) + new_feature_list]
        self.current_test_set = tmp_joined_test_table[list(self.current_test_set.columns) + new_feature_list]

        # self.repo_train_table_list = self.repo_train_table_list[action].drop(self.index_col, axis = 1)
        self.repo_train_table_list[action] = self.repo_train_table_list[action].drop(new_feature_list, axis = 1)
        self.repo_test_table_list[action] = self.repo_test_table_list[action].drop(new_feature_list, axis = 1)

        if (len(list(self.repo_train_table_list[action]))) == 0:
            if action in self.action_valid:
                self.action_valid.remove(action)



        # Update the model on current training set
        X_train, Y_train = self.get_training_dataset()
        self.current_model = self.model_training(X_train, Y_train)


        # Test the performance of the model
        X_test, Y_test = self.get_test_dataset()
        test_auc = self.model_test_rmse(X_test, Y_test)


        # Update the reward and the valid action
        self.try_num += 1

        self.prev_score = self.cur_score
        self.cur_score = test_auc

        if self.try_num > self.max_try_num or len(self.action_valid) == 0:
            print("Try too much times!!!")
            done = True
            print(list(self.current_training_set.columns))
            return self.cur_score - self.prev_score, test_auc, done, copy
        else:
            done = False
            return self.cur_score - self.prev_score, test_auc, done, copy

    def compute_join_name(self, agg, to_column, to_table) -> str:
        """
        Compute the name of the partial join, given the properties of the new join and the previous join name.

        :param join_key_property: (neo4j relation property, outbound label, inbound label)
        :param partial_join_name: Name of the partial join.
        :return: The name of the next partial join
        """
        joined_path = f"{agg}--{to_column}-{to_table}"
        return joined_path

    def get_feature_importances(self, tmp_model, X_train, Y_train):
        train = X_train.copy()
        train[self.target_col] = Y_train.copy()
        return tmp_model.feature_importance(
            data=train, model=tmp_model.get_model_names()[0], feature_stage="original"
        )

    def get_training_dataset(self):
        X_train = self.current_training_set.drop([self.target_col], axis = 1)
        Y_train = self.current_training_set[self.target_col]
        return X_train, Y_train

    def get_test_dataset(self):
        X_test = self.current_test_set.drop([self.target_col], axis = 1)
        Y_test = self.current_test_set[self.target_col]
        return X_test, Y_test

    def model_training(self, X_train, Y_train):
        train = X_train.copy()
        train[self.target_col] = Y_train.copy()
        predictor = TabularPredictor(label=self.target_col, verbosity=0, path="AutogluonModels").fit(train_data=train,
                                     hyperparameters=self.model)
        return predictor

    def model_test_rmse(self, X_test, Y_test):
        test = X_test.copy()
        test[self.target_col] = Y_test.copy()
        results = self.current_model.evaluate(data=test, model=self.current_model.get_model_names()[0])
        if 'accuracy' in results.keys():
            return abs(results['accuracy'])
        else:
            return abs(results['root_mean_squared_error'])

    def get_current_features(self):
        return self.selected_features_dict
