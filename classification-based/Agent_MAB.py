#coding: utf8
import random
import math
import time
import csv

class AutoFeature_agent(object):
    """
    The agent of AutoFeature
    """

    def __init__(self, env, mab_csv, random_state):
        self.env = env
        self.mab_result_csv = mab_csv
        self.random_state = random_state
        self.gamma = 0.6

        self.ucb_score_list = [0 for _ in range(len(self.env.repo_train_table_list))]
        self.action_num_list = [0 for _ in range(len(self.env.repo_train_table_list))]
        self.acc_reward_list = [0 for _ in range(len(self.env.repo_train_table_list))]

    def choose_action(self):
        """
        Choose a new action
        :return: the action chosen for next step
        """


        ucb_tmp_list = []
        for i in range(len(self.ucb_score_list)):
            if i in self.env.action_valid:
                ucb_tmp_list.append([i, self.ucb_score_list[i]])

        sort_ucb_tmp_list = sorted(ucb_tmp_list, key=lambda x: x[1], reverse=True)


        i = 0
        while True:
            # Need to check this here
            action = sort_ucb_tmp_list[i][0]
            isGood = False

            chosen_table = self.env.tables[action] + '.csv'
            connections_to_chosen_table = self.env.connections.loc[(self.env.connections['fk_table'] == chosen_table) | (self.env.connections['pk_table'] == chosen_table)]

            selected_features_dict = self.env.selected_features_dict
            for index, row in connections_to_chosen_table.iterrows():
                if row['pk_table'] in selected_features_dict.keys():
                    if row['pk_column'] in selected_features_dict[row['pk_table']]:
                        isGood = True
                        break
                if row['fk_table'] in selected_features_dict.keys():
                    if row['fk_column'] in selected_features_dict[row['fk_table']]:
                        isGood = True
                        break
            if isGood:
                break
            else:
                i += 1
                if i == len(sort_ucb_tmp_list):
                    return -1
                continue

        return action


    def update_ucb(self, action_index, step_reward):
        """
        update acc_reward and ucb score
        :param action: chosen action
        :param step_reward: reward in this step
        :return:
        """
        self.acc_reward_list[action_index] = (self.acc_reward_list[action_index] * self.action_num_list[action_index] + step_reward) / (self.action_num_list[action_index] + 1)
        self.action_num_list[action_index] += 1

        self.ucb_score_list[action_index] = self.acc_reward_list[action_index] + self.gamma * math.sqrt(2 * sum(self.action_num_list)) / (self.action_num_list[action_index] + 1)


    def augment(self):
        self.env.reset()
        time_start = time.time()

        X_test, Y_test = self.env.get_test_dataset()
        test_mse = self.env.model_test_rmse(X_test, Y_test)

        counter = 0
        while True:
            action = self.choose_action()
            reward, test_auc, done, ft_imp = self.env.step(action)

            # reward = - reward

            counter += 1

            X_test, Y_test = self.env.get_test_dataset()
            test_mse = self.env.model_test_rmse(X_test, Y_test)

            self.update_ucb(action, reward)

            if done:
                time_end = time.time()

                print(f"Final features：{self.env.get_current_features()}")
                print("The accuracy of current model：" + str(self.env.cur_score))
                print("Benefit：" + str(max(0, self.env.cur_score - self.env.original_score)))
                print("Time：" + str(time_end - time_start))

                predictor = self.env.current_model
                print('TEST')
                print(self.env.predictor.get_model_names()[0])
                feature_importance = dict(zip(list(ft_imp.index), ft_imp["importance"]))
                with open('../results/results_mab_first_scenraio.csv', 'a') as f:
                    f.write(f'{self.env.predictor.get_model_names()[0]},{self.env.agg},MAB,{self.env.folder},,{round(time_end - time_start, 2)},{round(time_end - time_start, 2)},,{self.env.cur_score},0,"{feature_importance}","{list(self.env.current_training_set.columns)}",,,\n')
                break