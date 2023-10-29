from Environment_RL import AutoFeature_env
from Agent_RL import Autofeature_agent

from sklearn.model_selection import train_test_split
import pandas as pd

from warnings import filterwarnings
filterwarnings("ignore", category=UserWarning)
filterwarnings("ignore", category=FutureWarning)

def main_RL(tables, folder, base_name, index_col, target_col):
    # Parameters for the environment
    base_train_path = f"../data2/{folder}/{base_name}_train.csv"
    base_test_path = f"../data2/{folder}/{base_name}_test.csv"
    repo_train_path = [f'../data2/{folder}/' + i + '_train.csv' for i in tables]
    repo_test_path = [f'../data2/{folder}/' + i + '_test.csv' for i in tables]

    connections = f"../data2/{folder}/connections.csv"

    for string in ['XT', 'XGB', 'GBM', 'RF']:
        model = {string: {}}

        model_target = 0
        max_try_num = 20

        env = AutoFeature_env(folder, base_name, base_train_path, base_test_path, repo_train_path, repo_test_path, index_col, target_col, model_target, model, tables, max_try_num)

        # Parameters for the agent
        learning_rate = 0.05
        reward_decay = 0.9
        e_greedy = 1
        update_freq = 50
        mem_cap = 1000
        BDQN_batch_size = 5

        autodata = Autofeature_agent(env, BDQN_batch_size, learning_rate, reward_decay, e_greedy, update_freq, mem_cap, BDQN_batch_size)

        print("Agent Ready!")

        # Train the workload
        autodata.train_workload()


if __name__ == "__main__":
    # tables = ['pm10_daily_summary.csv', 'pm25_frm_daily_summary.csv', 'pressure_daily_summary.csv',
    #           'rh_and_dp_daily_summary.csv', 'so2_daily_summary.csv', 'voc_daily_summary.csv', 'wind_daily_summary.csv'
    #           ]

    a = pd.read_csv('../results/results_mab_first_scenraio.csv')
    a = a[['algorithm', 'data_path', 'approach', 'data_label', 'join_time', 'total_time', 'feature_selection_time', 'depth', 'accuracy', 'train_time', 'feature_importance', 'join_path_features', 'cutoff_threshold',
           'redundancy_threshold', 'rank']]
    a.to_csv('../results/results_mab_first_scenraio.csv', index=False)

    folder = 'yprop'
    base_name = 'table_0_0'
    index_col = 'Key_0_0'
    target_col = 'yprop/table_0_0.oz252'

    tables = ['table_0_0', 'table_1_1', 'table_1_2', 'table_1_3']

    for entry in tables:
        df = pd.read_csv(f"../data2/{folder}/{entry}.csv")
        me = [f'{folder}/{entry}.{i}' for i in df.columns if i != index_col and i != target_col]
        if index_col in df.columns:
            me.append(index_col)
        if target_col in df.columns:
            me.append(target_col)

        re = [i for i in df.columns if i != index_col and i != target_col]
        if index_col in df.columns:
            re.append(index_col)
        if target_col in df.columns:
            re.append(target_col)

        df = df[re]
        df.columns = me

        a_train, a_test = train_test_split(df, test_size=0.2, random_state=42)
        a_train.to_csv(f"../data2/{folder}/{entry}_train.csv", index=False)
        a_test.to_csv(f"../data2/{folder}/{entry}_test.csv", index=False)

    tables = ['table_1_1', 'table_1_2', 'table_1_3']

    main_RL(tables, folder, base_name, index_col, target_col)