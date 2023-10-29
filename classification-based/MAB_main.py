from Environment_MAB import AutoFeature_env
from Agent_MAB import AutoFeature_agent
from warnings import filterwarnings

from sklearn.model_selection import train_test_split
import pandas as pd

filterwarnings("ignore", category=UserWarning)
filterwarnings("ignore", category=FutureWarning)

def main_MAB(tables, folder, base_name, index_col, target_col):

    # Parameters for the environment

    base_train_path = f"../data2/{folder}/{base_name}_train.csv"
    base_test_path = f"../data2/{folder}/{base_name}_test.csv"
    repo_train_path = [f'../data2/{folder}/' + i + '_train.csv' for i in tables]
    repo_test_path = [f'../data2/{folder}/' + i + '_test.csv' for i in tables]

    connections = f"../data2/{folder}/connections.csv"

    model_target = 0.60

    max_try_num = 30

    topl = 3

    random_state = 42

    #  {"RF": {}, "GBM": {}, "XGB": {}, "XT": {}, 'KNN': {},
    # 'LR': [{'penalty': 'L1'}, {'penalty': 'L2'}]
    # tree-based: RF, GBM, XGB, XT
    # models = {
    #     "LR": [
    #         {"penalty": "l1"},
    #         {"penalty": "l2"}
    #     ],
    #     "KNN": {},
    # }
    models = ["LR", "KNN"]
    for string in models:
        model = {string: {}}
        if string == 'LR':
            print('yes')
            model = {'LR': [{'penalty': 'L1'}, {'penalty': 'L2'}]}

        env = AutoFeature_env(folder, base_name, base_train_path, base_test_path, repo_train_path, repo_test_path, tables, connections, index_col, target_col, model_target, model, max_try_num, topl)

        res_csv = "./data/result_mab.csv"

        autofeature = AutoFeature_agent(env, res_csv, random_state)

        print("Agent Ready!")

        # Train the workload
        autofeature.augment()

def main():

    a = pd.read_csv('../results/results_mab_first_scenraio.csv')
    a = a[['algorithm','data_path','approach','data_label','join_time','total_time','feature_selection_time','depth','accuracy','train_time','feature_importance','join_path_features','cutoff_threshold','redundancy_threshold','rank']]
    a.to_csv('../results/results_mab_first_scenraio.csv', index=False)

    folder = 'covertype'
    base_name = 'table_0_0'
    index_col = 'Key_0_0'
    target_col = 'covertype/table_0_0.class'

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


    main_MAB(tables, folder, base_name, index_col, target_col)

    # tables = ['temp', 'co_daily_summary', 'hap_daily_summary', 'lead_daily_summary', 'no2_daily_summary', 'nonoxnoy_daily_summary',
    #           'o3_daily_summary', 'pm10_daily_summary', 'pm25_frm_daily_summary', 'pm25_nonfrm_daily_summary',
    #           'pm25_speciation_daily_summary', 'pressure_daily_summary', 'rh_and_dp_daily_summary',
    #           'so2_daily_summary', 'temperature_daily_summary', 'voc_daily_summary', 'wind_daily_summary']
    # tables = ['2010_Gen_Ed_Survey_Data', 'esl', 's2tr',
    #           '2013_NYC_School_Survey', 'gender', 'sat',
    #           'ap', 'math', 'Schools_Progress_Report_2012-2013',
    #           'oss', 'transfer',
    #           'crime', 'pe', 'yabc',
    #           'disc', 'qr'
    #           ]
    # all_con = pd.read_csv('../data/all_connections.csv')
    # all_con = all_con[['from_table', 'from_column', 'to_label', 'to_column']]
    # all_con.columns = ['pk_table', 'pk_column', 'fk_table', 'fk_column']
    # all_con.to_csv('../data/connections.csv', index=False)

if __name__ == "__main__":
    # all_con = pd.read_csv('../data/all_connections.csv')
    # all_con = all_con[['from_table', 'from_column', 'to_label', 'to_column']]
    # all_con.columns = ['pk_table', 'pk_column', 'fk_table', 'fk_column']
    # all_con.to_csv('../data/connections.csv', index=False)

    # conn = pd.read_csv('../data2/connections.csv')
    #
    # conn['pk_table'] = conn[['from_path', 'from_table']].apply(
    #     lambda x: x[0][0:-1] + '_' + x[1],
    #     axis=1
    # )
    # conn['fk_table'] = conn[['to_path', 'to_label']].apply(
    #     lambda x: x[0][0:-1] + '_' + x[1],
    #     axis=1
    # )
    #
    # print(conn[['fk_table', 'to_column', 'pk_table', 'from_column']])
    # conn[['fk_table', 'to_column', 'pk_table', 'from_column']].to_csv('../data2/connections.csv', index=False)

    main()
