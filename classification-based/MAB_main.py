import pandas as pd

from Environment_MAB import AutoFeature_env
from Agent_MAB import AutoFeature_agent
from warnings import filterwarnings
from sklearn.model_selection import train_test_split

filterwarnings("ignore", category=UserWarning)
filterwarnings("ignore", category=FutureWarning)


def main_MAB(tables, folder, base_name, index_col, target_col):

    # Parameters for the environment

    base_train_path = f"../data2/{folder}/{base_name}_train.csv"
    base_test_path = f"../data2/{folder}/{base_name}_test.csv"
    repo_train_path = [f'../data2/{folder}/' + i + '_train.csv' for i in tables]
    repo_test_path = [f'../data2/{folder}/' + i + '_test.csv' for i in tables]

    connections = f"../data2/{folder}/connections2.csv"

    model_target = 0.60

    max_try_num = 30

    topl = 3

    random_state = 42

    # All models: {"RF": {}, "GBM": {}, "XGB": {}, "XT": {}, 'KNN': {},
    # Exclude tree-based models: RF, GBM, XGB, XT
    # models = ["LR-1", "LR-2", "KNN"]
    models = ["LR-1", "KNN"]
    model_map = {
        "LR-1": {'LR': {'penalty': 'L1'}},
        # "LR-2": {'LR': {'penalty': 'L2'}},
        "KNN": {'KNN': {}}
    }
    for string in models:
        model = model_map[string]

        env = AutoFeature_env(folder, base_name, base_train_path, base_test_path, repo_train_path, repo_test_path, tables, connections, index_col, target_col, model_target, model, max_try_num, topl)

        res_csv = "./data/result_mab.csv"

        autofeature = AutoFeature_agent(env, res_csv, random_state)

        print("Agent Ready!")

        # Train the workload
        autofeature.augment()


def main(folder, base_name, index_col, target_col, dataset_tables):

    a = pd.read_csv('../results/results_mab_first_scenraio.csv')
    a = a[['algorithm','data_path','approach','data_label','join_time','total_time','feature_selection_time','depth','accuracy','train_time','feature_importance','join_path_features','cutoff_threshold','redundancy_threshold','rank']]
    a.to_csv('../results/results_mab_first_scenraio.csv', index=False)

    # folder = 'covertype'
    # base_name = 'table_0_0'
    # index_col = 'Key_0_0'
    # target_col = 'covertype/table_0_0.class'

    # tables = ['table_0_0', 'table_1_1', 'table_1_2', 'table_1_3']
    tables = dataset_tables
    # if folder == 'school':
    #     tables.append('base')
    # else:
    #     tables.append('table_0_0')
    # for table in dataset_tables:
    #     tables.append(table)

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

        # Add sampling only if necessary (covertype, eyemove, miniboone)
        # df = df.sample(frac=0.1, random_state=42)
        a_train, a_test = train_test_split(df, test_size=0.2, random_state=42)
        a_train.to_csv(f"../data2/{folder}/{entry}_train.csv", index=False)
        a_test.to_csv(f"../data2/{folder}/{entry}_test.csv", index=False)

    # tables = ['table_1_1', 'table_1_2', 'table_1_3']
    tables = []
    for table in dataset_tables:
        if table == base_name:
            continue
        else:
            tables.append(table)

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


def get_connections():
    file_path = f"../data2/all_connections_basic2.csv"
    df = pd.read_csv(file_path)

    # Keep only the correct datasets
    dataset_list = ['credit/', 'eyemove/', 'covertype/', 'jannis/', 'miniboone/', 'steel/', 'bioresponse/', 'school/']
    df = df[df['from_path'].isin(dataset_list)]
    df = df[df['to_path'].isin(dataset_list)]

    key_list = ['Key_0_0', 'DBN']
    df = df[df['from_column'].isin(key_list)]
    df = df[df['to_column'].isin(key_list)]

    # Merge columns - from table
    df['from_path'] = df['from_path'].str.rstrip('/')
    df['from_table'] = df['from_table'].str.rstrip('.csv')
    df['fk_table'] = df['from_path'] + '_' + df['from_table']
    df['fk_column'] = df['from_column']
    df = df.drop(columns=['from_path', 'from_table', 'from_column'])

    # Merge columns - to table
    df['to_path'] = df['to_path'].str.rstrip('/')
    df['to_label'] = df['to_label'].str.rstrip('.csv')
    df['pk_table'] = df['to_path'] + '_' + df['to_label']
    df['pk_column'] = df['to_column']
    df = df.drop(columns=['to_path', 'to_label', 'to_column'])

    new_file_path = f"../data2/connections2.csv"
    df.to_csv(new_file_path, index=False)


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

    # main()

    # credit_parameters = ['credit', 'table_0_0', 'Key_0_0', 'credit/table_0_0.class',
    #                      ['table_1_1', 'table_1_2']]
    # eyemove_parameters = ['eyemove', 'table_0_0', 'Key_0_0', 'eyemove/table_0_0.label',
    #                       ['table_1_1', 'table_1_2', 'table_1_3']]
    # covertype_parameters = ['covertype', 'table_0_0', 'Key_0_0', 'covertype/table_0_0.class',
    #                         ['table_1_1', 'table_1_2', 'table_1_3']]
    # jannis_parameters = ['jannis', 'table_0_0', 'Key_0_0', 'jannis/table_0_0.class',
    #                      ['table_1_1', 'table_1_2', 'table_1_3']]
    # miniboone_parameters = ['miniboone', 'table_0_0', 'Key_0_0', 'miniboone/table_0_0.signal',
    #                         ['table_1_1', 'table_1_2', 'table_1_3']]
    # steel_parameters = ['steel', 'table_0_0', 'Key_0_0', 'steel/table_0_0.Class',
    #                     ['table_1_1', 'table_1_2', 'table_1_3']]
    # bioresponse_parameters = ['bioresponse', 'table_0_0', 'Key_0_0', 'bioresponse/table_0_0.target',
    #                           ['table_1_1', 'table_1_2', 'table_1_3']]
    # school_parameters = ['school', 'base', 'DBN', 'school/base.class',
    #                      ['ap', 'crime', 'disc', 'esl', 'gender', 'math',
    #                       'oss', 'pe', 'qr', 's2tr', 'sat', 'transfer', 'yabc', '2010_Gen_Ed_Survey_Data',
    #                       'Schools_Progress_Report_2012-2013', '2013_NYC_School_Survey']]
    #
    # datasets = [credit_parameters, eyemove_parameters, covertype_parameters, jannis_parameters,
    #             miniboone_parameters, steel_parameters, bioresponse_parameters, school_parameters]
    #
    # for dataset in datasets:
    #     main(folder=dataset[0], base_name=dataset[1], index_col=dataset[2], target_col=dataset[3],
    #          dataset_tables=dataset[4])

    # get_connections()

    # CONNECTIONS SCENARIO
    file_path = f"../data2/connections2.csv"
    df = pd.read_csv(file_path)
    tables = df['fk_table'].unique()

    credit_parameters = ['credit_table_0_0', 'Key_0_0', 'class', tables]
    eyemove_parameters = ['eyemove_table_0_0', 'Key_0_0', 'label', tables]
    covertype_parameters = ['covertype_table_0_0', 'Key_0_0', 'class', tables]
    jannis_parameters = ['jannis_table_0_0', 'Key_0_0', 'class', tables]
    miniboone_parameters = ['miniboone_table_0_0', 'Key_0_0', 'signal', tables]
    steel_parameters = ['steel_table_0_0', 'Key_0_0', 'Class', tables]
    bioresponse_parameters = ['bioresponse_table_0_0', 'Key_0_0', 'target', tables]
    school_parameters = ['school_base', 'DBN', 'class', tables]

    datasets = [credit_parameters, eyemove_parameters, covertype_parameters, jannis_parameters,
                miniboone_parameters, steel_parameters, bioresponse_parameters, school_parameters]

    for dataset in datasets:
        main(folder='all', base_name=dataset[0], index_col=dataset[1], target_col=dataset[2],
             dataset_tables=dataset[3])
