import math
import pandas as pd
import numpy as np
import xgboost
from sklearn.metrics import mean_squared_error, roc_auc_score
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split


def main():
    tables = [('bioresponse_original', 'target'), ('covertype_original', 'class'), ('dataset_31_credit-g', 'class'),
              ('eyemove_original', 'label'), ('jannis_original', 'class'), ('miniboone_original', 'signal'),
              ('Steel Plates Faults Data Set', 'Class'), ('superconduct_original', 'criticaltemp'),
              ('yprop_original', 'oz252')]

    for (table, target_col) in tables:
        for string in ['XT', 'XGB', 'GBM', 'RF']:
            model = {string: {}}
            print(table, target_col, string)

            df = pd.read_csv(f'{table}.csv')
            current_training_set, current_test_set = train_test_split(df, test_size=0.2, random_state=42)

            train = current_training_set.drop(target_col, axis=1).copy()
            train[target_col] = current_training_set[[target_col]].copy()
            predictor = TabularPredictor(label=target_col, verbosity=0).fit(train_data=train,
                                                                                 hyperparameters=model)
            test = current_test_set.drop(target_col, axis=1).copy()
            test[target_col] = current_test_set[[target_col]].copy()
            results = predictor.evaluate(data=test, model=predictor.get_model_names()[0])
            if 'accuracy' in results.keys():
                cur_score = abs(results['accuracy'])
            else:
                cur_score = abs(results['root_mean_squared_error'])

            train = current_training_set.drop(target_col, axis=1).copy()
            train[target_col] = current_training_set[[target_col]].copy()
            feature_importance = predictor.feature_importance(
                data=train, model=predictor.get_model_names()[0], feature_stage="original"
            )
            with open('output.csv', 'a') as f:
                f.write(
                    f'{string},,ALL,{tables},,,,,{cur_score},0,"{feature_importance}","{list(current_training_set.columns)}",,,\n')


if __name__ == "__main__":
    main()
