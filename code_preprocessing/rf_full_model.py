import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import wandb
from itertools import product
from get_metrics import get_test_metrics


# Data preparation function
def logo_data_prep(pair_id, df):
    X_train = df[df['pair_id'] != pair_id]
    X_test = df[df['pair_id'] == pair_id]

    y_train = X_train['is_discomfort']
    y_test = X_test['is_discomfort']

    return X_train, y_train, X_test, y_test

# Select Modelities
def modalities_combination_data_prep(modalities_combination_vec, X_train, X_test):
    selected_modalities_train = pd.DataFrame()
    selected_modalities_test = pd.DataFrame()
    
    if modalities_combination_vec[0]: # audio
        selected_modalities_train = pd.concat([selected_modalities_train, X_train.iloc[:, 3:133]], axis=1)
        selected_modalities_test = pd.concat([selected_modalities_test, X_test.iloc[:, 3:133]], axis=1)
    if modalities_combination_vec[1]: # face
        selected_modalities_train = pd.concat([selected_modalities_train, X_train.iloc[:, 133:231]], axis=1)
        selected_modalities_test = pd.concat([selected_modalities_test, X_test.iloc[:, 133:231]], axis=1)
    if modalities_combination_vec[2]: # pose
        selected_modalities_train = pd.concat([selected_modalities_train, X_train.iloc[:, 231:333]], axis=1)
        selected_modalities_test = pd.concat([selected_modalities_test, X_test.iloc[:, 231:333]], axis=1)
    if modalities_combination_vec[3]: # sensor
        selected_modalities_train = pd.concat([selected_modalities_train, X_train.iloc[:, 333:344]], axis=1)
        selected_modalities_test = pd.concat([selected_modalities_test, X_test.iloc[:, 333:344]], axis=1)
    
    return selected_modalities_train, selected_modalities_test


PAIR_ID_LIST = [1001, 1200, 1302, 5001, 6002]


def train():

    wandb.init()
    config = wandb.config
    print(config)
    seed_value = 42
    
    test_metrics_list = {
        "test_accuracy": [],
        "test_precision": [],
        "test_recall": [],
        "test_f1": [],
        "test_accuracy_tolerant": [],
        "test_precision_tolerant": [],
        "test_recall_tolerant": [],
        "test_f1_tolerant": []
    }
    fold_importances = []
    
    if (config.feature_set_tag == 'Stat'):
        stat_feature_df = pd.read_csv("./stat/sign_features_ttest_full.csv")
        stat_feature = stat_feature_df['feature'].tolist()
    
    for fold in range(5):
        
        pair_id = PAIR_ID_LIST[fold]
        
        # select dataset and modalities
        if (config.dataset == 'clean'):
            X_train, y_train, X_test, y_test = logo_data_prep(pair_id, df_10hz_clean)
            X_train, X_test = modalities_combination_data_prep(config.modalities_combination, X_train, X_test)
            if (config.feature_set_tag == 'Stat'):
                repeat_features = [f for f in stat_feature if f in X_train.columns]
                X_train = X_train[repeat_features]
                X_test = X_test[repeat_features]
                
        elif (config.dataset == 'normalized'):
            X_train, y_train, X_test, y_test = logo_data_prep(pair_id, df_10hz_normalized)
            X_train, X_test = modalities_combination_data_prep(config.modalities_combination, X_train, X_test)
            if (config.feature_set_tag == 'Stat'):
                repeat_features = [f for f in stat_feature if f in X_train.columns]
                X_train = X_train[repeat_features]
                X_test = X_test[repeat_features]
                
        else: # pca, skip modalities selection
            X_train, y_train, X_test, y_test = logo_data_prep(pair_id, df_10hz_pca)
            X_train = X_train.drop('is_discomfort', axis=1)
            X_test = X_test.drop('is_discomfort', axis=1)

        feature_names = X_train.columns
        
        # balance training dataset
        smote = SMOTE(random_state=42) 
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        rf = RandomForestClassifier(
                random_state=seed_value,
                n_estimators=config.n_estimators,
                max_depth=config.max_depth
            )
        rf.fit(X_train_balanced, y_train_balanced)
    
        y_pred = rf.predict(X_test)
        test_metrics = get_test_metrics(y_pred, y_test, tolerance=1)
        for key in test_metrics:
            test_metrics_list[key].append(test_metrics[key])
        test_metrics = {f"t{fold}_{k}": v for k, v in test_metrics.items()}
        wandb.log(test_metrics)

        # wandb.log({f"t{fold}_conf_mat" : wandb.plot.confusion_matrix(probs=None,
        #         y_true=y_test.astype(int) , preds=y_pred.astype(int) ,
        #         class_names=['no_discomfort', 'is_discomfort'])})
        
        # print(test_metrics)
        print(confusion_matrix(y_test, y_pred))
        print(f'Fold {fold} Feature Importance:{rf.feature_importances_}')
        fold_importances.append(rf.feature_importances_)

    # Calculate average metrics and log to wandb
    avg_test_metrics = {f"avg_{key}": np.mean(values) for key, values in test_metrics_list.items()}
    wandb.log(avg_test_metrics)

    print("Average Metrics Across Groups:", avg_test_metrics)

    avg_feature_importances = np.mean(fold_importances, axis=0)
    feature_importance_dict = {feature_names[i]: avg_feature_importances[i] for i in range(len(feature_names))}
    sorted_feature_importance = dict(sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True))

    print("Sorted Feature Importances:", sorted_feature_importance)
    # wandb.log({"feature_importances": feature_importance_dict})




def main():
    global df_10hz_clean
    global df_10hz_normalized
    global df_10hz_pca 
    df_10hz_clean = pd.read_csv("../data/pair_ordered_features_overlap_high_discomfort_10hz_clean.csv")
    df_10hz_normalized = pd.read_csv("../data/pair_ordered_features_overlap_high_discomfort_10hz_normalized.csv")
    df_10hz_pca = pd.read_csv("../data/pair_ordered_features_overlap_high_discomfort_10hz_pca.csv")

    # Generate all combinations of 4 modalities in Boolean
    modalities_combinations = list(product([True, False], repeat=4)) # ['audio', 'face', 'pose', 'sensor']
    modalities_combinations = [comb for comb in modalities_combinations if any(comb)] # remove all False combination
    
    # Sweep configuration
    sweep_config = {
        'method': 'random',
        'name': 'random_forest_tuning',
        'parameters': {
            'feature_set_tag': {'values': ['Full', 'Stat']}, # Full, Stat, RF, Quali
            'dataset': {'values': ['clean', 'normalized', 'pca']},
            'n_estimators': {'values': [100, 200, 300, 500, 700, 1000]},
            'max_depth': {'values': [5, 10, 15, 20, 25, 30]},
            'modalities_combination': {'values': modalities_combinations}
        }
    }
        
    print(sweep_config)
    
    # Start the sweep
    sweep_id = wandb.sweep(sweep=sweep_config, project="SocialDisc-RF")
    wandb.agent(sweep_id, function=train)



if __name__ == '__main__':
    main()