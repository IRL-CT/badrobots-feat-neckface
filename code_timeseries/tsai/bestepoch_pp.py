import wandb
import numpy as np
import pandas as pd
import random
#from sklearn.model_selection import KFold

#import tensorflow as tf
from get_metrics import get_metrics
#import tsai


# # **************** UNCOMMENT AND RUN THIS CELL IF YOU NEED TO INSTALL/ UPGRADE TSAI & SKTIME ****************
# stable = True # Set to True for latest pip version or False for main branch in GitHub
#!pip install {"tsai -U" if stable else "git+https://github.com/timeseriesAI/tsai.git"} >> /dev/null
#!pip install sktime -U  >> /dev/null

import tsai

from tsai.basics import *
#import sktime
import sklearn
#my_setup(sktime, sklearn)
from tsai.models.MINIROCKET import *
from create_data_splits import create_data_splits, create_data_splits_ids

import datetime
from sklearn.model_selection import ParameterGrid
from TimeSeries_Helpers import *

#wandb waittime
nan = np.nan

param_grid_1_1 = {'model': ['LSTM_FCN'], 'n_epoch': [293], 'dropout_LSTM_FCN': [0.2], 'fc_dropout_LSTM_FCN': [0.5], 'n_estimators': [nan], 'stride_train': [1], 'stride_eval': [1], 'lr': [0.0002], 'focal_loss': [False], 'interval_length': [1], 'context_length': [0], 'oversampling': [False], 'batch_size': [128], 'batch_tfms': [None], 'dataset_processing': ['n.a.'], 'participant_minimum': [0.05]} 
param_grid_1_2 = {'model': ['LSTM_FCN'], 'n_epoch': [293], 'dropout_LSTM_FCN': [0.2], 'fc_dropout_LSTM_FCN': [0.5], 'n_estimators': [nan], 'stride_train': [1], 'stride_eval': [1], 'lr': [0.0002], 'focal_loss': [False], 'interval_length': [1], 'context_length': [0], 'oversampling': [False], 'batch_size': [128], 'batch_tfms': [None], 'dataset_processing': ['n.a.'], 'participant_minimum': [0.1]} 
param_grid_1_3 = {'model': ['LSTM_FCN'], 'n_epoch': [293], 'dropout_LSTM_FCN': [0.2], 'fc_dropout_LSTM_FCN': [0.5], 'n_estimators': [nan], 'stride_train': [1], 'stride_eval': [1], 'lr': [0.0002], 'focal_loss': [False], 'interval_length': [1], 'context_length': [0], 'oversampling': [False], 'batch_size': [128], 'batch_tfms': [None], 'dataset_processing': ['n.a.'], 'participant_minimum': [0.25]} 
param_grid_1_4 = {'model': ['LSTM_FCN'], 'n_epoch': [293], 'dropout_LSTM_FCN': [0.2], 'fc_dropout_LSTM_FCN': [0.5], 'n_estimators': [nan], 'stride_train': [1], 'stride_eval': [1], 'lr': [0.0002], 'focal_loss': [False], 'interval_length': [1], 'context_length': [0], 'oversampling': [False], 'batch_size': [128], 'batch_tfms': [None], 'dataset_processing': ['n.a.'], 'participant_minimum': [0.45]} 

param_grid_2_1 = {'model': ['GRU_FCN'], 'n_epoch': [293], 'dropout_LSTM_FCN': [0.2], 'fc_dropout_LSTM_FCN': [0], 'n_estimators': [nan], 'stride_train': [1], 'stride_eval': [1], 'lr': [0.01], 'focal_loss': [True], 'interval_length': [1], 'context_length': [0], 'oversampling': [False], 'batch_size': [128], 'batch_tfms': [None], 'dataset_processing': ['n.a.'], 'participant_minimum': [0.05]}
param_grid_2_2 = {'model': ['GRU_FCN'], 'n_epoch': [293], 'dropout_LSTM_FCN': [0.2], 'fc_dropout_LSTM_FCN': [0], 'n_estimators': [nan], 'stride_train': [1], 'stride_eval': [1], 'lr': [0.01], 'focal_loss': [True], 'interval_length': [1], 'context_length': [0], 'oversampling': [False], 'batch_size': [128], 'batch_tfms': [None], 'dataset_processing': ['n.a.'], 'participant_minimum': [0.1]}
param_grid_2_3 = {'model': ['GRU_FCN'], 'n_epoch': [293], 'dropout_LSTM_FCN': [0.2], 'fc_dropout_LSTM_FCN': [0], 'n_estimators': [nan], 'stride_train': [1], 'stride_eval': [1], 'lr': [0.01], 'focal_loss': [True], 'interval_length': [1], 'context_length': [0], 'oversampling': [False], 'batch_size': [128], 'batch_tfms': [None], 'dataset_processing': ['n.a.'], 'participant_minimum': [0.25]}
param_grid_2_4 = {'model': ['GRU_FCN'], 'n_epoch': [293], 'dropout_LSTM_FCN': [0.2], 'fc_dropout_LSTM_FCN': [0], 'n_estimators': [nan], 'stride_train': [1], 'stride_eval': [1], 'lr': [0.01], 'focal_loss': [True], 'interval_length': [1], 'context_length': [0], 'oversampling': [False], 'batch_size': [128], 'batch_tfms': [None], 'dataset_processing': ['n.a.'], 'participant_minimum': [0.45]}

param_grid_3_1 = {'model': ['gMLP'], 'n_epoch': [293], 'dropout_LSTM_FCN': [0.8], 'fc_dropout_LSTM_FCN': [0.8], 'n_estimators': [nan], 'stride_train': [1], 'stride_eval': [1], 'lr': [0.01], 'focal_loss': [True], 'interval_length': [1], 'context_length': [0], 'oversampling': [False], 'batch_size': [256], 'batch_tfms': [None], 'dataset_processing': ['n.a.'], 'participant_minimum': [0.05]}
param_grid_3_2 = {'model': ['gMLP'], 'n_epoch': [293], 'dropout_LSTM_FCN': [0.8], 'fc_dropout_LSTM_FCN': [0.8], 'n_estimators': [nan], 'stride_train': [1], 'stride_eval': [1], 'lr': [0.01], 'focal_loss': [True], 'interval_length': [1], 'context_length': [0], 'oversampling': [False], 'batch_size': [256], 'batch_tfms': [None], 'dataset_processing': ['n.a.'], 'participant_minimum': [0.1]}
param_grid_3_3 = {'model': ['gMLP'], 'n_epoch': [293], 'dropout_LSTM_FCN': [0.8], 'fc_dropout_LSTM_FCN': [0.8], 'n_estimators': [nan], 'stride_train': [1], 'stride_eval': [1], 'lr': [0.01], 'focal_loss': [True], 'interval_length': [1], 'context_length': [0], 'oversampling': [False], 'batch_size': [256], 'batch_tfms': [None], 'dataset_processing': ['n.a.'], 'participant_minimum': [0.25]}
param_grid_3_4 = {'model': ['gMLP'], 'n_epoch': [293], 'dropout_LSTM_FCN': [0.8], 'fc_dropout_LSTM_FCN': [0.8], 'n_estimators': [nan], 'stride_train': [1], 'stride_eval': [1], 'lr': [0.01], 'focal_loss': [True], 'interval_length': [1], 'context_length': [0], 'oversampling': [False], 'batch_size': [256], 'batch_tfms': [None], 'dataset_processing': ['n.a.'], 'participant_minimum': [0.45]}

param_grid_4_1 = {'model': ['InceptionTimePlus'], 'n_epoch': [419], 'dropout_LSTM_FCN': [0.5], 'fc_dropout_LSTM_FCN': [0.2], 'n_estimators': [nan], 'stride_train': [5], 'stride_eval': [1], 'lr': [0.01], 'focal_loss': [False], 'interval_length': [25], 'context_length': [0], 'oversampling': [False], 'batch_size': [128], 'batch_tfms': [None], 'dataset_processing': ['pca'], 'participant_minimum': [0.05]}
param_grid_4_2 = {'model': ['InceptionTimePlus'], 'n_epoch': [419], 'dropout_LSTM_FCN': [0.5], 'fc_dropout_LSTM_FCN': [0.2], 'n_estimators': [nan], 'stride_train': [5], 'stride_eval': [1], 'lr': [0.01], 'focal_loss': [False], 'interval_length': [25], 'context_length': [0], 'oversampling': [False], 'batch_size': [128], 'batch_tfms': [None], 'dataset_processing': ['pca'], 'participant_minimum': [0.1]}
param_grid_4_3 = {'model': ['InceptionTimePlus'], 'n_epoch': [419], 'dropout_LSTM_FCN': [0.5], 'fc_dropout_LSTM_FCN': [0.2], 'n_estimators': [nan], 'stride_train': [5], 'stride_eval': [1], 'lr': [0.01], 'focal_loss': [False], 'interval_length': [25], 'context_length': [0], 'oversampling': [False], 'batch_size': [128], 'batch_tfms': [None], 'dataset_processing': ['pca'], 'participant_minimum': [0.25]}
param_grid_4_4 = {'model': ['InceptionTimePlus'], 'n_epoch': [419], 'dropout_LSTM_FCN': [0.5], 'fc_dropout_LSTM_FCN': [0.2], 'n_estimators': [nan], 'stride_train': [5], 'stride_eval': [1], 'lr': [0.01], 'focal_loss': [False], 'interval_length': [25], 'context_length': [0], 'oversampling': [False], 'batch_size': [128], 'batch_tfms': [None], 'dataset_processing': ['pca'], 'participant_minimum': [0.45]}

param_grid_5_1 = {'model': ['InceptionTime'], 'n_epoch': [350], 'dropout_LSTM_FCN': [0.2], 'fc_dropout_LSTM_FCN': [0], 'n_estimators': [nan], 'stride_train': [5], 'stride_eval': [1], 'lr': [0.01], 'focal_loss': [False], 'interval_length': [12], 'context_length': [0], 'oversampling': [False], 'batch_size': [256], 'batch_tfms': [None], 'dataset_processing': ['n.a.'], 'participant_minimum': [0.05]}
param_grid_5_2 = {'model': ['InceptionTime'], 'n_epoch': [350], 'dropout_LSTM_FCN': [0.2], 'fc_dropout_LSTM_FCN': [0], 'n_estimators': [nan], 'stride_train': [5], 'stride_eval': [1], 'lr': [0.01], 'focal_loss': [False], 'interval_length': [12], 'context_length': [0], 'oversampling': [False], 'batch_size': [256], 'batch_tfms': [None], 'dataset_processing': ['n.a.'], 'participant_minimum': [0.1]}
param_grid_5_3 = {'model': ['InceptionTime'], 'n_epoch': [350], 'dropout_LSTM_FCN': [0.2], 'fc_dropout_LSTM_FCN': [0], 'n_estimators': [nan], 'stride_train': [5], 'stride_eval': [1], 'lr': [0.01], 'focal_loss': [False], 'interval_length': [12], 'context_length': [0], 'oversampling': [False], 'batch_size': [256], 'batch_tfms': [None], 'dataset_processing': ['n.a.'], 'participant_minimum': [0.25]}
param_grid_5_4 = {'model': ['InceptionTime'], 'n_epoch': [350], 'dropout_LSTM_FCN': [0.2], 'fc_dropout_LSTM_FCN': [0], 'n_estimators': [nan], 'stride_train': [5], 'stride_eval': [1], 'lr': [0.01], 'focal_loss': [False], 'interval_length': [12], 'context_length': [0], 'oversampling': [False], 'batch_size': [256], 'batch_tfms': [None], 'dataset_processing': ['n.a.'], 'participant_minimum': [0.45]}

#merge both
param_grid = [param_grid_1_1, param_grid_1_2, param_grid_1_3, param_grid_1_4, 
param_grid_2_1, param_grid_2_2, param_grid_2_3, param_grid_2_4, 
param_grid_3_1, param_grid_3_2, param_grid_3_3, param_grid_3_4, 
param_grid_4_1, param_grid_4_2, param_grid_4_3, param_grid_4_4, 
param_grid_5_1, param_grid_5_2, param_grid_5_3, param_grid_5_4]

print(param_grid)

param_grid = list(ParameterGrid(param_grid))

#print length
print("\n -----------------------\n Number of interations",
      len(param_grid), "x 5", "\n -----------------------")


df_name = 'training_data.csv'
#df_full = pd.read_csv('../../data/' + df_name)
#features = df_full.columns[4:]
#print('FEATURES', features)



print("\n -----------------------\n Number of interations",
      len(param_grid), "x 5", "\n -----------------------")


#seed randomizer
#random.seed(42)
#np.random.seed(42)
#randomize param_grid
#random.shuffle(param_grid)


for i, grid_config in enumerate(param_grid):
    if i >= 0:
        if True:
            print("Round:", i+1, "of", len(param_grid))
            print(grid_config)
            config = AttrDict(
                df_name = df_name,
                merged_labels=False,
                threshold=80,
                interval_length=grid_config["interval_length"],
                stride_train=grid_config["stride_train"],
                stride_eval=grid_config["stride_eval"],
                context_length=grid_config['context_length'],
                train_ids=[],
                valid_ids=[],
                test_ids=[],
                use_lvl1=True,
                use_lvl2=False,
                model=grid_config["model"],
                lr=grid_config["lr"],
                n_epoch=grid_config["n_epoch"],
                dropout_LSTM_FCN=grid_config["dropout_LSTM_FCN"],
                fc_dropout_LSTM_FCN=grid_config["fc_dropout_LSTM_FCN"],
                batch_tfms=grid_config["batch_tfms"],
                batch_size=grid_config["batch_size"],
                focal_loss=grid_config["focal_loss"],
                n_estimators=grid_config["n_estimators"],
                #features = features,
                oversampling=grid_config["oversampling"],
                undersampling=False,
                verbose=True,
                dataset = "neckface",
                dataset_processing = grid_config["dataset_processing"],
                participant_minimum = grid_config["participant_minimum"],
                # required by the current TimeSeries_Helpers: use the facial features only (no ViT features)
                feature_set_tag = "Full",
                modalities_combination = (True, False),
                

            )

            cross_validate_pp(val_fold_size=5, config=config,
                        group="all_bestepoch", name=str(grid_config))






