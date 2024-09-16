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

#ADJUST TO BEST PERFORMING
param_grid_1 = {'model': ['gMLP'], 'n_epoch': [500], 'dropout_LSTM_FCN': [nan], 'fc_dropout_LSTM_FCN': [nan], 'n_estimators': [nan], 'stride_train': [30], 'stride_eval': [30], 'lr': [0.0002], 'focal_loss': [False], 'interval_length': [5], 'context_length': [0], 'oversampling': [False], 'batch_size': [64], 'batch_tfms': [None], 'dataset_processing': [nan]}

param_grid_2 = {'model': ['gMLP'], 'n_epoch': [500], 'dropout_LSTM_FCN': [nan], 'fc_dropout_LSTM_FCN': [nan], 'n_estimators': [nan], 'stride_train': [30], 'stride_eval': [30], 'lr': [0.0002], 'focal_loss': [False], 'interval_length': [5], 'context_length': [0], 'oversampling': [False], 'batch_size': [64], 'batch_tfms': [None], 'dataset_processing': [nan]}

param_grid_3 = {'model': ['gMLP'], 'n_epoch': [500], 'dropout_LSTM_FCN': [nan], 'fc_dropout_LSTM_FCN': [nan], 'n_estimators': [nan], 'stride_train': [30], 'stride_eval': [5], 'lr': [0.0002], 'focal_loss': [False], 'interval_length': [5], 'context_length': [0], 'oversampling': [False], 'batch_size': [64], 'batch_tfms': [None], 'dataset_processing': [nan]}

param_grid_4 = {'model': ['gMLP'], 'n_epoch': [500], 'dropout_LSTM_FCN': [nan], 'fc_dropout_LSTM_FCN': [nan], 'n_estimators': [nan], 'stride_train': [30], 'stride_eval': [5], 'lr': [0.001], 'focal_loss': [False], 'interval_length': [5], 'context_length': [0], 'oversampling': [False], 'batch_size': [64], 'batch_tfms': [None], 'dataset_processing': [nan]}

param_grid_5 = {'model': ['gMLP'], 'n_epoch': [500], 'dropout_LSTM_FCN': [nan], 'fc_dropout_LSTM_FCN': [nan], 'n_estimators': [nan], 'stride_train': [30], 'stride_eval': [5], 'lr': [0.001], 'focal_loss': [False], 'interval_length': [5], 'context_length': [0], 'oversampling': [False], 'batch_size': [64], 'batch_tfms': [None], 'dataset_processing': [nan]}

#get the 5 grids into a list of parameter grids
#param_grid = list([ParameterGrid(param_grid_1), ParameterGrid(param_grid_2), ParameterGrid(param_grid_3), ParameterGrid(param_grid_4), ParameterGrid(param_grid_5)])
param_grid = [param_grid_1, param_grid_2, param_grid_3, param_grid_4, param_grid_5]

#param_grid = list(ParameterGrid(param_grid))


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
                interval_length=grid_config["interval_length"][0],
                stride_train=grid_config["stride_train"][0],
                stride_eval=grid_config["stride_eval"][0],
                context_length=grid_config['context_length'][0],
                train_ids=[],
                valid_ids=[],
                test_ids=[],
                use_lvl1=True,
                use_lvl2=False,
                model=grid_config["model"][0],
                lr=grid_config["lr"][0],
                n_epoch=grid_config["n_epoch"][0],
                dropout_LSTM_FCN=grid_config["dropout_LSTM_FCN"][0],
                fc_dropout_LSTM_FCN=grid_config["fc_dropout_LSTM_FCN"][0],
                batch_tfms=grid_config["batch_tfms"][0],
                batch_size=grid_config["batch_size"][0],
                focal_loss=grid_config["focal_loss"][0],
                n_estimators=grid_config["n_estimators"][0],
                #features = features,
                oversampling=grid_config["oversampling"][0],
                undersampling=False,
                verbose=True,
                dataset = "openface",
                dataset_processing = grid_config["dataset_processing"][0],
                model_numbering = i
            )

            cross_validate_bestepoch(val_fold_size=5, config=config,
                        group="all", name=str(grid_config))






