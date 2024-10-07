from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import json
import os
import numpy as np

DEFAULTS = {
    # dataset loader, specify the dataset here
    'dataset': {
        'path': '/data/acoustic_silentspeech/pilot_study/0221_10digits3', # where all datasets are stored
        'train_sessions': ['session_011','session_012','session_013','session_014','session_021','session_022','session_023','session_024','session_031','session_032','session_033','session_034','session_041','session_042','session_043','session_044'],
        'test_sessions': ['session_015','session_025','session_035','session_045'],
        'data_file': 'video.avi',                   # X
        'truth_file': 'ground_truth.npy',       # y
        # 'truth_file': 'ground_truth_regression.npy',       # y
        # 'truth_file': 'ground_truth_landmarks.npy',       # y
        # 'truth_file': 'ground_truth_blendshapes.npy',       # y
        'config_file': 'config.json',
        'shuffle': True,
        'input_views': 1,
    },
    # input pipeline, specify data augmentations here
    'input': {
        'input_size': (640, 240),
        'augment_affine': True,
        'affine_parameters': {
            # ((min_scale_x, min_scale_y), (max_scale_x, max_scale_y), rate)
            'scale': ((0.9, 0.9), (1.1, 1.1), 0.5),
            # (angle in degree, rate)
            'rotate': (8, 0.5),
            # ((min_move_x, min_move_y), (max_move_x, max_move_y), rate)
            'move': ((-6, -6), (6, 6), 0.5)
        },
        # disable everything except color aug
        'color_jitter': 0.01,  # color pertubation (if <=0 disabled)
        # param for train/val/test (640*480 -> 384*288)
        'scale_train': 288,  # If -1 do not scale
        'scale_val': 288,  # If -1 do not scale
        'scale_test': 288,
        # common params (from ImageNet)
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
        # num of workers for data loader
        'num_workers': 0,
        # subsample the training set (nearby video frames are similar)
        # 'sample_interval': 10,
        # batch size
        'batch_size': 20,  # must be N x #GPUs
    },
    # network architecture
    'network': {
        # multi gpu support
        'visible_devices': '0',
        'devices': [0],         # default: single gpu
        'backbone': 'resnet18',  # backbone network
        'pretrained': False,     # if the backbone is pre-trained
        'output_dims': 5,
        'frozen_stages': [],     # freeze part of the network
        'decoder': 'catfc',     # decoder for classification
        'feat_dim': 512,        # input feat dim to decoder
        'decoder_fc_std': 0.02,  # init fc std for decoder
        'decoder_fc_bias': None,  # bias init for decoder fc bias
        'dropout_prob': 0.5,    # dropout ratio for fc in decoder
        'loss_type': 'l2',   # which loss to use (l1 / l2 / huber / dis / frame_weighted / 4060 / ce (for classification))
        'output': 1,
        'human': 1,
        "if_weight_4060": 0,
        "if_weight_loss": 0,
        "if_weight": 0,  # weighted average for different points. For example, eye
        "if_weight_mouth": 0,
        'rnn': {
            'rnn_type': 'gru',
            # 'sequence_length': 35,
            'input_size': 256 * 4,
            'hidden_size': 128,
            'num_layers': 2,
            # 'num_classes': 8,
            # 'batch_size': 30,
            # 'num_epochs': 30,
            # 'learning_rate': 0.001,
            'random_sample': 1,
            'slide_step': 1,
            'bidirectional': True,
            'cat_backbone': False,       # True: at the end of the backbone
        },
    },
    # optimizer (for training)
    'optimizer': {
        'type': 'Adam',      # SGD or Adam
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'learning_rate': 0.0002,  # i gradient descent
        'decoder_lr_ratio': 1.0,
        'nesterov': True,
        'epochs': 30,    # change (traning cycle)
        # lr scheduler (cosine/multistep/etc)
        'schedule': {
            'type': 'cosine',
            'steps': [],  # in #epochs
            'gamma': 0.1  # used in multi-step decay
        },
    },
}


def _merge(src, dst):
    for k, v in src.items():
        if k in dst:
            if isinstance(v, dict):
                _merge(src[k], dst[k])
        else:
            dst[k] = v


def load_config(defaults=DEFAULTS):
    config = defaults

    config['network']['input_views'] = config['dataset']['input_views']

    if config['network']['loss_type'] == 'frame_weighted':
        raw_bp = np.load(config['dataset']['root_folder'] + '/train/silent_pos.npy')
        bp = raw_bp[:40] - [np.mean(raw_bp[range(0, 40, 2)]), np.mean(raw_bp[range(1, 40, 2)])] * 20
        config['network']['silent_pos'] = bp

    if config['network']['backbone'].lower() in ['lstm', 'gru']:
        config['network']['feat_dim'] = config['network']['rnn']['hidden_size'] * (config['network']['rnn']['bidirectional'] + 1)
        config['network']['rnn']['input_size'] = (config['input']['pixels_of_interest'][1] - config['input']['pixels_of_interest'][0]) * 1#config['input']['channels']

    # config['network']['weight'] = [2, 2, 2, 3, 3, 3, 2, 3, 3, 3, 4, 4, 4, 4, 4, 6, 4, 4, 4, 4, 4, 6]
    # config['network']['weighted_points'] = [0,6,12,13,14,15,16,17,18,19,  23,24,25,28,30,32,34, 35,36,40,44,45]
    
    config["network"]["weight"] = [3] * 23
    config["network"]["weighted_points"] = [18, 19, 20, 21, 22, 23, 24,
                                            25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]

    config['dataset']['root_folder'] = os.path.join(config['dataset']['path'], 'dataset')

    return config
