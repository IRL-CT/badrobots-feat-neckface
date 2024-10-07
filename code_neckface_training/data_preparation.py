'''
Pre-process data and generate dataset for training
8/2/2022, Ruidong Zhang, rz379@cornell.edu
'''

import os
import re
import json
import warnings
import argparse
import numpy as np

from utils import load_frame_time
from parse_rcv import parse_rcv
from load_save_gt import load_gt

def load_config(parent_folder, config_file):
    if len(config_file) == 0:
        config_file = os.path.join(parent_folder, 'config.json')
    if not os.path.exists(config_file):
        raise FileNotFoundError('Config file at %s not found' % config_file)
    config = json.load(open(config_file, 'rt'))
    assert(len(config['video']['files']) == len(config['video']['syncing_poses']))  # every video file should have a sync mark
    assert(len(config['ground_truth']['files']) == len(config['ground_truth']['syncing_poses']))  # every video file should have a sync mark
    assert(len(config['video']['files']) == len(config['ground_truth']['files']))   # at this moment, video files and gt files should have same amount

    for f in config['ground_truth']['files']:
        if not os.path.exists(os.path.join(parent_folder, f)):
            raise FileNotFoundError('Ground truth file at %s not found' % f)        # ground truth file must exist

    return config

def ts_to_idx(ts, all_ts):
    return np.argmin(np.abs(all_ts - ts))

def data_preparation(parent_folder, config_file=''):
    
    config = load_config(parent_folder, config_file)

    n_sessions = len(config['video']['files'])

    for n in range(n_sessions):
        print('Dealing with session %02d, ' % (n + 1), end='')
        session_target = os.path.join(parent_folder, 'dataset', 'session_%02d' % (n + 1))
        # if not os.path.exists(session_target):
        #     os.makedirs(session_target)
        video_file = config['video']['files'][n]
        video_sync = config['video']['syncing_poses'][n]
        gt_file = config['ground_truth']['files'][n]
        gt_sync = config['ground_truth']['syncing_poses'][n]


        # parsing ground truth file if not already done so
        if not os.path.exists(os.path.join(parent_folder, gt_file + '.npy')):
            print('Parsing ground truth file %s' % gt_file)
            parse_rcv(os.path.join(parent_folder, gt_file), npy=True)
        # gt_data = load_gt(os.path.join(parent_folder, gt_file + '.npy'))
        # session_gt = gt_data[gt_data[:, 0] > gt_sync + 5]       # 5s after the clapping

        gt_data = load_gt(os.path.join(parent_folder, gt_file + '.npy'))
        if 0 <= video_sync < 1600000000:
            session_video_ts = load_frame_time(os.path.join(parent_folder, video_file[:-4] + '.txt'))
            video_sync = session_video_ts[video_sync]
        if 0 <= gt_sync < 1600000000:
            gt_data = load_gt(os.path.join(parent_folder, gt_file + '.npy'))
            gt_sync = gt_data[gt_sync, 0]
            
        session_config = {
            'video_sync': video_sync,
            'ground_truth_sync': gt_sync
        }

        # source_paths = [os.path.join('../..', gt_file + '.npy'), os.path.join('../..', video_file), os.path.join('../..', video_file[:-4] + '.txt')]
        # target_paths = [session_gt_target, session_video_target, session_video_ts_target]

        source_paths = [os.path.join('../..', video_file), os.path.join('../..', video_file[:-4] + '.txt')]

        n_splits = 5

        for sp in range(n_splits):
            sp_gt = gt_data[int(len(gt_data) * sp / n_splits):int(len(gt_data) * (sp + 1) / n_splits)]
        

        # session_config_target = os.path.join(session_target, 'config.json')
        # session_gt_target = os.path.join(session_target, 'ground_truth.npy')
        # session_video_target = os.path.join(session_target, 'video' + video_file[-4:])
        # session_video_ts_target = os.path.join(session_target, 'video_ts.txt')
        # target_paths = [session_video_target, session_video_ts_target]

        # np.save(session_gt_target, sp_gt)

        # json.dump(session_config, open(session_config_target, 'wt'), indent=4)
        # for s, t in zip(source_paths, target_paths):
        #     if not os.path.exists(t):
        #         os.symlink(s, os.path.abspath(t))
        #         print('Linking %s -> %s' % (s, t))

        # testing session
            sp_session_target = session_target + '%02d' % (sp + 1)
            if not os.path.exists(sp_session_target):
                os.makedirs(sp_session_target)
            session_config_target = os.path.join(sp_session_target, 'config.json')
            session_gt_target = os.path.join(sp_session_target, 'ground_truth.npy')
            session_video_target = os.path.join(sp_session_target, 'video' + video_file[-4:])
            session_video_ts_target = os.path.join(sp_session_target, 'video_ts.txt')
            target_paths = [session_video_target, session_video_ts_target]

            np.save(session_gt_target, sp_gt)
            json.dump(session_config, open(session_config_target, 'wt'), indent=4)
            for s, t in zip(source_paths, target_paths):
                if not os.path.exists(t):
                    os.symlink(s, os.path.abspath(t))
                    print('Linking %s -> %s' % (s, t))
        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('Pre-process data and generate dataset for training')
    parser.add_argument('-p', '--path', help='path to the folder where files are saved (and dataset will be saved)')
    parser.add_argument('-c', '--config', help='path to the config.json file', default='')

    args = parser.parse_args()
    data_preparation(args.path, args.config)
