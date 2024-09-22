import os
import cv2
import json
import numpy as np
from libs.dataset.data_splitter import DataSplitter
from libs.utils import print_and_log, load_gt, load_frame_time

def check_data_integrity(data_piece):
    # print(np.sum(np.sum(np.abs(data_piece) < 1e-3, axis=1) == data_piece.shape[1]))
    return True or (np.sum(np.sum(np.abs(data_piece) < 1e-3, axis=1) == data_piece.shape[1]) == 0)

def filter_truth(all_truth, is_train=False):
    filtered_truth = []
    last_truth = None
    for truth in all_truth:
        if isinstance(truth[0], str) and len(truth[0].split()) == 1 and int(truth[0]) < 0:
            continue
        if isinstance(truth[0], str) and (len(truth[0].split()) > 1):# or int(truth[0]) >= 10:
            continue
        # if truth[3][0] == 'a':
        #     continue
        # if not is_train:
        filtered_truth += [truth]
        if is_train and last_truth is not None and float(truth[1]) - float(last_truth[2]) < 0.5:
            combined_label = last_truth[0] + ' ' + truth[0]
            combined_text = last_truth[3] + ' ' + truth[3]
            filtered_truth += [(combined_label, last_truth[1], truth[2], combined_text)]
        # last_truth = truth
    return filtered_truth


def read_from_folder(data_file, truth_file, config_file, input_config, is_train=False):
    data_pairs = []
    loaded_gt = []
    session_config = json.load(open(config_file, 'rt'))

    gt_syncing_ts = session_config['ground_truth_sync']
    video_syncing_ts = session_config['video_sync']
    
    cap = cv2.VideoCapture(data_file)
    video_ts = load_frame_time(data_file[:-4] + '_ts.txt')
    gt = np.load(truth_file)

    gt[:, 1:-3] *= 1000

    def video_ts_to_gt(t):
        gt_t = t - video_syncing_ts + gt_syncing_ts
        gt_idx = np.argmin(np.abs(gt_t - gt[:, 0]))
        return gt[gt_idx]

    grabbed = True
    n_frames = 0
    while grabbed and n_frames < len(video_ts):
        grabbed, img = cap.read()
        if not grabbed:
            continue
        # print(n_frames)
        this_ts = video_ts[n_frames]
        n_frames += 1
        if this_ts < video_syncing_ts + 5:
            continue
        sample_gt = video_ts_to_gt(this_ts)
        if not -0.1 < sample_gt[0] - this_ts < 0.1:
            continue
        # print(n_frames, this_ts, sample_gt[0])
        # input()
        # sample_gt[1:-3] *= 1000
        # scaled_gt = sample_gt[1:] * 1000
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, input_config['input_size'])
        loaded_gt += [sample_gt]
        data_pairs += [(img[None, ...].astype(np.float32), sample_gt[1:].astype(np.float32))]
    
    return data_pairs, loaded_gt

def generate_data(input_config, data_config, is_train):
    '''
     ------------- You should code here to load the data from disk to the program-------------------------
     you should read the training data into the 'train_data', a list:
     step1: read image: img
     step2: save the img and ground-truth in the list 'train_data += [(img, ground-truth)]'

    '''

    # train_data = pickle.load(open('tmp_train_data.pkl', 'rb'))
    # test_data = pickle.load(open('tmp_test_data.pkl', 'rb'))
    # test_loaded_gt = pickle.load(open('tmp_test_loaded_gt.pkl', 'rb'))
    
    train_data = []
    test_data = []

    if is_train:
        print_and_log('Loading training data...')
        for p in data_config['train_sessions']:
            data_file = os.path.join(data_config['root_folder'], p, data_config['data_file'])
            truth_file = os.path.join(data_config['root_folder'], p, data_config['truth_file'])
            config_file = os.path.join(data_config['root_folder'], p, data_config['config_file'])
            print_and_log('Loading from %s' % data_file)
            this_train_data, _ = read_from_folder(data_file, truth_file, config_file, input_config, is_train=True)
            train_data += this_train_data

    print_and_log('Loading testing data...')
    test_loaded_gt = []
    for p in data_config['test_sessions']:
        data_file = os.path.join(data_config['root_folder'], p, data_config['data_file'])
        truth_file = os.path.join(data_config['root_folder'], p, data_config['truth_file'])
        config_file = os.path.join(data_config['root_folder'], p, data_config['config_file'])
        print_and_log('Loading from %s' % data_file)
        this_test_data, this_loaded_gt = read_from_folder(data_file, truth_file, config_file, input_config)
        test_data += this_test_data
        test_loaded_gt += this_loaded_gt

    input_config['model_input_channels'] = test_data[0][0].shape[0]

    # tmp, save the loading process
    # pickle.dump(train_data, open('tmp_train_data.pkl', 'wb'))
    # pickle.dump(test_data, open('tmp_test_data.pkl', 'wb'))
    # pickle.dump(test_loaded_gt, open('tmp_test_loaded_gt.pkl', 'wb'))
    data_splitter = DataSplitter(train_data, test_data, input_config['batch_size'], input_config['num_workers'], data_config['shuffle'], input_config)
    return data_splitter.train_loader, data_splitter.test_loader, test_loaded_gt
