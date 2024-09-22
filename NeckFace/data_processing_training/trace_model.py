from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from cgi import test
from pprint import pprint
from libs.utils import (AverageMeter, save_checkpoint, IntervalSampler,
                        create_optim, create_scheduler, print_and_log, save_gt, generate_cm, extract_labels)
from libs.models import EncoderDecoder as ModelBuilder
from libs.models import Point_dis_loss, calculate_dis, get_criterion, wer_sliding_window
from libs.core import load_config
from libs.dataset import generate_data  # , gen_test
import torch.multiprocessing as mp
import torch.utils.data
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch
import random
import numpy as np
import pandas as pd
import logging
from torch.utils.mobile_optimizer import optimize_for_mobile


# python imports
import argparse
import os
import re
import time
import math
import cv2
import pickle

# control the number of threads (to prevent blocking ...)
# should work for both numpy / opencv
# it really depends on the CPU / GPU ratio ...
TARGET_NUM_THREADS = '4'
os.environ['OMP_NUM_THREADS'] = TARGET_NUM_THREADS
os.environ['OPENBLAS_NUM_THREADS'] = TARGET_NUM_THREADS
os.environ['MKL_NUM_THREADS'] = TARGET_NUM_THREADS
os.environ['VECLIB_MAXIMUM_THREADS'] = TARGET_NUM_THREADS
os.environ['NUMEXPR_NUM_THREADS'] = TARGET_NUM_THREADS
# os.environ['CUDA_VISIBLE_DEVICES'] = GPUs.select()
# numpy imports
# torch imports

# for visualization
# from torch.utils.tensorboard import SummaryWriter


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# the arg parser
parser = argparse.ArgumentParser(description='Hand pose from mutliple views')
parser.add_argument('--print-freq', default=30, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('-v', '--valid-freq', default=3, type=int,
                    help='validation frequency (default: 5)')
parser.add_argument('-o', '--output', default='temp', type=str,
                    help='the name of output file')
parser.add_argument('-i', '--input', default='', type=str,
                    help='overwrites dataset.data_file')
parser.add_argument('--stacking', default='', choices=['vertical', 'channel', ''], type=str,
                    help='overwrites input.stacking')
parser.add_argument('-p', '--path', default='', type=str,
                    help='path to dataset parent folder, overwrites dataset.path')
parser.add_argument('-ts', '--test-sessions', default='', type=str,
                    help='overwrites dataset.test_sessions, comma separated, e.g. 5,6,7')
parser.add_argument('--exclude-sessions', default='', type=str,
                    help='remove these sessions from training AND testing, comma separated, e.g. 5,6,7')
parser.add_argument('--train-sessions', default='', type=str,
                    help='overwrites dataset.train_sessions, default using all but testing sessions for training, comma separated, e.g. 5,6,7')
parser.add_argument('-g', '--visible-gpu', default='', type=str,
                    help='visible gpus, comma separated, e.g. 0,1,2 (overwrites config file)')
# parser.add_argument('-m', '--mode', default='', type=str,
#                     help='the mode of training')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--epochs', default=-1, type=int,
                    help='total epochs to run, overwrites optimizer.epochs')
parser.add_argument('--epochs-to-run', default=-1, type=int,
                    help='how many epochs left to run, overwrites --epochs')
parser.add_argument('--lr', default=0, type=float,
                    help='learning rate, overwrites optimizer.learning_rate')
parser.add_argument('--bb', default='',
                    help='backbone, overwrites network.backbone')
parser.add_argument('--bn', default=0, type=int,
                    help='batchsize, overwrites input.batch_size')
parser.add_argument('--coi', default='',
                    help='channels of interest, comma-separated, overwrites input.channels_of_interest')
parser.add_argument('-t', '--if_train', default=0, type=int,
                    help='If train this model (default: none)')
# parser.add_argument('-a', '--augment', default=0, type=int,
#                     help='if use the image augment')
# parser.add_argument('--all', action='store_true',
#                     help='use the model to run over training and testing set')
# parser.add_argument('--train_file', nargs='+')

# parser.add_argument('--test_file', nargs='+')

# main function for training and testing

def save_array(pred, loaded_gt, filename, cm):
    if pred is not None:
        save_arr = [(loaded_gt[i][0], loaded_gt[i][1], loaded_gt[i][2], loaded_gt[i][3], pred[i]) for i in range(len(pred))]
        if False and cm:
            truths = [int(x[0]) for x in save_arr]
            preds = [x[4] for x in save_arr]
            labels = extract_labels(loaded_gt)
            generate_cm(np.array(truths), np.array(preds), labels, filename[:-4] + '_cm.png')
    else:
        save_arr = loaded_gt
    save_gt(save_arr, filename)

def main(args):
    # ===================== initialization for training ================================
    print(time.strftime('begin: %Y-%m-%d %H:%M:%S', time.localtime()))
    torch.cuda.empty_cache()
    # parse args
    # best_metric = 100000.0
    metric_text = 'metric'
    args.start_epoch = 0
    
    torch.set_num_threads(int(TARGET_NUM_THREADS))

    config = load_config()  # load the configuration
    #print('Current configurations:')
    # pprint(config)
    #raise KeyboardInterrupt
    # prepare for output folder
    output_dir = args.output
    os.environ['CUDA_VISIBLE_DEVICES'] = config['network']['visible_devices']
    if len(args.visible_gpu):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.visible_gpu
    print_and_log('Using GPU # %s' % os.environ['CUDA_VISIBLE_DEVICES'])
    if len(args.path):
        config['dataset']['path'] = args.path.rstrip('/')
        config['dataset']['root_folder'] = os.path.join(config['dataset']['path'], 'dataset')
        output_dir = os.path.basename(config['dataset']['path'])
    if not args.output == 'temp':
        output_dir = (os.path.basename(config['dataset']['path']) + '_' + args.output).replace('__', '_')
    if len(args.input):
        config['dataset']['data_file'] = args.input
    if len(args.stacking):
        config['input']['stacking'] = args.stacking
    if len(args.test_sessions):
        all_session_names = os.listdir(config['dataset']['root_folder'])
        all_session_names.sort()
        test_sessions = [s for s in args.test_sessions.split(',') if len(s)]
        train_sessions = [s for s in args.train_sessions.split(',') if len(s)]
        exclude_sessions = [s for s in args.exclude_sessions.split(',') if len(s)]
        config['dataset']['train_sessions'] = []
        config['dataset']['test_sessions'] = []
        for ss in all_session_names:
            if args.if_train and re.match(r'session_\w+', ss) is None:
                continue
            session_suffix = re.findall(r'session_(\w+)', ss)[0]
            if session_suffix in test_sessions and session_suffix not in exclude_sessions:
                config['dataset']['test_sessions'] += [ss]
            elif (len(args.train_sessions) == 0 or session_suffix in train_sessions) and session_suffix not in exclude_sessions:
                config['dataset']['train_sessions'] += [ss]
    if args.epochs > 0:
        config['optimizer']['epochs'] = args.epochs
    if args.epochs_to_run > 0:
        config['optimizer']['epochs'] = args.start_epoch + args.epochs_to_run
    if args.lr > 0:
        config['optimizer']['learning_rate'] = args.lr
    if args.bn > 0:
        config['input']['batch_size'] = args.bn
    if len(args.bb) > 0:
        config['network']['backbone'] = args.bb
    if len(args.coi) > 0:
        config['input']['channels_of_interest'] = [int(x) for x in args.coi.split(',')]
    torch.cuda.empty_cache()
    ckpt_folder = os.path.join('./ckpt', output_dir)
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)
    log_path = os.path.join(ckpt_folder, 'logs.txt')
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=log_path, level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')
    logging.info(time.strftime('begin: %Y-%m-%d %H:%M:%S', time.localtime()))
    # use spawn for mp, this will fix a deadlock by OpenCV (do not need)
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')

    # fix the random seeds (the best we can)
    fixed_random_seed = 20220217
    torch.manual_seed(fixed_random_seed)
    np.random.seed(fixed_random_seed)
    random.seed(fixed_random_seed)

    # set up transforms and dataset
    # ===================== packet the training data and testinf data ================================
    train_dataset, val_dataset, test_gt = generate_data(input_config=config['input'], data_config=config['dataset'], is_train=args.if_train)

    if config['network']['loss_type'] == 'ce':  # classification task, determine output dimension based on number of labels
        config['network']['output_dims'] = max([int(x[0]) for x in test_gt]) + 1
        print_and_log('Classification task detected, output dimension: %d' % config['network']['output_dims'])
    elif config['network']['loss_type'] == 'ctc':  # classification task, determine output dimension based on number of labels
        config['network']['output_dims'] = max([max([int(xx) for xx in x[0].split()]) for x in test_gt]) + 2
        print_and_log('Classification task detected, output dimension: %d' % config['network']['output_dims'])
    else:
        config['network']['output_dims'] = test_gt[0].shape[0] - 1
        print_and_log('Regression task detected, output dimension: %d' % config['network']['output_dims'])
    config['network']['input_channels'] = config['input']['model_input_channels']
    print_and_log(time.strftime('finish data: %Y-%m-%d %H:%M:%S', time.localtime()))
    # create model w. loss
    model = ModelBuilder(config['network'])  # load the designed model
    # GPU you will use in training
    master_gpu = config['network']['devices'][0]
    # model = model.cuda(master_gpu)  # load model from CPU to GPU
    # create optimizer
    optimizer = create_optim(model, config['optimizer'])  # gradient descent
    # data parallel
    # if you want use multiple GPU
    # model = nn.DataParallel(model, device_ids=config['network']['devices'])
    logging.info(model)
    # set up learning rate scheduler
    if args.if_train:
        num_iters_per_epoch = len(train_dataset)
        scheduler = create_scheduler(
            optimizer, config['optimizer']['schedule'],
            config['optimizer']['epochs'], num_iters_per_epoch)

    # ============================= retrain the trained model (if need usually not) =========================================
    # resume from a checkpoint?
    if args.resume:
        #args.if_train = 0
        print_and_log('loading trained model.....')
        if os.path.isfile(args.resume):
            print_and_log('=> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(args.resume,
                                    map_location=lambda storage, loc: storage.cuda(master_gpu))
            # args.start_epoch = 0
            args.start_epoch = checkpoint['epoch']
            best_metric = checkpoint['best_metric']
            new_state_dict = {}
            for k in checkpoint['state_dict'].keys():
                new_state_dict[k[7:]] = checkpoint['state_dict'][k]
            model.load_state_dict(new_state_dict)
            if args.epochs_to_run > 0:
                config['optimizer']['epochs'] = args.start_epoch + args.epochs_to_run
            # only load the optimizer if necessary
            if args.if_train:
                scheduler = create_scheduler(
                    optimizer, config['optimizer']['schedule'],
                    config['optimizer']['epochs'], num_iters_per_epoch)
                # optimizer.load_state_dict(checkpoint['optimizer'])
                # scheduler.load_state_dict(checkpoint['scheduler'])
            print_and_log('=> loaded checkpoint {} (epoch {}, metric {:.3f})'
                  .format(args.resume, checkpoint['epoch'], best_metric))
        else:
            print_and_log('=> no checkpoint found at {}'.format(args.resume))
            return

    # training: enable cudnn benchmark
    cudnn.enabled = True
    cudnn.benchmark = True

    validate(val_dataset, model, args, config)

################################ save the file ###########################################

    # if not args.if_train:
    #     metric_test, loss_test, pred_all_test, fake_gts = validate(val_dataset, model, args, config)
    #     if config['network']['loss_type'] == 'ctc' and config['input']['test_sliding_window']['applied']:
    #         # print(pred_all_test[:10])
    #         # print(fake_gts[:10])
    #         metric_test, pred_all_test, _ = wer_sliding_window(pred_all_test, fake_gts, test_gt)
    #     print_and_log('**** Testing loss: %.4f, metric: %.4f ****' % (loss_test, metric_test))
    #     if config['network']['loss_type'] in ['ce', 'ctc']:
    #         save_array(pred_all_test, test_gt, os.path.join(ckpt_folder, 'test_pred.txt'), config['network']['loss_type'] == 'ce')
    #         # for i in range(len(test_gt)):
    #         #     print(test_gt[i][0], pred_all_test[i])
    #         #     if test_gt[i][0] != pred_all_test[i]:
    #         #         print(test_gt[i])
    #         all_acc = np.mean([test_gt[x][0] == pred_all_test[x] for x in range(len(pred_all_test))])
    #         print_and_log('Exact match acc: %.4f' % all_acc)
    #     else:
    #         save_array(None, pred_all_test, os.path.join(ckpt_folder, 'test_pred.npy'), False)
    #         save_array(None, test_gt, os.path.join(ckpt_folder, 'test_gt.npy'), False) 
    #     # save_array(pred_all_test, test_gt, os.path.join(ckpt_folder, 'test_pred.txt'))

    # print_and_log(time.strftime('end: %Y-%m-%d %H:%M:%S', time.localtime()))

def validate(val_loader, model, args, config):
    '''Test the model on the validation set'''
    # set up meters
    batch_time = AverageMeter()
    metrics = AverageMeter()
    losses = AverageMeter()

    # metric_action = AverageMeter()
    # metric_peak = AverageMeter()

    # switch to evaluate mode
    model.eval()
    # master_gpu = config['network']['devices'][0]
    end = time.time()

    # prepare for outputs
    pred_list = []
    # truth_list = []

    loss_type = config['network']['loss_type']
    criterion = get_criterion(loss_type)
    # criterion = get_criterion(loss_type)
    # criterion = Point_dis_loss
    # criterion = calculate_dis
    # criterion = nn.functional.l1_loss
    # criterion = nn.functional.cross_entropy

    output_str_length = 0
    sliding_window = (loss_type == 'ctc' and config['input']['test_sliding_window']['applied'])
    # if sliding_window:
    fake_gts = []
        # raw_preds_all = []

    for i, (example_input, example_target) in enumerate(val_loader):
        # if i < 1000: continue
        out = model(example_input[0])
        print(example_input[0], example_input[0].shape)
        print(example_target[0])
        print(out)
        # break
        traced_script_module = torch.jit.trace(model, example_input[0])
        optimized_traced_model = optimize_for_mobile(traced_script_module)
        optimized_traced_model._save_for_lite_interpreter("traced_model.ptl")
        out = optimized_traced_model(example_input[0])
        print(out)
        break

    print('Traced successfully')
    return


################################################################################
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)