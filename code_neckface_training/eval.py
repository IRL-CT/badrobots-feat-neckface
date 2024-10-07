from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from libs.utils import (AverageMeter, load_frame_time)
from libs.models import EncoderDecoder as ModelBuilder
from libs.core import load_config
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import torch
import random
import numpy as np
import torch.nn as nn

# python imports
import argparse
import os
import time
import cv2
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
                    help='print frequency (default: 30)')
parser.add_argument('-v', '--videos', default='', type=str,
                    help='path(s) to videos to be evaluated, comma separated')
parser.add_argument('-g', '--visible-gpu', default='', type=str,
                    help='visible gpus, comma separated, e.g. 0,1,2 (overwrites config file)')
# parser.add_argument('-m', '--mode', default='', type=str,
#                     help='the mode of training')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# main function for training and testing

def main(args):
    # ===================== initialization for training ================================
    print(time.strftime('begin: %Y-%m-%d %H:%M:%S', time.localtime()))
    torch.cuda.empty_cache()
    # parse args
    torch.set_num_threads(int(TARGET_NUM_THREADS))

    config = load_config()  # load the configuration
    input_config = config['input']
    #print('Current configurations:')
    
    torch.cuda.empty_cache()
    
    # use spawn for mp, this will fix a deadlock by OpenCV (do not need)
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')

    # fix the random seeds (the best we can)
    fixed_random_seed = 20220217
    torch.manual_seed(fixed_random_seed)
    np.random.seed(fixed_random_seed)
    random.seed(fixed_random_seed)

    config['network']['output_dims'] = 55
    config['network']['input_channels'] = 1

    model = ModelBuilder(config['network'])  # load the designed model
    # GPU you will use in training
    master_gpu = config['network']['devices'][0]
    model = model.cuda(master_gpu)  # load model from CPU to GPU
    model = nn.DataParallel(model, device_ids=config['network']['devices'])
    # ============================= retrain the trained model (if need usually not) =========================================
    # resume from a checkpoint?

    print('loading trained model.....')
    if os.path.isfile(args.resume):
        print('=> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(args.resume,
                                map_location=lambda storage, loc: storage.cuda(master_gpu))
        # args.start_epoch = 0
        best_metric = checkpoint['best_metric']
        model.load_state_dict(checkpoint['state_dict'])
        # only load the optimizer if necessary
        print('=> loaded checkpoint {} (epoch {}, metric {:.3f})'
                .format(args.resume, checkpoint['epoch'], best_metric))
    else:
        print('=> no checkpoint found at {}'.format(args.resume))
        return
    
    cudnn.enabled = True
    cudnn.benchmark = True
    
    # set up transforms and dataset
    # ===================== packet the training data and testinf data ================================
    
    for video in args.videos.split(','):
        print('Loading video', video)
        cap = cv2.VideoCapture(video)
        grabbed = True
        n_frames = 0
        loaded_frames = []
        frame_ts = load_frame_time(video[:-4] + '.txt')
        while grabbed:
            grabbed, img = cap.read()
            if not grabbed:
                continue
            n_frames += 1
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            loaded_frames += [img[None, ...].astype(np.float32)]
        cap.release()

        # align the frames and timestamps and drop last batch, just to be convenient
        valid_frames = min(len(frame_ts), n_frames)
        valid_frames -= valid_frames % input_config['batch_size']
        loaded_frames = loaded_frames[:valid_frames]
        frame_ts = frame_ts[:valid_frames]

        if len(loaded_frames) < input_config['batch_size']:
            continue

        loaded_frames = np.array(loaded_frames, dtype=loaded_frames[0].dtype)
        loaded_frames.shape = (-1, input_config['batch_size'], loaded_frames.shape[1], loaded_frames.shape[2], loaded_frames.shape[3])
        frame_ts = np.array(frame_ts)

        val_loader = torch.Tensor(loaded_frames)
        print('%d frames loaded, %d batches' % (valid_frames, valid_frames // input_config['batch_size']))
        preds = validate(val_loader, model, args, config)
        preds_with_ts = np.c_[frame_ts, preds]
        save_path = video[:-4] + '_preds.npy'
        np.save(save_path, preds_with_ts)
        print('Eval done, saved to', save_path)

    print(time.strftime('end: %Y-%m-%d %H:%M:%S', time.localtime()))

def validate(val_loader, model, args, config):
    '''Test the model on the validation set'''
    # set up meters
    batch_time = AverageMeter()
    
    # switch to evaluate mode
    model.eval()
    end = time.time()

    # prepare for outputs
    pred_list = []
    # truth_list = []

    output_str_length = 0

    # loop over validation set
    for i, (input_arr) in enumerate(val_loader):
        
        input_arr = input_arr.cuda(config['network']['devices'][0], non_blocking=True)
        # forward the model
        # print(input_arr.shape)
        with torch.no_grad():
            output = model(input_arr)
        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        pred_list += [output[0].cpu()]

        # printing
        if i % (args.print_freq * 2) == 0 or i == len(val_loader) - 1:
            output_str = 'Test: [{:4d}/{:4d}], Time: {:.2f} ({:.2f})'.format(i + 1, len(val_loader),
                                                  batch_time.val, batch_time.avg)
            print(output_str, end='\r')
            output_str_length = max(output_str_length, len(output_str))
            
    pred_list = np.concatenate(pred_list)
    return pred_list

################################################################################
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)