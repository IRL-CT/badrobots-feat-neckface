'''
Metrics for the evaluation
3/5/2022, Ruidong Zhang, rz379@cornell.edu
'''

import math
import torch
import numpy as np
from itertools import groupby
from torch.nn import functional as F
from torchmetrics.text.wer import WordErrorRate as WER

def acc(outputs, targets, require_pred=False):
    preds = outputs.cpu().detach().numpy().argmax(axis=1)
    truths = targets.cpu().detach().numpy()
    metric = np.mean(preds == truths)

    return metric, preds, None

def wer(outputs, targets, require_pred=False, print_result=False, blank_label=-1):
    if blank_label < 0: # default is the largest one (bc zero has been taken)
        blank_label = outputs.shape[2] - 1
    # outputs: N x L x D
    _, batch_raw_pred = torch.max(outputs, dim=2)
    # print(outputs.shape, batch_raw_pred.shape)
    all_preds = []
    raw_preds = []
    metrics = []
    criterion = WER()
    for i in range(outputs.shape[0]):
        raw_pred = batch_raw_pred[i].cpu().detach().numpy()
        raw_preds += [raw_pred]
        this_pred = ' '.join([str(c) for c, _ in groupby(raw_pred) if c != blank_label])
        all_preds += [this_pred]
        if targets is not None:
            metrics += [criterion(this_pred, targets[i])]
    #     if print_result:
    #         print(raw_pred, end=' ')
    # print(targets, all_preds)
    if len(metrics):
        metric = np.mean(metrics)
    else:
        metric = None
    return metric, all_preds, raw_preds

def wer_sliding_window(raw_preds, fake_gts, loaded_truths, blank_label=-1):
    max_label = max([max([int(xx) for xx in x[0].split()]) for x in loaded_truths]) + 1
    if blank_label == -1:
        blank_label = max_label
    max_idx = max([x[1] for x in fake_gts] + [x[5] for x in loaded_truths])
    pixels_per_pred = (fake_gts[0][1] - fake_gts[0][0]) // len(raw_preds[0])
    raw_pred_freq = np.zeros((max_idx // pixels_per_pred + 1, max_label + 1), dtype=int)
    for raw_pred, fake_gt in zip(raw_preds, fake_gts):
        start_idx = fake_gt[0] // pixels_per_pred + 1
        # print(raw_pred, fake_gt, start_idx)
        for i in range(1, len(raw_pred) - 1):
            raw_pred_freq[start_idx + i, raw_pred[i]] += 1
    raw_pred_freq[np.sum(raw_pred_freq, axis=1) == 0, blank_label] = 1
    raw_preds_all = np.argmax(raw_pred_freq, axis=1)
    assembled_preds = []
    metrics = []
    criterion = WER()
    for truth in loaded_truths:
        truth_text = truth[0]
        truth_raw_preds = raw_preds_all[math.ceil(truth[4] / pixels_per_pred): (truth[5] // pixels_per_pred)]
        truth_assembled_pred = ' '.join([str(c) for c, _ in groupby(truth_raw_preds) if c != blank_label])
        assembled_preds += [truth_assembled_pred]
        metrics += [criterion(truth_assembled_pred, truth_text)]
        # print(truth, (truth[4] // pixels_per_pred, truth[5] // pixels_per_pred), truth_raw_preds, truth_assembled_pred, metrics[-1])
    # print(metrics)
    return np.mean(metrics), assembled_preds, raw_preds_all


def mae(outputs, targets, require_pred=False):
    criterion = F.l1_loss
    metric = criterion(outputs.data, targets.data)

    preds = None
    if require_pred:
        preds = outputs.cpu().detach().numpy()

    return metric, preds, None


def get_criterion(loss_type):
    if loss_type == 'ce':
        return acc
    elif loss_type == 'ctc':
        return wer
    else:
        return mae