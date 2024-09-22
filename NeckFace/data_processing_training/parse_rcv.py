'''
Convert raw received data into csv file
11/30/2021, Ruidong Zhang, rz379@cornell.edu
'''

import re
import argparse
import numpy as np
import pandas as pd


def decode_line(line, line_headers):
    large_parts = line.replace(' ', '').split(',')
    if len(large_parts) != 5:
        return None
    ts = large_parts[1]
    blendshapes = large_parts[2]
    orientations = large_parts[3].split(':')
    data_piece = np.zeros((1 + 52 + 3,))
    data_piece[0] = float(ts)
    line_covered = np.zeros((52,))
    for bs in blendshapes.split('~'):
        this_column, this_val = bs.split(':')
        line_covered[line_headers.index(this_column) - 1] = 1
        data_piece[line_headers.index(this_column)] = float(this_val)
    if np.sum(line_covered) < 52:
        print(line)
    for i in range(3):
        data_piece[-(i + 1)] = float(orientations[i])
    # print(data_piece)
    # input()
    return data_piece

def parse_rcv(filepath, npy=False, no_rot=False):
    line_headers = [
        'time stamp', 
        'eyeBlink_L', 
        'eyeLookDown_L', 
        'eyeLookIn_L', 
        'eyeLookOut_L', 
        'eyeLookUp_L', 
        'eyeSquint_L', 
        'eyeWide_L', 
        'eyeBlink_R', 
        'eyeLookDown_R', 
        'eyeLookIn_R', 
        'eyeLookOut_R', 
        'eyeLookUp_R', 
        'eyeSquint_R', 
        'eyeWide_R', 
        'jawForward', 
        'jawLeft', 
        'jawRight', 
        'jawOpen', 
        'mouthClose', 
        'mouthFunnel', 
        'mouthPucker', 
        'mouthLeft', 
        'mouthRight', 
        'mouthSmile_L', 
        'mouthSmile_R', 
        'mouthFrown_L', 
        'mouthFrown_R', 
        'mouthDimple_L', 
        'mouthDimple_R', 
        'mouthStretch_L', 
        'mouthStretch_R', 
        'mouthRollLower', 
        'mouthRollUpper', 
        'mouthShrugLower', 
        'mouthShrugUpper', 
        'mouthPress_L', 
        'mouthPress_R', 
        'mouthLowerDown_L', 
        'mouthLowerDown_R', 
        'mouthUpperUp_L', 
        'mouthUpperUp_R', 
        'browDown_L', 
        'browDown_R', 
        'browInnerUp', 
        'browOuterUp_L', 
        'browOuterUp_R', 
        'cheekPuff', 
        'cheekSquint_L', 
        'cheekSquint_R', 
        'noseSneer_L', 
        'noseSneer_R', 
        'tongueOut',
        'faceRotationX',
        'faceRotationY',
        'faceRotationZ',
    ]
    with open(filepath, 'rb') as f:
        raw_file = f.read().decode()
        lines = []
        # ts_poses = []
        tss = re.finditer(r's, \d+\.\d+,', raw_file)
        last_line_start = 0
        for ts in tss:
            # ts_poses += [ts.span()]
            if ts.span()[0] > 0:
                lines += [raw_file[last_line_start: ts.span()[0]]]
            last_line_start = ts.span()[0]
        lines += [raw_file[last_line_start:]]

        csv_data = []
        for line in lines:
            line_data = decode_line(line, line_headers)
            if line_data is not None:
                csv_data += [line_data]

        print('%d lines decoded' % len(csv_data))
        csv_data = np.array(csv_data)
        if no_rot:
            csv_data = csv_data[:, :-3]
            # line_headers = line_headers[:-3]
        if npy:
            np.save(filepath + '.npy', csv_data)
        else:
            pd.DataFrame(csv_data).to_csv(filepath + '.csv', header=line_headers, index=None)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', help='path to the captured file')
    parser.add_argument('--npy', help='save npy file instead of csv', action='store_true')
    parser.add_argument('--no_rot', help='do not output head rotation', action='store_true')
    args = parser.parse_args()
    parse_rcv(args.file, args.npy, args.no_rot)