'''
Some utils for data processing
2/17/2022, Ruidong Zhang, rz379@cornell.edu
'''

def load_default_audio_config():
    default_audio_config = {
        'sampling_rate': 50000,
        'n_channels': 2,
        'channels_of_interest': [],
        'signal': 'FMCW',
        "tx_file": "fmcw19k.wav",
        'frame_length': 600,
        'sample_depth': 16,
        'bandpass_range': [19000, 23000]
    }
    return default_audio_config

def display_progress(cur_progress, total, update_interval=1000):
    if cur_progress % update_interval == 0 or cur_progress == total - 1:
        print('Progress %d/%d %.2f%%' % (cur_progress + 1, total, 100 * (cur_progress + 1) / total), end='\r')
    if cur_progress == total - 1:
        print('Done.   ')

def load_frame_time(frame_time_file):
    frame_times = []
    with open(frame_time_file, 'rt') as f:
        for l in f.readlines():
            if l[0] > '9' and l[0] < '0':
                continue
            frame_times += [float(l)]

    return frame_times

def extract_labels(loaded_gt):
    labels = {}
    n_cls = max([int(x[0]) for x in loaded_gt]) + 1
    for x in loaded_gt:
        labels[int(x[0])] = x[3]
        if len(labels) == n_cls:
            break
    labels_ordered = [labels[x] if x in labels else '' for x in range(n_cls)]
    return labels_ordered