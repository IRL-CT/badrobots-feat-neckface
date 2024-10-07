# Data pre-processing and training for NeckFace

> Note: re-organized code, not the original version as in the paper

> 12/18/2022, Ruidong Zhang, rz379@cornell.edu

## Preparing data

After data collection, you should have the following files from each session:

1. A video file, collected from the Raspberry Pi.
2. The timestamp file accompanying the video file.
3. The ground truth file, originally with no extension, collected from the laptop which receives data from the iOS device.

You can create a folder for each session and put the three files in it. An example of the file structure is:

    TestParticipant1
    ├── part1
    │   ├── ground_truth_raw_part1
    │   ├── part1.avi
    │   ├── part1.txt
    │   └── Screen Recording Part 1.MP4
    ├── part2
    │   ├── ground_truth_raw_part2
    │   ├── part2.avi
    │   ├── part2.txt
    │   └── Screen Recording Part 2.MP4
    └── part3
        ├── ground_truth_raw_part3
        ├── part3.avi
        ├── part3.txt
        └── Screen Recording Part 3.MP4

> Note: only for better organization purpose, you can also put those files from different sessions all together.

## Preparing `config.json`

The next step is to specify the sessions and files in the `config.json` file. Create this file at `TestParticipant1`. In later steps, this file tells the programs how sessions are configured and where to find the files. You just need to modify the following example for the folder above:

    {
        "video": {
            "files": [
                "part1/part1.avi",
                "part2/part2.avi",
                "part3/part3.avi"
            ],
            "syncing_poses": [
                0,
                0,
                0
            ]
        },
        "ground_truth": {
            "files": [
                "part1/ground_truth_raw_part1",
                "part2/ground_truth_raw_part2",
                "part3/ground_truth_raw_part3"
            ],
            "syncing_poses": [
                0,
                0,
                0
            ]
        }
    }

### Explanation of the `config.json`

Two fields specify the video files and ground truth files, respectively. In each field, you need to specify the location of the files in `files` and syncing positions in `syncing_poses`. In all fields, each line represents one session. They must be exactly matching each other. For instance, the first line in `syncing_poses` in `video` specifies the syncing position of the first line in `files`. Then, the first line in `ground_truth`'s fields specifies the file and syncing position of the corresponding ground truth file to the first lines in `video`'s fields. The order must not be messed up.

You do not need to specify the timestamp files of the videos. The program will find them according to the filename of the video files.

### How to find `syncing_poses`

`syncing_poses` is used to mark the exact time of the clapping in the video and ground truth. This is to make sure that the two files are perfectly synchronized. Synchronization is VERY important. If the ground truth and video is not synced well, the model will get wrong labels during training and will not work.

The best way to sync them is to look at both videos. For Raspberry Pi's video, find the exact clapping moment in the recorded video (e.g., in `part1/part1.avi`), find the frame number of the clap, put it in the `syncing_poses` field of `video`. For ground truth, find the exact clapping moment in screen recordings (e.g., in `part1/Screen Recording Part 1.MP4`), copy the timestamp displayed on the top of the screen to the `syncing_poses` field of `ground_truth`.

An simple yet sometimes not so accurate way of syncing is simply putting `0`s in both fields, as shown in the sample `config.json`. This is because both recorded video and ground truth use absolute timestamp to record the time. When both devices are connected to the Internet, there usually is not much difference between them. For best accuracy, please calibrate the clocks of both devices before starting data collection.

## Generating `dataset` for training

The next step is to parse the ground truth file to a format accepted by later training steps and generate uniform file structures for training. This can be done by simple running

    data_preparation.py -p /path/to/TestParticipant1

This will generate a `.npy` for each of the ground truth file and a `dataset` folder. The file structure looks like this after this step:

    TestParticipant1
    ├── config.json
    ├── dataset
    │   ├── session_01
    │   │   ├── config.json
    │   │   ├── ground_truth.npy -> ../../part1/ground_truth_raw_part1.npy
    │   │   ├── video.avi -> ../../part1/part1.avi
    │   │   └── video_ts.txt -> ../../part1/part1.txt
    │   ├── session_02
    │   │   ├── config.json
    │   │   ├── ground_truth.npy -> ../../part2/ground_truth_raw_part2.npy
    │   │   ├── video.avi -> ../../part2/part2.avi
    │   │   └── video_ts.txt -> ../../part2/part2.txt
    │   └── session_03
    │       ├── config.json
    │       ├── ground_truth.npy -> ../../part3/ground_truth_raw_part3.npy
    │       ├── video.avi -> ../../part3/part3.avi
    │       └── video_ts.txt -> ../../part3/part3.txt
    ├── part1
    │   ├── ground_truth_raw_part1
    │   ├── ground_truth_raw_part1.npy
    │   ├── part1.avi
    │   ├── part1.txt
    │   └── Screen Recording Part 1.MP4
    ├── part2
    │   ├── ground_truth_raw_part2
    │   ├── ground_truth_raw_part2.npy
    │   ├── part2.avi
    │   ├── part2.txt
    │   └── Screen Recording Part 2.MP4
    └── part3
        ├── ground_truth_raw_part3
        ├── ground_truth_raw_part3.npy
        ├── part3.avi
        ├── part3.txt
        └── Screen Recording Part 3.MP4

The data preparation is done here.

## Training

The training script is `train.py`. An example command to train the model is

    python train.py -p /path/to/TestParticipant1 -ts 01 -g 0 -t 1 -o train12

Training options:

- `-p` points to the path of the dataset folder
- `-ts` or `--test-sessions` sessions that will be used for testing. This corresponds to the `xx` in `sessions_xx` in the generated `dataset` folder, and eventually goes back to the line you wrote in the `config.json`. In short, `01` will use the files written at the first line in the fields of `config.json` to be the testing session. For multiple sessions, use a style like `-ts 01,02`.
- `-g` GPU to use, starting from `0`.
- `-t` `1` for training, `0` for testing only.
- `-o` name of the checkpoint folder. With `-o train12`, you can expect a folder `TestParticipant1_train12` under `ckpt`. This folder will be the place where testing results and model checkpoints are saved.

Other useful options:

- `--train-sessions` sessions that will be used for training. If left empty, all sessions excluding those specifid by `-ts` and `--exclude-sessions` will be used for training. Same format as `-ts`.
- `--exclude-sessions` sessions specified here will not be used for training NOR testing. Same format as `-ts`.
- `--resume` points to the model checkpoint (`.pth.tar`) to resume. If your training is interrupted somehow, you can use this to resume training.

Other things to adjust:

- Data augmentation parameters are specified in `libs/core/config.py` (lines 27-35).
- Batch size is specified in `libs/core/config.py` (line 50). Adjust it lower if you experience `CUDA out of memory` error.
- Epochs is specified in `libs/core/config.py` (line 97). Normally 20-50 epochs is sufficient. The judging standard is that if increasing this number does not lead to better performance, then it is enough.

If configured properly, the training should start and generate a folder under `ckpt` to save its outputs.

## Fine-tuning the NeckFace model

The NeckFace meta model `neckface_us_full_model.tar` is trained on all data from the user study of NeckFace. When collecting new data from new users, fine-tuning on this model will yield much better performance.

To fine-tune, just specifying the `--resume` parameter to the path of the `neckface_us_full_model.tar`. In order to make sure that the model runs as expected, you also need to set the `--epochs-to-run` and `--lr`. The full command is

    python train.py -p /path/to/TestParticipant1 -ts 01 -g 0 -t 1 -o train12 --resume /path/to/neckface_us_full_model.tar --epochs-to-run 30 --lr 0.0002

A pre-trained model from the NeckFace user study can be found [here](https://cornell.box.com/s/cuqqeuz9eqyauwbkb1rkakqi6vul21op).
## Tracing model

After training is done, the next step is to transfer the trained model onto your laptop to complete the real-time demo. To do this, you need to convert the format of the model (trace the model) first.

The command is

    python trace_model.py -p /path/to/TestParticipant1 -ts 01 --resume ckpt/TestParticipant1_ts01/model_best.pth.tar

If executed properly, this will generate a `traced_model.ptl` file. This is the file you need.

## Running prediction

After obtaining the model, you can use the `eval.py` to generate predictions for a certain video. You need to specify the path to the model checkpoint and video(s) to be processed.

    python eval.py --resume /path/to/model/checkpoint.tar -v /path/to/video1.avi,/path/to/video2.avi

Use commas (`,`) for multiple video files. Do not add space before and after commas. This command will generate a prediction file for each video specified, named as `video1_preds.npy` and saved in the same path as `video1.avi`. The format of the prediction file is specified below. If the timestamp file and the video file have different number of frames, the one with fewer frames will be used.

### Prediction results

The prediction is formatted as an `npy` file. The shape of the array is `T x 56`, where `T` is the number of frames in the video/timestamp. For each line, `56` elements are present, consisting of `1` Unix timestamp of the frame + `52` facial blendshapes + `3` head rotation angles, as listed in the table below.

| Index | Description |
| --- | ----------- |
| 0 | Unix timestamp |
| 1 | eyeBlink_L |
| 2 | eyeLookDown_L |
| 3 | eyeLookIn_L |
| 4 | eyeLookOut_L |
| 5 | eyeLookUp_L |
| 6 | eyeSquint_L |
| 7 | eyeWide_L |
| 8 | eyeBlink_R |
| 9 | eyeLookDown_R |
| 10 | eyeLookIn_R |
| 11 | eyeLookOut_R |
| 12 | eyeLookUp_R |
| 13 | eyeSquint_R |
| 14 | eyeWide_R |
| 15 | jawForward |
| 16 | jawLeft |
| 17 | jawRight |
| 18 | jawOpen |
| 19 | mouthClose |
| 20 | mouthFunnel |
| 21 | mouthPucker |
| 22 | mouthLeft |
| 23 | mouthRight |
| 24 | mouthSmile_L |
| 25 | mouthSmile_R |
| 26 | mouthFrown_L |
| 27 | mouthFrown_R |
| 28 | mouthDimple_L |
| 29 | mouthDimple_R |
| 30 | mouthStretch_L |
| 31 | mouthStretch_R |
| 32 | mouthRollLower |
| 33 | mouthRollUpper |
| 34 | mouthShrugLower |
| 35 | mouthShrugUpper |
| 36 | mouthPress_L |
| 37 | mouthPress_R |
| 38 | mouthLowerDown_L |
| 39 | mouthLowerDown_R |
| 40 | mouthUpperUp_L |
| 41 | mouthUpperUp_R |
| 42 | browDown_L |
| 43 | browDown_R |
| 44 | browInnerUp |
| 45 | browOuterUp_L |
| 46 | browOuterUp_R |
| 47 | cheekPuff |
| 48 | cheekSquint_L |
| 49 | cheekSquint_R |
| 50 | noseSneer_L |
| 51 | noseSneer_R |
| 52 | tongueOut |
| 53 | faceRotationX |
| 54 | faceRotationY |
| 55 | faceRotationZ |

### Visualize offline results

You can use the same visualization system used for online predictions, with the help of a script to stream the prediction over TCP. To achieve this, first run the blendshape visualization program that listens at port `6001`. Then, use the stream script

    python serve_offline_vis.py -p /path/to/preds.npy

The default location that the script sends to is `127.0.0.1:6001`. To use other locations, use the `--port` and `--addr` options.


## NeckFace original image format

![NeckFace original image format](NeckFace_format.png)

The image is then converted to the size configured in `libs/core/config.py` in the field `input_size`.

The initial image processing when loading data into the memory is configured in `libs/dataset/data_generator.py`, data augmentation is done in `libs/dataset/dataset.py`.

## Dataset folder structure

A dumped `tree` command output of the dataset structure can be found [here](dataset_tree.txt). The `config.json` I used for the study can be found [here](config.json).