## Dataset pre-processing and creation for NeckFace
#### File Structure

```
.
├── code_create_datasets
│   ├── neckface_device_recordings
│   │   ├── convert_to_mp4.py
│   │   ├── neckface_createDataset.py
│   │   ├── neckface_survey_recordings_createDataset.py
│   │   ├── numerical_preds_extraction_through_timestamps.py
│   │   ├── neckface_read_frames_from_npy.py
│   │   └── test_neckface_createDataset.py
│   ├── neckface_openface_feature_extraction
│   │   ├── extract_openface_feature_files.py
│   │   ├── openFace_features_downsampling.py
│   │   └── openFace_neckFace_mapping.py
│   └── study_metadata.ipynb
│   └── README.md
└── data
    ├── neckface_dataset
    │   ├── failureOccurrence_unix_timeStamp_mapping.xlsx
    │   ├── neckface_device_frame_dataset
    │   │   ├── 1
    |   |   |..
    |   |   |..
    │   │   └── 30
    │   ├── neckface_device_frame_dataset.zip
    │   ├── neckface_device_recordings
    │   │   ├── format_avi
    │   │   └── format_mp4
    │   ├── participant_log.xlsx
    │   ├── pred_outputs
    │   │   ├── 1
    |   |   |..
    |   |   |..
    │   │   ├── 9
    │   │   └── all_participant_preds.csv
    │   └── preds
    │       ├── 1-selected
    |       |..
    |       |..
    │       └── 9-selected
    ├── neckface_openface_dataset
    │   ├── 12_fps_downsampled_neckface_openface_dataset
    │   │   ├── 12_fps_all_participants_openface_neckface.csv
    │   │   ├── 12_fps_all_participants_openface_neckface.npy
    │   │   ├── 2102
    │   │   ├── 2746
    │   │   │..
    │   │   │..
    │   │   └── 9623
    │   ├── openface_dataset
    │   │   ├── 2102
    │   │   ├── 2746
    │   │   │..
    │   │   │..
    │   │   └── 9623
    │   └── openface_numerical_features_dataset
    │       ├── 2102
    │       ├── 2746
    │       │..
    │       │..
    │       ├── 9623
    │       └── all_participants_openface_features.csv
    ├── openface_neckface_mapped_dataset
    │   ├── 2102
    │   │   └── openface_to_neckface_matched_dataset.csv
    │   ├── ..
    │   ├── ..
    │   ├── 9623
    │   │   └── openface_to_neckface_matched_dataset.csv
    │   └── all_participants_openface_to_neckface_matched_dataset.csv
    └── neckface_extracted_study_data.zip
```

#### Neckface Device Recordings Frame Extraction

- Code Structure: `code_create_datasets/neckface_device_recordings`

1. `convert_to_mp4.py`: Converts the `.avi` video files to `.mp4` - for easier extraction of frames using `cv2`
1. `neckface_createDataset.py`: This script considers the timestamps from the neckface device and extracts all the frames from the neckface IR camera recording of the participant session recordings.
1. `neckface_survey_recordings_createDataset.py`: This script extracts the survey recording frames `[RGB]` from the NeckFace survey study.
    ```
    - The failureOccurrence_unix_timeStamp_mapping.xlsx dataset is read to obtain the unix timestamps of the [start, end] durations of each stimulus video viewed by the given participant.
    - For each stimulus video's [start, stop] unix timestamp, from the `data/preds/{participant_id}-selected/{participant_id}_survey_preds.npy` file, we obtain the indicies of the datapoints that fall within this timestamp range by matching the unix_timestamp present in the `.npy` file.  
    - Since, the rate at which the recordings are obtained are same, it makes it easier to map the frame with the timestamp.  
    - Once the indices are obtained for the given stimulus video, we iterate through the session_recording and obtain only the frames that fall within this range. We then extract the frame along with the class_label and participant metadata.  
    - This extracted data is stored @ `data/neckface_device_frame_dataset/{participant_id}/`
    ```
1. `neckface_read_frames_from_npy.py`: Helps to verify if the frames are extracted correctly along with the right class_labels for a given participant.

#### Neckface Numerical Prediction Extraction

- Code Structure: `code_create_datasets/neckface_device_recordings/`

1. `numerical_preds_extraction_through_timestamps.py`: This acts very similar to the Neckface Device Recordings Frame Extraction logic. Except, instead of extracting frames from the `{participant_id}_survey.avi` video file, it extracts the datapoints from the `preds` folder containing the `survey_preds.npy`.


#### Neckface OpenFace Numerical Features Extraction

- Code Structure: `code_create_datasets/neckface_openface_feature_extraction/`

1. `extract_openface_feature_files.py`: This script reads the OpenFace extraction output consists of the `hog, .avi, .csv, details.txt, and aligned` files. Extracts the `.csv` files which consist of the numerical prediction of the Facial Activation Units and moves it to the new directory for downsampling for features and frames.

1. `openFace_features_downsampling.py`: This script performs downsampling on the datapoints of the OpenFace extracted features to the required `downsampling_size` and merges all the downsampled_features of all participants into a single dataset (both in `csv` and `npy` file formats)

1. `openFace_neckFace_mapping.py`: This script matches the OpenFace numerical features dataset to the NeckFace feature dataset where the timestamps are the same. If there exists no exact matches, OpenFace frames that are next closest to the current NeckFace frame are considered (based on the timestamps).
Working:
This script reads in the `failureOccurrence_unix_timeStamp_mapping.xlsx` file and obtains the `unix_start` and `unix_end` timestamps for a given participant recording. With the help of this information, we update the OpenFace dataset to hold the true unixTimestamp for each frame `unix_start_time + frame_timestamp (converted to unix) = true_timestamp`.  
It also reduces the number of features to match the features that of the NeckFace features. Once this dataset is obtained, it then looks for OpenFace datapoints (frames) that are matching in timestamps with respect to the timestamps from the neckface device dataset.  If exact matches of timestamps are not found, we consider the next closest openface_timestamp relative to the current neckface_timestamp.

#### Datasets

1. `neckface_extracted_study_data.zip`: Contains the `preds` files from the neckface device.
1. `preds/`: The `participant_id/{survey}_preds.npy` files are manually selected and downloaded from the box. This contains the entire session recording's extracted numerical predicition from the neckface device.
1. `pred_outputs/`: This contains the extracted numerical datapoints that fall within the ranges of the `[start, end]` unix timestamps of when the participant viewed a given stimulus_video. These datapoints are obtained by executing the `code_create_datasets/neckface_device_recordings/numerical_preds_extraction_through_timestamps.py` script.
1. `neckface_device_recordings/`: This file consits of the `{participant_id}_survey.avi` video files that is manually selected and downloaded from the box.
1. `participant_log.xlsx`: Used to determine whether a given participant's data is to be considered in the dataset creation depending on whether the participant is included in the final study or not.
1. `failureOccurrence_unix_timeStamp_mapping.xlsx`: This `df` is read to obtain the unix timestamps of the `[start, end]` durations of each stimulus video viewed by the given participant.