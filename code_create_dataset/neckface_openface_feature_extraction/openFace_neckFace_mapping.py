## This script reads in the OpenFace extracted facial features dataset (csvs) and performs a 1:1 mapping on the NeckFace device timestamps
## consider_datapoint = openface_timestamp == neckface_timestamp
## If the exact timestamp is not found (i.e: openface_timestamp == neckface_timestamp), we consider the next closest timestamp to the neckface_timestamp
## We set a error threshold to check how many datapoints cross the threshold when matched to the next closest timestamp.

import os
import math
import shutil
import numpy as np
import pandas as pd

from tqdm import tqdm
from pprint import pprint
from functools import partial


files_to_ignore = [".DS_Store", "all_response_features.csv", "all_participants_openface_features.csv"]

def get_features():
    final_features = ['participant_id', 'response_video', 'class', 'unixTimestamp']
    required_features = [
        'timestamp',
        'gaze_0_x', 'gaze_0_y', 'gaze_0_z',
        'gaze_1_x', 'gaze_1_y', 'gaze_1_z',
        'gaze_angle_x', 'gaze_angle_y',
        'pose_Tx', 'pose_Ty', 'pose_Tz', 
        'pose_Rx', 'pose_Ry', 'pose_Rz',
        'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 
        'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r',
        'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 
        'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r',
        'AU45_r', 'AU01_c', 'AU02_c', 'AU04_c', 
        'AU05_c', 'AU06_c', 'AU07_c', 'AU09_c', 
        'AU10_c', 'AU12_c', 'AU14_c', 'AU15_c',
        'AU17_c', 'AU20_c', 'AU23_c', 'AU25_c',
        'AU26_c','AU28_c','AU45_c'
    ]
    return final_features, required_features

def convert_to_unix_timestamp(row, unix_start_timestamps):
    """Calculate the UnixTimestamp of the frame given the timestamp of the frame and when the recording began

    Args:
        row : OpenFace data row - containing features and other metadata
        unix_start_timestamps (Dict): Contains the unix_start and unix_stop timestamps of the given recording for a given participant

    Returns:
        frame_unix_timestamp: converted frame timestamp
    """
    start_timestamp = unix_start_timestamps[row['participant_id']][row['response_video']]['unix_timestamp_start']
    return start_timestamp + (row['timestamp'] * 1000) ## converting the seconds to milliseconds

def merge_all_participant_openface_features(openface_features_dataset_path, participant_recording_timestamp_info_path, participant_log_path):
    """Reads in all the OpenFace extracted feature files for all the participants.
    Reduces the number of features to the required_features + some metadata.
    Merges the features dataset of all the recordings for a given participant into a single dataset
    Merges all the participants features dataset into a all_participants_dataset.

    Args:
        openface_features_dataset_path (str): OpenFace extracted features path
        participant_recording_timestamp_info_path (str): Contains the unix_start and unix_stop timestamps of all recordings for all participants
        participant_log_path (str): Participant Log File path (contains the participant_id during the study and qualtrics_id) 
    """
    output_path = openface_features_dataset_path + "all_participants_openface_features.csv"
    all_participants_df = pd.DataFrame()
    participants = [participant for participant in os.listdir(openface_features_dataset_path) if participant not in files_to_ignore]

    qualtrics_to_participant_id = get_participant_id_mappings(
        participant_log_path = participant_log_path
    )
    # participant_id_to_qualtrics_id = {value: key for key, value in participant_id_to_qualtrics_id.items()}

    start_stop_unix_timestamps = get_participant_start_stop_unix_timestamps(participant_recording_timestamp_info_path)
    target_labels = {
        "c": 0, ## Control
        "f": 1 ## Failure
    }

    

    final_features, required_features = get_features()
    for participant_qualtrics_id in tqdm(sorted(participants), total=len(participants), desc="Merging openface features: "):
        participant_id = qualtrics_to_participant_id[int(participant_qualtrics_id)]
        feature_file_path = openface_features_dataset_path + participant_qualtrics_id + "/"
        feature_files = [file for file in os.listdir(feature_file_path) if file not in files_to_ignore]
        participant_df = pd.DataFrame()
        for csv_file in sorted(feature_files):
            stimulus_video_name = csv_file.split("_")[0]
            class_label = stimulus_video_name[0]
            csv_file_path = feature_file_path + csv_file
            features_df = pd.read_csv(csv_file_path)
            
            ## Remove whitespace characters
            features_df.columns = features_df.columns.str.replace(" ", "")
            
            ## Create a copy to consider only the required feature columns
            required_df = features_df[required_features].copy()
            required_df["participant_id"] = participant_id
            required_df["response_video"] = stimulus_video_name
            required_df["class"] = target_labels[class_label]
            # Create a new column with Unix timestamps
            required_df["unixTimestamp"] = required_df.apply(
                lambda row: convert_to_unix_timestamp(row, unix_start_timestamps=start_stop_unix_timestamps),
                axis=1
            )
            required_df = required_df[final_features + required_features] ## Reorganize the column order
            required_df["unixTimestamp"] = required_df["unixTimestamp"].astype("int")
            participant_df = pd.concat([participant_df, required_df])
        participant_df.to_csv(feature_file_path + "all_response_features.csv", index=False)
        all_participants_df = pd.concat([all_participants_df, participant_df])
    all_participants_df.to_csv(output_path, index=False)

def get_participant_start_stop_unix_timestamps(participant_recording_timestamp_info_path):
    """Creates a dict containing information about the participant recordings.
    {
        participant_id: {
            stimulus_video_name: {
                qualtrics_id: str,
                class_label: int,
                unix_timestamp_start: int,
                unix_timestamp_end: int,
            }
        }, ...
    }

    Args:
        participant_recording_timestamp_info_path (str): Contains the unix_start and unix_stop timestamps of all recordings for all participants

    Returns:
        start_stop_timestamps (Dict): participant_recording_timestamp_info
    """
    start_stop_df = pd.read_excel(participant_recording_timestamp_info_path)
    participant_ids = start_stop_df["participant_id"].unique()
    start_stop_timestamps = {}
    for participant_id in participant_ids:
        participant_info = start_stop_df[start_stop_df["participant_id"] == participant_id].drop_duplicates(
            subset=["stimulusVideo_name"],
            keep="first" ## Drop all rows that have the same stimulusVideo_name (when the participant has viewed the stimlus video multiple times) and keep only the first occurrence
        )
        timestamps = {}
        for index, row in participant_info.iterrows():
            qualtrics_id = row["qualtrics_participant_id"]
            class_label = row["stimulusVideo_class"]
            stimulus_video_name = row["stimulusVideo_name"]
            unix_timestamp_start_column = "responseVideo_unixTime_start" if class_label == "control" else "failureOccurrence_unixTimestamp"
            unix_timestamp_start = int(row[unix_timestamp_start_column])
            unix_timestamp_end = int(row["responseVideo_unixTime_end"])
            timestamps[stimulus_video_name] = {
                "qualtrics_id": qualtrics_id,
                "class_label" : class_label,
                "unix_timestamp_start": unix_timestamp_start,
                "unix_timestamp_end": unix_timestamp_end
            }
        start_stop_timestamps[participant_id] = timestamps
    return start_stop_timestamps

def get_participant_id_mappings(participant_log_path):
    """Since the OpenFace extracted features dataset is named according to the participant's qualtircs_id, we map these ids to the participant's study_id

    Args:
        participant_log_path (str): Participant Study Log File path

    Returns:
        qualtrics_id_to_participant_id (Dict): {qualtrics_id: participant_id}
    """
    participant_info_df = pd.read_excel(participant_log_path)
    participant_info_df.fillna(0, inplace=True)
    participant_info_df[["participant_number", "qualtrics_id"]] = participant_info_df[["participant_number", "qualtrics_id"]].astype("int")
    
    # Create a mapping from qualtrics_id to participant_number
    qualtrics_id_to_participant_id = dict(zip(participant_info_df["qualtrics_id"], participant_info_df["participant_number"]))
    return qualtrics_id_to_participant_id

def get_neckface_preds(neckface_data_path, participant_id):
    """Given the participant_id, it reduces the neckface_preds dataframe containing data from all participants to return the datapoints belonging only to the participant_id

    Args:
        neckface_data_path (str): path to all_participant_preds.csv
        participant_id (int): participant_id during the study

    Returns:
        participant_preds_df (pd.DataFrame()): neckface extracted features for the given participant_id
    """
    neckface_preds_df = pd.read_csv(neckface_data_path)
    participant_preds_df = neckface_preds_df[neckface_preds_df["participant_id"] == participant_id]
    return participant_preds_df

def get_openface_feature_preds(openface_data_path, qualtrics_id):
    """Given the participant_id, it returns the openface dataframe containing data from all recordings for the given participant_id

    Args:
        openface_data_path (str): openface dataset path
        qualtrics_id (int): qualtrics_id

    Returns:
        participant_features_df (pd.DataFrame()): OpenFace dataset for the given participant_id
    """
    feature_file_path = openface_data_path + f"{qualtrics_id}/all_response_features.csv"
    participant_features_df = pd.read_csv(feature_file_path)    
    return participant_features_df

def perform_openface_to_neckface_data_mapping(
    participant_log_path,
    neckface_features_dataset_path,
    openface_features_dataset_path,
    output_path,
    error_threshold=1/12,
):
    """Finds matching rows in openface to the neckface dataset based on the unixtimestamps, where:
    openface[unix_timestamp] == neckface[unix_timestamp], if this the exact timestamp is not found,
    we consider the next closest openface[unix_timestamp] datapoint where openface_ts > neckface_ts.

    Args:
        participant_log_path (str): path to participant id information
        neckface_features_dataset_path (str): path to the neckface extracted dataset - where the timestamps are split and sorted based on the stimulus video viewed
        openface_features_dataset_path (str): path to the openface extracted dataset
        output_path (str): output path to the openface_to_neckface_matched dataset
        error_threshold (float, optional): To observe how many openface_ts are greater than neckface_ts by the error_threshold seconds (later converted to unix milliseconds). Defaults to 1/12.
    """
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    qualtrics_id_to_participant_id = get_participant_id_mappings(participant_log_path)
    ## Participants Excluded 
    participants_to_exclude = {13, 17, 27, 29, 30}
    participants_to_consider = [int(participant) for participant in os.listdir(openface_features_dataset_path) if (participant not in files_to_ignore) and (qualtrics_id_to_participant_id[int(participant)] not in participants_to_exclude)]

    all_participants_openface_neckface_mappings = []
    for participant_qualtrics_id in tqdm(sorted(participants_to_consider), total=len(participants_to_consider), desc="Matching participant dataset: "):
        openface_features_df = get_openface_feature_preds(
            openface_data_path=openface_features_dataset_path,
            qualtrics_id=participant_qualtrics_id,
        )
        neckface_preds_df = get_neckface_preds(
            neckface_data_path=neckface_features_dataset_path,
            participant_id=qualtrics_id_to_participant_id[participant_qualtrics_id]
        )
        # print(openface_features_df.head(5))
        # print(neckface_preds_df.head(5))
        # break
        # if qualtrics_id_to_participant_id[participant_qualtrics_id] not in [5]:
        #     continue
        stimulus_videos = openface_features_df["response_video"].unique()
        closest_greatest_rows = []
        absolute_differences = []
        for video in stimulus_videos:
            ## Filter by video to ensure that the timestamps are monotonically increasing
            nf_video_df = neckface_preds_df[neckface_preds_df["stimulus_video_name"] == video]
            of_video_df = openface_features_df[openface_features_df["response_video"] == video]
            
            # ## The unix timestamp columns are in milliseconds, here we convert them to seconds
            # nf_video_df["corrected_timestamps"] = nf_video_df["corrected_timestamps"].apply(lambda timestamp: timestamp / 1000)
            # of_video_df["unixTimestamp"] = of_video_df["unixTimestamp"].apply(lambda timestamp: timestamp / 1000)
            for neckface_ts in nf_video_df["corrected_timestamps"]:
                # Find the openface timestamps that is >= neckface timestamp
                greater_or_equal_ts = of_video_df[of_video_df["unixTimestamp"] >= neckface_ts]
                if not greater_or_equal_ts.empty:
                    openface_row = greater_or_equal_ts.iloc[0]  # Get the first matching row
                    abs_difference = abs(openface_row["unixTimestamp"] - neckface_ts)
                    absolute_differences.append(abs_difference)
                    closest_greatest_rows.append(openface_row)
        openface_neckface_mapped_df = pd.DataFrame(closest_greatest_rows).reset_index(drop=True)

        unix_mapping_error_threshold = error_threshold * 1000 ## Threshold in terms of unix milliseconds (1s = 1000 ms)
        number_of_errors = sum(1 for difference in absolute_differences if difference > unix_mapping_error_threshold)
        number_of_matches = len(absolute_differences)
        error_rate = (number_of_errors / number_of_matches) * 100 if number_of_matches > 0 else 0
        
        # Store the participant dataset
        participant_of_to_nf_matches_output_path = output_path + f"{participant_qualtrics_id}/"
        if not os.path.exists(participant_of_to_nf_matches_output_path):
            os.mkdir(participant_of_to_nf_matches_output_path)
        openface_neckface_mapped_df.to_csv(participant_of_to_nf_matches_output_path + "openface_to_neckface_matched_dataset.csv", index=False)

        all_participants_openface_neckface_mappings.append(openface_neckface_mapped_df)
        # print(f"Matching for Participant ID: {qualtrics_id_to_participant_id[participant_qualtrics_id]}")
        # print(f"Shape of OpenFace Dataset: {openface_features_df.shape}")
        # print(f"Shape of NeckFace Dataset: {neckface_preds_df.shape}")
        # print(f"Shape of OpenFace to NeckFace Matched Dataset: {openface_neckface_mapped_df.shape}")
        # print(f"Participant: {qualtrics_id_to_participant_id[participant_qualtrics_id]} Error Rate in Matched Rows: {error_rate:.3f} ({number_of_errors}/ {number_of_matches})")
        # break # from participant
    all_participants_openface_neckface_mappings_df = pd.concat(all_participants_openface_neckface_mappings, ignore_index=True)
    all_participants_openface_neckface_mappings_df.to_csv(output_path + "all_participants_openface_to_neckface_matched_dataset.csv", index=False)

def main():
    version = "v2"
    data_path = "../../data/"
    participant_log_path = data_path + "neckface_dataset/participant_log.xlsx"
    participant_recording_timestamp_info_path = data_path + "neckface_dataset/failureOccurrence_unix_timeStamp_mapping.xlsx"
    neckface_extracted_timestamps = data_path + f"neckface_dataset/pred_outputs_{version}/all_participant_preds.csv"
    openface_features_dataset_path = data_path + "neckface_openface_dataset/openface_numerical_features_dataset/"

    ## Iterate through all the openface datasets and merge them based on the participants and also by all participants
    ## This also updates the timestamp from the openface frame to the unix_timestamp of when the participant started the survey for a particular stimulus_video
    ## i.e: frame_seconds (converted to unix) + unix_start_time (for a particular stimulus video)
    ## NOTE: For when the stimulus_video is of class "failure", the unix_start_time is based on from the moment of failure occurrence in the stimulus_video
    # merge_all_participant_openface_features(
    #     openface_features_dataset_path=openface_features_dataset_path,
    #     participant_recording_timestamp_info_path=participant_recording_timestamp_info_path,
    #     participant_log_path=participant_log_path
    # )

    ## Map the OpenFace data based on the timestamps in NeckFace data
    ## Calculate the error in matches based on the error_threshold (here, pass in seconds - later in the method, it is converted into unix milliseconds)
    openface_to_neckface_mapped_output_path = data_path + "openface_neckface_mapped_dataset/"
    error_threshold = 1 / 12
    perform_openface_to_neckface_data_mapping(
        participant_log_path = participant_log_path,
        neckface_features_dataset_path = neckface_extracted_timestamps,
        openface_features_dataset_path = openface_features_dataset_path,
        output_path = openface_to_neckface_mapped_output_path,
        error_threshold=error_threshold,
    )

if __name__ == "__main__":
    main()