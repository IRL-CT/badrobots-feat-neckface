### This notebook performs downsampling on the OpenFace Extracted features' dataframes

### 1. Downsampling - to a specified ```sampling_size```
### 2. Preprocesses the downsampled data 
### - Excludes datapoints in ```failure videos``` which are before the ```failureOccurrence_timestamp```
### 3. Merges all ```participants``` all ```videos``` that are downsampled to a particular ```sampling_size``` and preprocessed (as per #2)

import os
import math
import copy
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

files_to_ignore = [".DS_Store"]

def get_features():
    final_features = ['participant_id', 'response_video', 'class']
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


def merge_dataframes(participant_info_path, downsampling_size, data_path):
    downsampled_participants = [participant for participant in os.listdir(data_path) if participant not in files_to_ignore and not participant.endswith(".csv")]
    print(downsampled_participants)
    all_participants_df = pd.DataFrame()
    
    participant_info_df = pd.read_excel(participant_info_path)
    participant_info_df.fillna(0, inplace=True)
    participant_info_df[["participant_number", "qualtrics_id"]] = participant_info_df[["participant_number", "qualtrics_id"]].astype("int")
    
    # Create a mapping from qualtrics_id to participant_number
    qualtrics_to_participant = dict(zip(participant_info_df["qualtrics_id"], participant_info_df["participant_number"]))
    
    for participant in tqdm(sorted(downsampled_participants), total=len(downsampled_participants), desc="Merging Participants: "):
        participant_file_path = data_path + participant + "/"
        participant_files = [file for file in os.listdir(participant_file_path) if file not in files_to_ignore]
        for feature_file in sorted(participant_files):
            feature_file_path = participant_file_path + feature_file
            feature_df = pd.read_csv(feature_file_path)
            all_participants_df = pd.concat([all_participants_df, feature_df])
        #     break # from feature_file
        # break # from participants

    ## Rename the participant_id column - as initially they refer to the qualtrics_id
    all_participants_df.rename(columns={
        "participant_id": "qualtrics_id"
    }, inplace=True)
    original_columns = list(all_participants_df.columns)
    
    ## Map the qualtrics_id to the participant_id and add this as a new column
    all_participants_df["participant_id"] = all_participants_df["qualtrics_id"].map(qualtrics_to_participant)
    all_participants_df = all_participants_df[["participant_id"] + original_columns] ## Rearrange the column headers
    
    ## Save the files in csv and npy file formats
    all_participants_output_path = data_path + f"{downsampling_size}_fps_all_participants_openface_neckface.csv"
    all_participants_numpy_output_path = data_path + f"{downsampling_size}_fps_all_participants_openface_neckface.npy"
    all_participants_df.to_csv(all_participants_output_path, index=False)
    all_participants_array = all_participants_df.to_numpy()
    np.save(all_participants_numpy_output_path, all_participants_array)

def downsample_features(final_features, required_features, downsampling_size, dataset_path, output_path):
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    rate = 1 / downsampling_size
    frequency = round(1 / rate)
    # print(f"rate, frequency = {rate, frequency}")

    participants = [participant for participant in os.listdir(dataset_path) if participant not in files_to_ignore]
    total_frame_count = {
        "c": 0, # control
        "f": 0 # failure
    }
    class_labels = {
        "c": 0, # control
        "f": 1 # failure
    }
    for participant in tqdm(sorted(participants), desc="Processing Participants: ", total=len(participants)):
        participant_frame_count = 0
        data_points_below_confidence_score = 0

        participant_file_path = dataset_path + participant + "/"
        participant_files = [file for file in os.listdir(participant_file_path) if file not in files_to_ignore]
        participant_output_path = output_path + participant + "/"
        if not os.path.exists(participant_output_path):
            os.mkdir(participant_output_path)

        for csv_file in sorted(participant_files):
            ## Identify the class to which the csv_file belongs to
            target_class = csv_file[0]
            response_video_name = csv_file.split("_")[:2][0]
            downsampled_csv_file_output_path = participant_output_path + response_video_name + f"_{frequency}fps.csv"
            # print(downsampled_csv_file_output_path)

            ## Read the csv file
            csv_file_path = participant_file_path + csv_file
            openface_df = pd.read_csv(csv_file_path)
            number_of_frames = openface_df.shape[0]

            ## Remove whitespace characters
            openface_df.columns = openface_df.columns.str.replace(" ", "")

            ## Check the number of datapoints below confidence level
            data_points_below_confidence_score += (openface_df["confidence"] < 0.80).sum()

            ## Count the frames for participant and class
            participant_frame_count += number_of_frames
            total_frame_count[target_class] += number_of_frames

            # Create a new dataframe - 'required_df' that retains all the feature columns that are required from the original dataframe - 'df'
            required_df = openface_df[required_features].copy()
            required_df["participant_id"] = participant
            required_df["class"] = class_labels[target_class]
            required_df["response_video"] = response_video_name
            required_df = required_df[final_features + required_features] ## Reorganize the column order

            ## Downsampling
            # Obtain the last_timestamp for every response_video data
            last_timestamp = required_df['timestamp'].values[-1] # NOTE: this is in seconds
            # print(participant, csv_file, last_timestamp, frequency, last_timestamp * frequency)
            last_timestamp = round(last_timestamp * frequency) / frequency
            # Now we get a list of timestamps once every 'sampling_size' seconds, until the last timestamp
            valid_timestamps = np.arange(0, last_timestamp + rate, rate)

            # A new dataframe to store the downsampled datapoints from the 'required_df'
            downsampled_df = pd.DataFrame()
            for time in valid_timestamps:
                if time == valid_timestamps[0]:
                    # Retrieve the rows between the range (from_time, time]
                    downsampled_df = pd.concat([downsampled_df, required_df[required_df["timestamp"] <= time]])
                    from_time = time
                else:
                    datapoints_within_window = required_df[(required_df["timestamp"] > from_time) & (required_df["timestamp"] <= time)].copy()
                    # Check if there exists any datapoints within the range specified: (from_time, time]
                    if len(datapoints_within_window) == 0:
                        print(f'No datapoints within the specified range! ({from_time}, {time}]')
                    else:
                        # Now that you have datapoints within the specified window range: (from_time, time]
                        # Calculate some statistics on the feature columns present in the dataframe
                        # i.e: mean(), max(), etc.. on the respective feature columns
                        # NOTE: here we reduce the range of dataPoints to a single dataPoint after calculating the statistics
                        for column in datapoints_within_window.columns:
                            # for the feature columns that contain values based on classification
                            # i.e: AU##_c :- columns, find the max value amongst them (i.e: 0 or 1)
                            if '_c' in column:
                                datapoints_within_window.loc[:, column] = datapoints_within_window[column].max()
                            # for all other columns except few, calculate the aggregate value
                            elif column not in ['participant_id', 'class', 'response_video', 'timestamp']:
                                datapoints_within_window.loc[:, column] = datapoints_within_window[column].mean()
                            elif column == 'timestamp':
                                datapoints_within_window.loc[:, column] = time
                        # Now we add the dataPoints_within_window :- that have been reduced to a single datapoint
                        # to the final - downsampled_df
                        downsampled_df = pd.concat([downsampled_df.T, datapoints_within_window.iloc[0, :]], axis = 1).T
                    # Update the range of the time :- to shift the window
                    from_time = time

            downsampled_df.to_csv(downsampled_csv_file_output_path, index=False)
            # break # from file
        data_points_below_confidence_percentage = data_points_below_confidence_score / participant_frame_count * 100
        print(f"Participant {participant}: Datapoints below 80% confidence level = {data_points_below_confidence_percentage:.03f}% ({data_points_below_confidence_score} / {participant_frame_count})")
        # break # from participant

def main():
    final_features, required_features = get_features()

    downsampling_size = 12
    participant_info_path = "../../data/neckface_dataset/participant_log.xlsx"
    dataset_path = "../../data/neckface_openface_dataset/"
    neckface_openface_features_path = dataset_path + "openface_numerical_features_dataset/"
    output_path = dataset_path + f"{downsampling_size}_fps_downsampled_neckface_openface_dataset/"
    
    downsample_features(
        final_features=final_features,
        required_features=required_features,
        downsampling_size=downsampling_size,
        dataset_path=neckface_openface_features_path,
        output_path=output_path
    )
    
    merge_dataframes(
        participant_info_path=participant_info_path,
        downsampling_size=downsampling_size,
        data_path=output_path
    )

if __name__ == "__main__":
    main()