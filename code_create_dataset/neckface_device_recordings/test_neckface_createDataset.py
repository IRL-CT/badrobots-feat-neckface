import os
import cv2
import json
import math
import traceback
import numpy as np
import pandas as pd
from pprint import pprint
from tqdm import tqdm
from datetime import datetime

class Neckface:
    def __init__(self, data_path, output_path):
        self.files_to_ignore = [".DS_Store"]
        self.data_path = data_path
        self.output_path = output_path
        self.preds_path = self.data_path + "pred_outputs/all_participant_preds.csv"
    
    def get_survey_start_time(self, participant_id, df):
        filtered_df = df[df['participant_id'] == participant_id]
        
        if not filtered_df.empty:
            survey_start_time = filtered_df["corrected_timestamps"].min()
            return survey_start_time
        else:
            return None

    def get_start_stop_unix_timestamps(self, participant_id, stimulus_video_name, df):
        # Filter the DataFrame based on participant_id and stimulus_video_name
        filtered_df = df[(df['participant_id'] == participant_id) & (df['stimulus_video_name'] == stimulus_video_name)]
        
        if not filtered_df.empty:
            # Extract the start and stop timestamps
            start_timestamp = filtered_df['corrected_timestamps'].min()
            stop_timestamp = filtered_df['corrected_timestamps'].max()
            return start_timestamp, stop_timestamp
        else:
            return None, None
    
    def convert_unix_to_seconds(self, unix_timestamp):
        # If the timestamp is in milliseconds or microseconds, adjust accordingly
        if unix_timestamp > 1e12:
            # Assuming it's in microseconds
            return unix_timestamp / 1e6
        elif unix_timestamp > 1e9:
            # Assuming it's in milliseconds
            return unix_timestamp / 1e3
        else:
            # It's already in seconds
            return unix_timestamp

    def get_duration(self, unix_start, unix_stop):
        start_time_sec = unix_start / 1000
        stop_time_sec = unix_stop / 1000
        
        duration = stop_time_sec - start_time_sec
        return round(duration, 2)

    def get_data(self):
        preds_df = pd.read_csv(self.preds_path)
        recordings_path = self.data_path + "neckface_device_recordings/format_mp4"
        recordings = [video for video in os.listdir(recordings_path) if video not in self.files_to_ignore and video.endswith(".mp4")]
        participant_ids = preds_df["participant_id"].unique()
        stimulus_video_names = preds_df["stimulus_video_name"].unique()
        
        participant_start_times = {}
        participant_viewing_dict = {}
        for participant in sorted(participant_ids):
            print(f"Participant ID: {participant}")
            participant_min_start_unix = float("inf")
            survey_start_time = self.get_survey_start_time(participant_id=participant, df=preds_df)
            participant_viewing_dict[participant] = {}
            for stimulus_video in sorted(stimulus_video_names):
                start, stop = self.get_start_stop_unix_timestamps(
                    participant_id=participant,
                    stimulus_video_name=stimulus_video,
                    df=preds_df
                )
                viewing_duration = self.get_duration(start, stop)
                start_duration = self.get_duration(survey_start_time, start)
                # print(f"Start: {start_duration} | Duration: {viewing_duration}")
                
                participant_viewing_dict[participant][stimulus_video] = (start_duration, viewing_duration)
                # duration = self.get_duration(unix_start=start, unix_stop=stop)
                # print(f"Stimulus Video: {stimulus_video} - Start: {start} | Stop: {stop} | Duration: {duration}")
                # start_s, stop_s = self.convert_unix_to_seconds(start), self.convert_unix_to_seconds(stop)
                # print(f"Stimulus Video: {stimulus_video} - Start: {start_s} | Stop: {stop_s}")
                # print(f"Stimulus Video: {stimulus_video} - Start: {start_s} | Stop: {stop_s}")

            # print(f"Participant ID: {participant} | Survey Start && Min Start: {survey_start_time == participant_min_start_unix}")
        # pprint(participant_viewing_dict)
        
        sorted_participant_timestamps = {}
        # Iterate over participants
        for participant, videos in participant_viewing_dict.items():
            # Sort videos based on start_duration (assuming tuple structure is (start_duration, viewing_duration))
            sorted_videos = sorted(videos.items(), key=lambda item: item[1][0])  # Sort by start_duration

            # Create sorted participant data
            sorted_participant_timestamps[f"{participant}"] = {
                stimulus_video: {
                    'start_duration': start_duration,
                    'viewing_duration': viewing_duration
                }
                for stimulus_video, (start_duration, viewing_duration) in sorted_videos
            }
            # Print sorted videos
            # for stimulus_video, durations in sorted_videos:
            #     start_duration, viewing_duration = durations
            #     print(f"Stimulus Video: {stimulus_video} | Start Duration: {start_duration} | Viewing Duration: {viewing_duration}")
        
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        sorted_participant_timestamps_output_path = self.output_path + "sorted_participant_timestamps.json"
        with open(sorted_participant_timestamps_output_path, "w") as json_file:
            json.dump(sorted_participant_timestamps, json_file, indent=4)
        
if __name__ == "__main__":
    data_path = "../../data/neckface_dataset/"
    output_path = data_path + "neckface_device_frame_dataset/"
    neckface_obj = Neckface(
        data_path=data_path,
        output_path=output_path
    )
    neckface_obj.get_data()