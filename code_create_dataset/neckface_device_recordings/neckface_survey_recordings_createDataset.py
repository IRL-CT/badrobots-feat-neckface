## This script extracts the frames from the NeckFace survey study recordings
## The frames are matched to be almost a 1:1 mapping of the timestamps from the neckface_survey_prediction timestamps

import os
import cv2
import traceback
import numpy as np
import pandas as pd
from tqdm import tqdm

files_to_ignore = [".DS_Store"]

class NeckfaceSurvey:
    def __init__(self, data_path, survey_recordings_path, output_path):
        self.data_path = data_path
        self.survey_recordings_path = survey_recordings_path
        self.output_path = output_path

    def get_image_data(self, image, label):
        """Converts the image frame into a numpy array

        Args:
            image (cv2 obj): image
            label (int): stimulus video label

        Returns:
            X_data: image converted into a np.array
            y_data: label converted into a np.array
        """
        X_data = []
        y_data = []
        
        try:
            X_data.append(image)
            y_data.append(label)
        except Exception as e:
            traceback.print_exc()
        return np.array(X_data), np.array(y_data)

    def get_participant_id_mappings(self, participant_log_path):
        """Since the survey recording dataset is named according to the participant's qualtircs_id, we map these ids to the participant's study_id

        Args:
            participant_log_path (str): Participant Study Log File path

        Returns:
            qualtrics_id_to_participant_id (Dict): {qualtrics_id: participant_id}
        """
        participant_info_df = pd.read_excel(participant_log_path)
        participant_info_df.fillna(0, inplace=True)
        participant_info_df[["participant_number", "qualtrics_id"]] = participant_info_df[["participant_id", "qualtrics_participant_id"]].astype("int")
        
        # Create a mapping from qualtrics_id to participant_number
        qualtrics_id_to_participant_id = dict(zip(participant_info_df["qualtrics_id"], participant_info_df["participant_number"]))
        return qualtrics_id_to_participant_id

    def get_participant_start_stop_unix_timestamps(self, participant_recording_timestamp_info_path):
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

    def get_neckface_preds(self, neckface_data_path, participant_id):
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

    def process_survey_recordings(
        self, 
        participant_log_path,
        neckface_timestamps_path,
        participant_recording_timestamp_info_path, 
        survey_recordings_path, 
    ):
        qualtircs_id_to_participant_id = self.get_participant_id_mappings(participant_recording_timestamp_info_path)
        participants_to_exclude = {13, 17, 27, 29, 30}
        participants_to_consider = [participant for participant in os.listdir(survey_recordings_path) if (participant not in files_to_ignore) and (qualtircs_id_to_participant_id[int(participant)] not in participants_to_exclude)]
        
        participant_start_stop_ts = self.get_participant_start_stop_unix_timestamps(participant_recording_timestamp_info_path=participant_recording_timestamp_info_path)
        processing_metadata = {}
        frame_counts = {
            "control": 0,
            "failure": 0
        }
        for participant_qualtrics_id in tqdm(sorted(participants_to_consider), total=len(participants_to_consider), desc="Processed Participants: "):
            participant_data_path = survey_recordings_path + participant_qualtrics_id + "/mp4StudyVideo/"
            stimulus_videos = [video for video in os.listdir(participant_data_path) if video not in files_to_ignore]
            neckface_preds_df = self.get_neckface_preds(neckface_data_path = neckface_timestamps_path, participant_id=qualtircs_id_to_participant_id[int(participant_qualtrics_id)])
            
            neckface_survey_recording_frames = []
            neckface_survey_recording_labels = []
            neckface_survey_recording_participant_data = []
            neckface_survey_recording_video_name = []
            
            participant_output_path = self.output_path + f"/{qualtircs_id_to_participant_id[int(participant_qualtrics_id)]}/"
            if not os.path.exists(participant_output_path):
                os.makedirs(participant_output_path)

            for stimulus_video in stimulus_videos:
                stimulus_video_path = participant_data_path + stimulus_video
                stimulus_video_name = stimulus_video.split(".")[0]
                participant_start_time = participant_start_stop_ts[qualtircs_id_to_participant_id[int(participant_qualtrics_id)]][stimulus_video_name]["unix_timestamp_start"]
                participant_class = participant_start_stop_ts[qualtircs_id_to_participant_id[int(participant_qualtrics_id)]][stimulus_video_name]["class_label"]
                
                ## Filter by video to ensure the timestamps are monotonically increasing
                nf_df = neckface_preds_df[neckface_preds_df["stimulus_video_name"] == stimulus_video_name]
                
                frame_counter = 0
                cap = cv2.VideoCapture(stimulus_video_path)
                ret = True
                for neckface_ts in nf_df["corrected_timestamps"]:
                    while ret:
                        ret, image = cap.read()
                        if not ret:
                            break # end of video
                        frame_timestamp = participant_start_time + (frame_counter * 1000 // 30)
                        frame_counter += 1
                        if frame_timestamp >= neckface_ts:
                            # # Display the frame using OpenCV
                            # cv2.imshow('Video Frame', image)

                            # # Wait for 1 ms between frames and break the loop if 'q' is pressed
                            # if cv2.waitKey(1) & 0xFF == ord('q'):
                            #     break
                            X_data, y_data = self.get_image_data(image, participant_class)
                            neckface_survey_recording_frames.append(X_data)
                            neckface_survey_recording_labels.append(y_data)
                            neckface_survey_recording_participant_data.append(qualtircs_id_to_participant_id[int(participant_qualtrics_id)])
                            neckface_survey_recording_video_name.append(stimulus_video_name)
                            ## Track Frame Counts
                            frame_counts[participant_class] = frame_counts.get(participant_class, 0) + 1
                            break # move to the next neckface timestamp
                # break # from video
            # # Write to file after processing each responseVideo of the participant in 'append' mode and reset the pixel_data variable to be = empty to prevent memory overleak
            processing_metadata[qualtircs_id_to_participant_id[int(participant_qualtrics_id)]] = len(neckface_survey_recording_frames)
            with open(f'{participant_output_path}pixel_data.npy', 'wb') as f:
                np.save(f, np.array(neckface_survey_recording_frames))
            with open(f'{participant_output_path}label_data.npy', 'wb') as f:
                np.save(f,  np.array(neckface_survey_recording_labels))
            with open(f'{participant_output_path}participant_data.npy', 'wb') as f:
                np.save(f,  np.array(neckface_survey_recording_participant_data))
            with open(f'{participant_output_path}video_name_data.npy', 'wb') as f:
                np.save(f,  np.array(neckface_survey_recording_video_name))
            # break # from participant
        print(f"Frame Label Counts: {frame_counts}")
        print(f"Total frames extracted from all participants are as follows: {sum(processing_metadata.values())}")
        print(f"Number of frames extracted per participant is as follows: \n{processing_metadata}")

    def rename_response_videos(self, participant_recording_timestamp_info_path, survey_recordings_path):
        qualtircs_id_to_participant_id = self.get_participant_id_mappings(participant_recording_timestamp_info_path)
        participants_to_exclude = {13, 17, 27, 29, 30}
        participants_to_consider = [participant for participant in os.listdir(survey_recordings_path) if (participant not in files_to_ignore) and (qualtircs_id_to_participant_id[int(participant)] not in participants_to_exclude)]
        
        participant_recordings_info_df = pd.read_excel(participant_recording_timestamp_info_path)
        
        for participant_qualtrics_id in tqdm(sorted(participants_to_consider), total=len(participants_to_consider), desc="Processed Participants: "):
            participant_data_path = survey_recordings_path + participant_qualtrics_id + "/mp4StudyVideo/"
            stimulus_videos = [video for video in os.listdir(participant_data_path) if video not in files_to_ignore]
            for stimulus_video in stimulus_videos:
                stimulus_video_path = participant_data_path + stimulus_video
                # Find the corresponding row in the DataFrame where responseVideo matches the stimulus_video name
                match_row = participant_recordings_info_df[participant_recordings_info_df['responseVideo'] == stimulus_video]
                if not match_row.empty:
                    stimulusVideo_name = match_row["stimulusVideo_name"].values[0]
                    new_video_path = os.path.join(participant_data_path, f"{stimulusVideo_name}.mp4")
                    os.rename(stimulus_video_path, new_video_path)
def main():
    data_path = "../../data/neckface_dataset/"
    survey_recordings_path = data_path + "neckface_survey_recordings/"
    participant_log_path = data_path + "participant_log.xlsx"
    participant_recording_timestamp_info_path = data_path + "failureOccurrence_unix_timeStamp_mapping.xlsx"
    neckface_timestamps_path = data_path + "pred_outputs_v2/all_participant_preds.csv"
    output_path = "../../data/neckface_survey_dataset/"

    survey_obj = NeckfaceSurvey(
        data_path=data_path, 
        survey_recordings_path = survey_recordings_path,
        output_path=output_path,
    )

    # participant_id_mappings = survey_obj.get_participant_id_mappings(
    #     participant_log_path=participant_log_path
    # )
    
    # start_stop_timestamps = survey_obj.get_participant_start_stop_unix_timestamps(
    #     participant_recording_timestamp_info_path=participant_recording_timestamp_info_path
    # )
    
    survey_obj.process_survey_recordings(
        participant_log_path=participant_log_path,
        neckface_timestamps_path=neckface_timestamps_path,
        participant_recording_timestamp_info_path=participant_recording_timestamp_info_path,
        survey_recordings_path=survey_recordings_path,
    )
    # survey_obj.rename_response_videos(
    #     participant_recording_timestamp_info_path=participant_recording_timestamp_info_path,
    #     survey_recordings_path=survey_recordings_path
    # )
    

if __name__ == "__main__":
    main()