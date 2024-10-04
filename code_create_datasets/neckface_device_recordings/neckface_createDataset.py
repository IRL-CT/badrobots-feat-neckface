## This script considers the timestamps from the neckface device and extracts all the frames from the neckface IR camera recording of the participant session recordings.

import os
import cv2
import traceback
import numpy as np
import pandas as pd
from tqdm import tqdm


class Neckface:
    def __init__(self, data_path, output_path, version):
        self.files_to_ignore = [".DS_Store"]
        self.data_path = data_path
        self.output_path = output_path
        self.version = version
        self.neckface_recordings_path = self.data_path + "neckface_device_recordings/format_mp4/"
        self.unix_timestamp_mappings_path = self.data_path + "failureOccurrence_unix_timeStamp_mapping.xlsx"
        self.unix_timestamp_mappings_df = pd.read_excel(self.unix_timestamp_mappings_path)
        
        ## Class mappings - control (0) vs failure (1)
        self.class_mappings = {
            "ch": 0, # Control Human
            "cr": 0, # Control Robot
            "fh": 1, # Failure Human
            "fr": 1  # Faulure Robot
        }

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

    def preprocess_data_files(self, path, pred_type="pred"):
        ## Load the numpy file and convert it to a df (for easier extraction)
        survey_preds = np.load(path)
        data_pred_df = pd.DataFrame(survey_preds)
        
        """
        In the unix_timestamp_mappings_df, the start and end timestamps are integers (with precision upto milliseconds)
        In the data_pred_df, the feature/column '0' --> unix timestamps
        The values in these columns are separated by a decimal point and
        they are precise to the microsecond.
        
        As a result, a new column 'corrected_timestamps' is created so as to match the type
        """
        data_pred_df["corrected_timestamps"] = (data_pred_df[0] * 1000).astype('int')
        return data_pred_df

    def consider_participant(self, participant_id):
        """Extracts the frame only for participants considered in the study

        Args:
            participant_id (_type_): participant number

        Returns:
            consider_participant (bool): Y/N regarding participant consideration
        """
        excluded_participants = {13, 17, 27, 29, 30}
        # return self.participant_inclusion_dict.get(participant_id) == "Y"
        return participant_id not in excluded_participants

    def process_frames(self):
        """
        Based on the participant start and stop timings obtained from "failureOccurrence_unix_timeStamp_mapping.xlsx" dataset
        For a given participant, for a given stimulus video, the range of start and stop is obtained
        For failure videos, the participant frames are considered only from the moment the failure occurs in the stimulus video.
        
        For a given participant's session recording:
        - We iterate through the record of all stimulus video watched and obtain the start and stop duration (unix timestamps)
        - Based on this, we obtain the indices of the unixTimestamp of when they viewed a stimulus video
        - We then start reading the session recording frame-by-frame
        - We implement a "frame_counter"
        - If the frame_counter falls within this index ranges within the [start_unix, stop_unix], we then convert that frame along with other metadata
        - Once the frames from all stimulus videos are extracted, it is then saved.
        """
        
        ## Participant session recordings
        participant_recordings = [video for video in os.listdir(self.neckface_recordings_path) if video not in self.files_to_ignore and video.endswith(".mp4")]
        ## Participant unix_timestamps
        preds_path = self.data_path + f"preds_{self.version}/"
        frame_counts = {
            "control": 0,
            "failure": 0
        }

        ## Iterate through the participant session recording
        for recording in tqdm(sorted(participant_recordings), total=len(participant_recordings), desc="Processed Participant:"):
            participant_id = int(recording.split("_")[0])
            if participant_id == 28:
                timestamps_path = preds_path + f"{participant_id}-selected/" + f"{participant_id}_survey_1_preds.npy"
            else:
                timestamps_path = preds_path + f"{participant_id}-selected/" + f"{participant_id}_survey_preds.npy"
            recording_file_path = self.neckface_recordings_path + recording
            
            ## Check if the participant is to be included in the study
            include_participant = self.consider_participant(participant_id=participant_id)
            if not include_participant:
                print(f"Participant {participant_id} excluded")
                continue
            
            data_pred_df = self.preprocess_data_files(timestamps_path)
            
            ## Obtain the range of timestamp values belonging to a particular participant
            unix_timestamp_mappings_df = self.unix_timestamp_mappings_df[self.unix_timestamp_mappings_df["participant_id"] == participant_id].drop_duplicates(
                subset=["stimulusVideo_name"],
                keep="first" ## Drop all rows that have the same stimulusVideo_name (when the participant has viewed the stimlus video multiple times) and keep only the first occurrence
            )

            ## Create dirs to store the data for a given participant
            data_pred_output_path = self.output_path + f"{participant_id}/"
            if not os.path.exists(data_pred_output_path):
                print(f"Starting frame extraction for participant {participant_id}")
                os.makedirs(data_pred_output_path)
            else:
                print(f"Frames extracted for participant {participant_id}")
                # continue

            ## Placeholders to store data
            neckface_recording_frames = []
            neckface_recording_labels = []
            neckface_recording_participant_data = []
            neckface_recording_video_name = []

            ## Iterate through each of the stimulus video [start, stop] unix timestamps for the participant
            print(f"Participant: {participant_id}")
            for index, row in unix_timestamp_mappings_df.iterrows():
                stimulus_video_name = row["stimulusVideo_name"]
                stimulus_video_class = stimulus_video_name[:2]
                if stimulus_video_class in self.class_mappings:
                    class_label = self.class_mappings[stimulus_video_class]
                
                ## If the class is == 1, it is a failure stimulus video, consider frames only upon failure
                start_column_name = "responseVideo_unixTime_start" if class_label == 0 else "failureOccurrence_unixTimestamp"
                start_time = int(row[start_column_name])
                end_time = row["responseVideo_unixTime_end"]

                try:
                    ## Obtain the range of timestamps that are valid and fall within the start and stop time
                    ## The data_pred_df is from the ".npy" file created from the reconstruction video
                    pred_df = data_pred_df[(data_pred_df["corrected_timestamps"] >= start_time) & (data_pred_df["corrected_timestamps"] <= end_time)]

                    ## Get the indices of these valid timestamps - for the given stimulus video
                    indices = pred_df.index
                    indices = indices.to_numpy()
                    
                    ## Get the beginning and ending of the stimulus video timestamp index
                    start_index, stop_index = min(indices), max(indices)
                    
                    frame_counter = 0
                    cap = cv2.VideoCapture(recording_file_path)
                    ret = True
                    i = 0
                    while ret:
                        ret, img = cap.read()
                        ## For each video, the all the frames are read
                        ## But when the frame_counter iterator falls within the range of the [beginning, end] of the stimulus video timestamp, we extract only those frames
                        if ret and (start_index <= frame_counter <= stop_index): # and (frame_counter < len(data_pred_df["corrected_timestamps"])):
                            # # Display the frame using OpenCV
                            # cv2.imshow('Video Frame', img)

                            # # Wait for 1 ms between frames and break the loop if 'q' is pressed
                            # if cv2.waitKey(1) & 0xFF == ord('q'):
                            #     break
                            X_data, y_data = self.get_image_data(img, class_label)
                            # print(y_data)
                            neckface_recording_frames.append(X_data)
                            neckface_recording_labels.append(y_data)
                            neckface_recording_participant_data.append(participant_id)
                            neckface_recording_video_name.append(stimulus_video_name)
                            i = frame_counter
                            ## Calculate the frame counts
                            label_name = "control" if class_label == 0 else "failure"
                            frame_counts[label_name] = frame_counts.get(label_name, 0) + 1
                        frame_counter += 1
                    cap.release() ## Close the video file
                    cv2.destroyAllWindows() ## Close all the frame windows
                    # print(f"Number of Frames = {frame_counter}")
                    # print(f"Frames to consider = {len(indices)}")
                    # print(f"Frames considered = {i}")
                    # break ## From a given stimulus video
                except Exception as e:
                    print(f"Exception thrown for video: {stimulus_video_name} for participant {participant_id}")
                    traceback.print_exc()
            # # Write to file after processing each responseVideo of the participant in 'append' mode and reset the pixel_data variable to be = empty to prevent memory overleak
            with open(f'{data_pred_output_path}pixel_data.npy', 'wb') as f:
                np.save(f, np.array(neckface_recording_frames))
            with open(f'{data_pred_output_path}label_data.npy', 'wb') as f:
                np.save(f,  np.array(neckface_recording_labels))
            with open(f'{data_pred_output_path}participant_data.npy', 'wb') as f:
                np.save(f,  np.array(neckface_recording_participant_data))
            with open(f'{data_pred_output_path}video_name_data.npy', 'wb') as f:
                np.save(f,  np.array(neckface_recording_video_name))
            print(f'Saved the numpy files for participant {participant_id} where the number of frames in the video = {len(neckface_recording_frames)}')
            # break ## From the participant session
        print(f"Frame Counts from all participants = {frame_counts}")

def main():
    data_path = "../../data/neckface_dataset/"
    version = "v2"
    output_path = data_path + f"neckface_device_frame_dataset/{version}/"
    neckface_obj = Neckface(
        data_path=data_path,
        output_path=output_path,
        version=version
    )
    neckface_obj.process_frames()

if __name__ == "__main__":
    main()