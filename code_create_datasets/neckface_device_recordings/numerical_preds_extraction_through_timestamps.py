import os
import ast
import traceback
import numpy as np
import pandas as pd
from tqdm import tqdm

class UnixTimestamps:
    def __init__(self, data_path, output_path, version):
        self.files_to_ignore = [".DS_Store"]
        self.data_path = data_path
        self.output_path = output_path
        self.version = version
        self.unix_timestamp_mappings_path = self.data_path + "failureOccurrence_unix_timeStamp_mapping.xlsx"
        self.unix_timestamp_mappings_df = pd.read_excel(self.unix_timestamp_mappings_path)
        
        ## Class mappings - control (0) vs failure (1)
        self.class_mappings = {
            "ch": 0, # Control Human
            "cr": 0, # Control Robot
            "fh": 1, # Failure Human
            "fr": 1  # Faulure Robot
        }
        
        ## NOTE: Not needed anymore as we are excluding participants manually: participant_ids: (13, 17, 27, 29, 30)
        # ## Read the participant logs to create datasets for only the participants who are to be considered in the study
        # self.participant_log_path = self.data_path + "participant_log.xlsx"
        # self.participant_log_df = pd.read_excel(self.participant_log_path)
        # ## Drop all the participants who don't have a "id"
        # self.participant_log_df = self.participant_log_df.dropna(subset=["participant_number"])
        # ## Fill in the values for participant inclusion (those who are to be excluded have this value set to "N")
        # self.participant_log_df["participant_neckface"].fillna("Y", inplace=True)
        # ## Create a dict of the participant inclusion information
        # ## key: value :: participant_id: inclusion (Y or N)
        # self.participant_inclusion_dict = {int(key): value for key, value in self.participant_log_df.set_index("participant_number")["participant_neckface"].to_dict().items() if key}
    
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
        data_pred_df["corrected_timestamps"] = (data_pred_df[0]  * 1000).astype('int')
        return data_pred_df
    
    def consider_participant(self, participant_id):
        """Extracts the frame only for participants considered in the study

        Args:
            participant_id (_type_): participant number

        Returns:
            consider_participant (bool): Y/N regarding participant consideration
        """
        excluded_participants = {13, 17, 27, 29, 30}
        return participant_id not in excluded_participants
        # return self.participant_inclusion_dict.get(participant_id) == "Y"

    def extract_response_video_frames(self):
        """
        For each participant, from the neckface session recordings, this method extracts:
        1. Participant Response Frames belonging to a specific stimulus video
        2. Participant Response Frames belonging to all stimulus videos
        
        By, identifying the range of frames that fall within the [start, end] unix_timestamps.
        
        The [start, end] unix_timestamps are obtained from the S3 recordings 
        of when the participant actually started viewing the stimulusVideo.
        This data is present in the unix_timestamp_mappings_df.
        """

        participant_preds_path = self.data_path + f"preds_{self.version}/"
        participant_folders = [participant for participant in os.listdir(participant_preds_path) if participant not in self.files_to_ignore]

        all_participant_frames = []
        for participant in sorted(participant_folders):
            participant_id = int(participant.split("-")[0])
            
            ## Check if the given participant is included in the study when creating the dataset
            include_participant = self.consider_participant(participant_id)
            if not include_participant:
                print(f"Participant ID: {participant_id} exlcuded.")
                continue
                
            data_preds_path = participant_preds_path + f"{participant}/"
            data_pred_files = [pred_file for pred_file in os.listdir(data_preds_path) if pred_file not in self.files_to_ignore]

            for data_pred_file in data_pred_files:
                pred_type = data_pred_file.split("_")[1]
                data_pred_file_path = data_preds_path + data_pred_file
                data_pred_df = self.preprocess_data_files(data_pred_file_path)
                if pred_type == "end":
                    continue

                ## Obtain the range of timestamp values beloning to a particular participant
                unix_timestamp_mappings_df = self.unix_timestamp_mappings_df[self.unix_timestamp_mappings_df["participant_id"] == participant_id].drop_duplicates(
                    subset=["stimulusVideo_name"],
                    keep="first" ## Drop all rows that have the same stimulusVideo_name (when the participant has viewed the stimlus video multiple times) and keep only the first occurrence
                )

                data_pred_output_path = self.output_path + f"pred_outputs_{self.version}/" + f"{participant_id}/" + f"{participant_id}_{pred_type}/"
                if not os.path.exists(data_pred_output_path):
                    os.makedirs(data_pred_output_path)

                all_stimulusVideo_frames = []
                for index, row in unix_timestamp_mappings_df.iterrows():
                    stimulus_video_name = row["stimulusVideo_name"]
                    stimulus_video_class = stimulus_video_name[:2]

                    if stimulus_video_class in self.class_mappings:
                        class_label = self.class_mappings[stimulus_video_class]

                    ## If the class is == 1, it is a failure stimulus video, consider frames only upon failure
                    start_column_name = "responseVideo_unixTime_start" if class_label == 0 else "failureOccurrence_unixTimestamp"
                    start_time = int(row[start_column_name])
                    end_time = row["responseVideo_unixTime_end"]

                    pred_df = data_pred_df[(data_pred_df["corrected_timestamps"] >= start_time) & (data_pred_df["corrected_timestamps"] <= end_time)]

                    ## Create few new columns to store participant metadata
                    stimulus_response_df = pred_df.assign(
                            participant_id=participant_id,
                            stimulus_video_name=stimulus_video_name,
                            class_label=class_label,
                            pred_type=pred_type
                        )

                    # New column order
                    new_column_order = ['participant_id', 'stimulus_video_name', 'class_label', 'pred_type'] + [col for col in pred_df.columns if col not in ['participant_id', 'stimulus_video_name', 'class_label', 'pred_type']]
                    
                    ## Rearrange columns
                    stimulus_response_df = stimulus_response_df[new_column_order]
                    
                    ## Save the current participant response to a particular stimulus video to a csv
                    if not stimulus_response_df.empty:
                        stimulus_response_df.to_csv(data_pred_output_path + f"{participant_id}_{stimulus_video_name}_{pred_type}.csv", index=False)
                        all_stimulusVideo_frames.append(stimulus_response_df)

                ## Concatenate the participant responses to all stimulus videos to a single csv
                if len(all_stimulusVideo_frames) != 0:
                    all_stimulusVideo_frames_df = pd.concat(all_stimulusVideo_frames, ignore_index=True)
                    all_stimulusVideo_frames_df.to_csv(data_pred_output_path + f"{participant_id}_all_stimulusVideo_frames_{pred_type}.csv", index=False)
                    all_participant_frames.append(all_stimulusVideo_frames_df)
                else:
                    print(f"Participant: {participant_id}, Pred Type: {pred_type}, Extraction Empty")
        all_stimulusVideo_df = pd.concat(all_participant_frames, ignore_index=True)
        all_stimulusVideo_df.to_csv(f"{self.output_path}" + f"/pred_outputs_{self.version}/" + "all_participant_preds.csv", index=False)

if __name__=="__main__":
    data_path = "../../data/neckface_dataset/"
    output_path = "../../data/neckface_dataset/"
    version = "v2"
    unix_timestamps_obj = UnixTimestamps(
            data_path=data_path,
            output_path=output_path,
            version=version
        )
    unix_timestamps_obj.extract_response_video_frames()