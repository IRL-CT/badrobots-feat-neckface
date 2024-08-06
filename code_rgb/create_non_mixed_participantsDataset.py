import os
import cv2
import random
import datetime
import traceback
import numpy as np
# import tensorflow as tf
from tqdm import tqdm

files_to_ignore = [".DS_Store"]

def getParticipants(participant_directory_path):
    """
    Get a sorted list of participants in the given directory.

    Args:
        participant_directory_path (str): Path to the directory containing participant data.

    Returns:
        list: Sorted list of participant IDs.
    """
    participants = sorted(
        participant_folder for participant_folder in os.listdir(participant_directory_path) if
        participant_folder not in files_to_ignore)
    return participants

def moveImages(participant_data_directory_path, participants, output_path):
    """
    Move images and create metadata files for a given dataset type.

    Args:
        participant_data_directory_path (str): Path to the directory containing participant data.
        participants (list): List of participant IDs.
        output_path (str): Output path for storing images and metadata.
    """

    # Iterate through the participants in the given dataset fold and read the pixel and label data from the .npy files
    # Then save this as an image to the respective fold directory
    # Frames have a naming convention of:
    # "{participant_id}_{frame_number}_{video_name_of_the_corresponding_frame}_{frame_label}.png"
    for participant in tqdm(participants, total=len(participants), desc="Processing Participants: "):

        # Path where the frames are to be stored
        image_output_path = output_path + f"{participant}/" + "frames/"
        if not os.path.exists(image_output_path):
            os.makedirs(image_output_path)

        # Path where frames metadata text files are to be stored
        metadata_output_path = output_path + f"{participant}/" + "meta/"
        if not os.path.exists(metadata_output_path):
            os.makedirs(metadata_output_path)

        participant_data_path = participant_data_directory_path + participant
        pixel_data_path = participant_data_path + "/pixel_data.npy"
        label_data_path = participant_data_path + "/label_data.npy"
        video_name_data_path = participant_data_path + "/video_name_data.npy"

        try:
            # Read Frame, Label data, and Video Name of the frame it belongs to
            pixel_data = np.load(pixel_data_path)
            frame_label = np.load(label_data_path)
            video_name_data = np.load(video_name_data_path)
        except Exception as e:
            traceback.print_exc()
            pass

        for i in range(len(pixel_data)):
            frame = pixel_data[i][0]
            label = frame_label[i][0]
            video_name = video_name_data[i]

            frame_name = f"{participant}_{i}_{video_name}_{label}.png"
            
            frame_output_path = image_output_path + f"{label}/"
            if not os.path.exists(frame_output_path):
                os.mkdir(frame_output_path)
            frame_output_path = frame_output_path + frame_name
            frame_annotation = frame_output_path + f" {label}" + "\n"

            # Write the path of the frame and the frame label to the metadata file for the Dataloader class's
            # annotation field
            with open(f"{metadata_output_path}/metadata.txt", "a") as metadata_file:
                metadata_file.write(frame_annotation)

            cv2.imwrite(frame_output_path, frame)


def makeCustomDataset(participant_data_path, output_path):
    """
    Create a custom dataset with training, validation, and testing splits.

    Args:
        participant_data_path (str): Path to the directory containing participant data.
        output_path (str): Output path for storing images and metadata.
    """
    study_participants = getParticipants(participant_data_path)

    moveImages(
        participant_data_directory_path=participant_data_path,
        participants=study_participants,
        output_path=output_path,
    )

def main():
    """
    Main function to create a custom dataset from study data.
    """

    # Define paths and other metadata required to process and read the study data
    color_channel = "BGR"
    data_frame_rate = 30
    dataset_path = "../../../data/neckface_device_frame_dataset/"
    output_directory = "../../../data/"

    # Define study paths
    # NeckFace
    output_path = output_directory + f"final_study_data_{color_channel}_{data_frame_rate}fps/"
    
    # Path where the new dataset is to be created at
    neckface_model_output = output_directory + f"non_mixed_superbad_custom_dataset/data_prefix_{data_frame_rate}_fps/"
    # superbad_model_output = output_directory + f"non_mixed_superbad_custom_dataset/data_prefix_{data_frame_rate}_fps/"

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # if not os.path.exists(superbad_model_output):
    #     os.makedirs(superbad_model_output)

    # Change the participant_data_path as per the required study whose dataset is to be created for model training with DataLoaders.
    makeCustomDataset(
        participant_data_path=dataset_path,
        output_path=output_path
    )

    # makeCustomDataset(
    #     participant_data_path=superbad_frame_data_path,
    #     output_path=superbad_model_output
    # )


if __name__ == "__main__":
    main()
