import os
import cv2
import numpy as np

files_to_ignore = [".DS_Store", "sorted_participant_timestamps.json"]


def display_frames(participant_numpy_file_path):
    participant_folders = [folder for folder in os.listdir(participant_numpy_file_path) if folder not in files_to_ignore]
    
    class_label_counts = {}
    frames_count = 0
    
    for participant_folder in participant_folders:
        if participant_folder != "2":
            continue
        participant_folder_path = participant_numpy_file_path + participant_folder + "/"
        
        participant_frame_path = participant_folder_path + "pixel_data.npy"
        participant_label_path = participant_folder_path + "label_data.npy"
        participant_data_path = participant_folder_path + "participant_data.npy"
        participant_video_name_path = participant_folder_path + "video_name_data.npy"
        
        # Load the frame data & check its dimensions
        pixel_data = np.load(participant_frame_path)
        print(f'\nLen of pixel_data : {len(pixel_data)}')
        print(f'pixel_data.shape : {pixel_data.shape}')
        first_frame = pixel_data[0][0]
        print(f'Shape of first frame: {first_frame.shape}')

        # Load the label data & check its dimensions
        frame_label = np.load(participant_label_path)
        print(f'\nLen of frame_label: {len(frame_label)}')
        print(f'shape of frame_label: {frame_label.shape}')
        first_label = frame_label[0]
        print(f'Shape of first_label: {first_label.shape}')
        print(f'First frame label value: {first_label}')

        # Load the participant data & check its dimensions
        participant_data = np.load(participant_data_path)
        print(f'\nLen of participant_data: {len(participant_data)}')
        print(f'shape of participant_data: {participant_data.shape}')
        first_participant_data = participant_data[0]
        print(f'Shape of first_participant_data: {first_participant_data.shape}')
        print(f'first_participant_data value: {first_participant_data}')

        video_name_data = np.load(participant_video_name_path)
        print(f"\nLen of Video Name Data: {len(video_name_data)}")
        print(f"Shape of Video Name Data: {video_name_data.shape}")
        first_video_data = video_name_data[0]
        print(f"Len of First Video Name Data: {len(first_video_data)}")
        print(f"Shape of First Video Name Data: {first_video_data.shape}")
        print(f"First video data value = {first_video_data}")
        
        print(f"Displaying frames for participant : {first_participant_data}")
        # # Display the frames of a given participant
        # x = 0
        for i in range(len(pixel_data)):
            frame = pixel_data[i][0]
            label = frame_label[i]
            video_name = video_name_data[i]
            frames_count += 1
            class_label_counts[label[0]] = class_label_counts.get(label[0], 0) + 1
            ## Test for just 1 response video
            if video_name != "fh1":
                continue
            cv2.imshow(f'Frame: {i} | Label: {label} | Participant: {first_participant_data} | {video_name}', frame)
            cv2.waitKey(100)
            cv2.destroyWindow(f'Frame: {i} | Label: {label} | Participant: {first_participant_data} | {video_name}')
        cv2.destroyAllWindows()
        break # from participant
    print(f"Class Label Count: {class_label_counts}")
    print(f"Total number of frames = {frames_count}")

def main():
    # For Neckface IR recording dataset
    version = "v2"
    data_path = f"../../data/neckface_dataset/neckface_device_frame_dataset/{version}/" 
    display_frames(data_path)

    # ## For Neckface Survey recording dataset
    data_path = f"../../data/neckface_survey_dataset/"
    display_frames(data_path)


if __name__ == "__main__":
    main()