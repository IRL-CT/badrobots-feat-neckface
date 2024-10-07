### The below script - identifies all the OpenFace Facial Extraction Data files from the participant response videos
### Copies and moves the ".csv" files containing the numerical facial feature datasets for each participant into a new directory
import os
import glob
import shutil

files_to_ignore = [".DS_Store"]

def extract_numerical_feature_files(dataset_path, output_path):
    participants = [participant for participant in os.listdir(dataset_path) if participant not in files_to_ignore]
    print(f"# of NeckFace Participants: {len(participants)}")
    
    for participant in sorted(participants):
        participant_data_path = dataset_path + participant + "/"
        
        ## Identify all the numerical facial feature datasets from openface
        participant_files = [file for file in os.listdir(participant_data_path) if file not in files_to_ignore and file.endswith('.csv')]
        # print(participant, len(participant_files))
        
        participant_output_path = output_path + participant + "/"
        if not os.path.exists(participant_output_path):
            os.mkdir(participant_output_path)

        print(f"Moving Numerical Feature Files for P: {participant}, # of Files = {len(participant_files)}")
        for csv_file in sorted(participant_files):
            csv_file_path = os.path.join(participant_data_path, csv_file)
            csv_destination_path = os.path.join(participant_output_path, csv_file)
            shutil.copy(
                src=csv_file_path,
                dst=csv_destination_path
            )
            # break # from feature file
        # break # from participant

def main():
    dataset_path = "../../data/neckface_openface_dataset/openface_dataset/"
    output_path = "../../data/neckface_openface_dataset/openface_numerical_features_dataset/"
    
    extract_numerical_feature_files(
        dataset_path=dataset_path,
        output_path=output_path
    )

if __name__=="__main__":
    main()