import h5py
import os
import pandas as pd
import numpy as np

# Replace with your actual file path
folder_path = '../../data/vit_features/'
files = os.listdir(folder_path)

#create a big DF, with structure 'participant_id,stimulus_video_name,class_label,pred_type' and then the features
vit_df = pd.DataFrame(columns=['participant_id','stimulus_video_name','class_label','pred_type'])

for file in files:
    file_path = os.path.join(folder_path, file)
    print(f"Checking file: {file_path}")
    # Open the h5 file
    with h5py.File(file_path, 'r') as f:

        # See what's inside
        print(list(f.keys()))
        # Get the dataset
        features = f['features']
        

        # Print the shape (dimensions)
        print(f"Shape of features: {features.shape}")
        
        # Optional: Print other information
        print(f"Data type: {features.dtype}")
        #print(f"Total size in memory: {features.size * features.dtype.itemsize / (1024*1024):.2f} MB")

        # Get the participant_id, stimulus_video_name, class_label, pred_type
        participant_id = file.split('.')[0]
        stimulus_video_name = 'na'
        class_label = 'na'
        pred_type = 'na'
        time = 'na'
        #turn features into a numpy array
        features = np.array(features)

        #reshape from (l,1,d) to (l,d)
        features = features.reshape(features.shape[0],features.shape[2])
        #add to the df, start with participant id, repeated for d times
        temp_df = pd.DataFrame(features)
        temp_df['participant_id'] = participant_id
        temp_df['stimulus_video_name'] = stimulus_video_name
        temp_df['class_label'] = class_label
        temp_df['pred_type'] = pred_type
        temp_df['time'] = time
        #add to the big df
        part_df = temp_df[['participant_id','stimulus_video_name','class_label','pred_type','time']]
        part_df = pd.concat([part_df,temp_df[temp_df.columns[:-5]]],axis=1)
        vit_df = pd.concat([vit_df,part_df],axis=0)


        f.close()

print(vit_df.shape)
print(vit_df.head())
#save the df
vit_df.to_csv('../../data/vit_features.csv',index=False)