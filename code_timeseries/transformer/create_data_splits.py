import torch
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler

'''
Generates and returns:
- an array of sequences of data
- an array of corresponding target values

Requires:
- an array of input data that will be used to create sequences 
- an array of target values corresponding to the data
- an array of ids or sessions that will be used to group sequences
- an integer equal to the length of each sequence that will be created.
'''
def create_sequences(data, target, sessions, sequence_length):
    sequences = []
    targets = []

    unique_sessions = np.unique(sessions)
    for session in unique_sessions:
        session_indices = np.where(sessions == session)[0]
        session_data = data[session_indices]
        session_target = target[session_indices]

        if len(session_data) >= sequence_length:
            for i in range(len(session_data) - sequence_length + 1):
                sequences.append(session_data[i : i + sequence_length])
                targets.append(session_target[i + sequence_length - 1])
    
    return np.array(sequences), np.array(targets)

def create_data_splits(df, num_folds=5, fold_no=0, seed_value=42, sequence_length=1):
    
    try:
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)

        # Make a copy of the DataFrame with reset index to avoid duplicate indices
        df = df.copy().reset_index(drop=True)
        print(df.columns)
        
        # Extract features and target
        features = df.iloc[:, 5:]
        target = df.iloc[:, 2].values.astype('int')
        # Extract sessions - assuming this is a column in the dataframe
        # If it's the 'participant_id', use that, otherwise use an appropriate column
        sessions = df['participant_id'].values  # Change this if sessions are stored in a different column
        
        print("Features shape:", features.shape)
        print('features head', features.head())
        
        # Get unique participants and shuffle them
        unique_participants = np.unique(df['participant_id'].values)
        num_participants = len(unique_participants)
        print('unique participants', unique_participants)
        
        if num_participants < num_folds:
            raise ValueError(f"Number of participants ({num_participants}) is less than number of folds ({num_folds})")
        
        # Shuffle participants
        np.random.shuffle(unique_participants)

        # Calculate sizes for each set
        val_size = int(np.ceil(0.2 * num_participants))  # 20% for validation
        test_size = int(np.ceil(0.1 * num_participants))  # 10% for test
        train_size = num_participants - val_size - test_size  # Remaining (~70%) for training
        
        # Make sure we have at least one participant in each set
        if val_size < 1 or test_size < 1 or train_size < 1:
            raise ValueError(f"Not enough participants ({num_participants}) to create valid splits")
        
        # Create rolling folds
        val_folds = []
        test_folds = []
        
        for i in range(num_folds):
            # For validation set, take consecutive 20% slice, wrap around if needed
            val_start = (i * val_size) % num_participants
            if val_start + val_size <= num_participants:
                val_fold = unique_participants[val_start:val_start + val_size]
            else:
                # Wrap around for validation set
                val_fold = np.concatenate((
                    unique_participants[val_start:],
                    unique_participants[:val_size - (num_participants - val_start)]
                ))
            
            # For test set, take next consecutive 10% slice, wrap around if needed
            test_start = (val_start + val_size) % num_participants
            if test_start + test_size <= num_participants:
                test_fold = unique_participants[test_start:test_start + test_size]
            else:
                # Wrap around for test set
                test_fold = np.concatenate((
                    unique_participants[test_start:],
                    unique_participants[:test_size - (num_participants - test_start)]
                ))
            
            val_folds.append(val_fold)
            test_folds.append(test_fold)

        print('folds', val_folds, test_folds)

        # Get the current fold's validation and test participants
        val_participants = val_folds[fold_no]
        test_participants = test_folds[fold_no]
        
        # Train participants are all remaining participants
        train_participants = np.setdiff1d(unique_participants, 
                                         np.concatenate((val_participants, test_participants)))

        print(val_participants, test_participants, train_participants)
        
        # Print splits for verification
        print(f"Fold {fold_no} participants distribution:")
        print(f"Train participants: {len(train_participants)} ({len(train_participants)/num_participants:.1%})")
        print(f"Val participants: {len(val_participants)} ({len(val_participants)/num_participants:.1%})")
        print(f"Test participants: {len(test_participants)} ({len(test_participants)/num_participants:.1%})")
        
        # Get boolean masks for each split
        train_mask = df['participant_id'].isin(train_participants)
        val_mask = df['participant_id'].isin(val_participants)
        test_mask = df['participant_id'].isin(test_participants)

        print(train_mask)
        
        # Extract data using boolean masks
        X_train = features.loc[train_mask].reset_index(drop=True)
        y_train = target[train_mask]
        session_train = sessions[train_mask]
        
        X_val = features.loc[val_mask].reset_index(drop=True)
        y_val = target[val_mask]
        session_val = sessions[val_mask]
        
        X_test = features.loc[test_mask].reset_index(drop=True)
        y_test = target[test_mask]
        session_test = sessions[test_mask]
        
        # Print dataset sizes
        print("Train shapes:", X_train.shape, y_train.shape)
        print("Validation shapes:", X_val.shape, y_val.shape)
        print("Test shapes:", X_test.shape, y_test.shape)

        # Create sequences
        X_train_sequences, y_train_sequences = create_sequences(X_train.values, y_train, session_train, sequence_length)
        X_val_sequences, y_val_sequences = create_sequences(X_val.values, y_val, session_val, sequence_length) 
        X_test_sequences, y_test_sequences = create_sequences(X_test.values, y_test, session_test, sequence_length)
        
        print("Train sequences shape:", X_train_sequences.shape, y_train_sequences.shape)
        print("Val sequences shape:", X_val_sequences.shape, y_val_sequences.shape)
        print("Test sequences shape:", X_test_sequences.shape, y_test_sequences.shape)
        
        # Check if any sequences are empty
        if len(X_train_sequences) == 0 or len(X_val_sequences) == 0 or len(X_test_sequences) == 0:
            print(f"Sequences for fold {fold_no} are empty. Skipping this fold.")
            return None
        
        return (
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            X_train_sequences, y_train_sequences,
            X_val_sequences, y_val_sequences,
            X_test_sequences, y_test_sequences,
            sequence_length
        )
    except Exception as e:
        print(f"Error in create_data_splits: {e}")
        return None


def create_data_splits_ids(df, num_folds = 5, fold_no=0, seed_value=42, sequence_length=1):
    
    try:
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)

        # Make a copy of the DataFrame with reset index to avoid duplicate indices
        df = df.copy().reset_index(drop=True)
        print(df.columns)
        
        # Extract features and target
        features = df.iloc[:, 5:]
        target = df.iloc[:, 2].values.astype('int')
        # Extract sessions - assuming this is a column in the dataframe
        # If it's the 'participant_id', use that, otherwise use an appropriate column
        sessions = df['participant_id'].values  # Change this if sessions are stored in a different column
        
        print("Features shape:", features.shape)
        print('features head', features.head())
        
        # Get unique participants and shuffle them
        unique_participants = np.unique(df['participant_id'].values)
        num_participants = len(unique_participants)
        print('unique participants', unique_participants)
        
        if num_participants < num_folds:
            raise ValueError(f"Number of participants ({num_participants}) is less than number of folds ({num_folds})")
        
        # Shuffle participants
        np.random.shuffle(unique_participants)

        # Calculate sizes for each set
        val_size = int(np.ceil(0.2 * num_participants))  # 20% for validation
        test_size = int(np.ceil(0.1 * num_participants))  # 10% for test
        train_size = num_participants - val_size - test_size  # Remaining (~70%) for training
        
        # Make sure we have at least one participant in each set
        if val_size < 1 or test_size < 1 or train_size < 1:
            raise ValueError(f"Not enough participants ({num_participants}) to create valid splits")
        
        # Create rolling folds
        val_folds = []
        test_folds = []
        train_folds = []
        
        for i in range(num_folds):
            # For validation set, take consecutive 20% slice, wrap around if needed
            val_start = (i * val_size) % num_participants
            if val_start + val_size <= num_participants:
                val_fold = unique_participants[val_start:val_start + val_size]
            else:
                # Wrap around for validation set
                val_fold = np.concatenate((
                    unique_participants[val_start:],
                    unique_participants[:val_size - (num_participants - val_start)]
                ))
            
            # For test set, take next consecutive 10% slice, wrap around if needed
            test_start = (val_start + val_size) % num_participants
            if test_start + test_size <= num_participants:
                test_fold = unique_participants[test_start:test_start + test_size]
            else:
                # Wrap around for test set
                test_fold = np.concatenate((
                    unique_participants[test_start:],
                    unique_participants[:test_size - (num_participants - test_start)]
                ))
            
            val_folds.append(val_fold)
            test_folds.append(test_fold)

            #train fold is just remaining participants
            train_fold = np.setdiff1d(unique_participants, 
                                         np.concatenate((val_fold, test_fold)))

            train_folds.append(train_fold)

            #return fold lists
        return val_folds, test_folds, train_folds
    except Exception as e:
        print(f"Error in create_data_splits_ids: {e}")
        return None
        
    
