
import numpy as np
import pandas as pd
import pickle
import os
import random
#import torch
#no warnings
import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.utils import to_categorical



#def create_sequences(data, target, sequence_length):
#    sequences = [data[i : i + sequence_length] for i in range(len(data) - sequence_length + 1)]
#    targets = target[sequence_length - 1 : ]
#    return np.array(sequences), np.array(targets)

def create_sequences(data, target, sessions, sequence_length, hot_encode = 1):
    sequences = []
    targets = []

    #hot one encode the target
    
    # Split data by session and then create sequences
    unique_sessions = np.unique(sessions)
    for session in unique_sessions:
        session_indices = np.where(sessions == session)[0]
        session_data = data[session_indices]
        session_target = target[session_indices]
        
        if len(session_data) >= sequence_length:
            for i in range(len(session_data) - sequence_length + 1):
                sequences.append(session_data[i : i + sequence_length])
                targets.append(session_target[i + sequence_length - 1])

    #hot one encode the target
    if hot_encode == 1:
        targets = pd.get_dummies(targets).values
    else:
        targets = to_categorical(targets, num_classes = 2)

        
    return np.array(sequences), np.array(targets)

def create_sequences_nooh(data, target, sessions, sequence_length):
    sequences = []
    targets = []

    #hot one encode the target
    
    # Split data by session and then create sequences
    unique_sessions = np.unique(sessions)
    for session in unique_sessions:
        session_indices = np.where(sessions == session)[0]
        session_data = data[session_indices]
        session_target = target[session_indices]
        
        if len(session_data) >= sequence_length:
            for i in range(len(session_data) - sequence_length + 1):
                sequences.append(session_data[i : i + sequence_length])
                targets.append(session_target[i + sequence_length - 1])

        
    #hot one encode the target
    #targets = pd.get_dummies(targets).values
    
    return np.array(sequences), np.array(targets)

def createDataSplits(df, fold_no, with_val = 1, hot_encode = 1, label_column = 'class_label',num_folds = 5, results_directory= '../logs/', seed_value = 42, sequence_length = 1):

    """
    Split the data into train, validation, and test sets for each fold.
    df - dataframe containing the data
    fold_no - number of fold to get the data for
    with_val - whether to include validation set or not
    label_column - column name for the target class
    fold_column - column name for the fold number
    results_directory - directory to store the results
    seed_value - seed value for reproducibility
    sequence_length - length of the sequence for RNNs or transformers
    
    """

    try:

        # # Set seed
        random.seed(seed_value)
        np.random.seed(seed_value)
        #tf.random.set_seed(seed_value)
        #torch.manual_seed(seed_value)
        #torch.cuda.manual_seed_all(seed_value)
    

        # get features and target class
        features = df.iloc[:,5:]
        target_class = df[label_column].values
        target_class = target_class.astype('int')
        sessions = df['participant_id'].values


        #get number of classes
        num_classes = len(np.unique(target_class))
        #if fold number is none
        if num_folds is None:
            train_indices = df.index
            X_train = features.loc[train_indices]
            y_train = target_class[train_indices]
            session_train = sessions[train_indices]
            X_train = X_train.reset_index(drop=True)
            X_train_sequences, y_train_sequences = create_sequences(X_train.values, y_train, session_train, sequence_length, hot_encode=hot_encode)
            X_val = None
            y_val = None
            X_val_sequences = None
            y_val_sequences = None
            X_test = None
            y_test = None
            X_test_sequences = None
            y_test_sequences = None


        else:
            
            fold_sessions = df['participant_id'].unique()

            #get number of participants per fold


            if with_val == 1:
                num_test = int(np.ceil(0.2*len(fold_sessions)))
                num_val = int(np.ceil(0.1*len(fold_sessions)))
                num_train = int(len(fold_sessions) - num_test - num_val)

       
                np.random.shuffle(fold_sessions)


                # Initialize lists to store train, validation, and test participants for each fold
                val_folds = []
                test_folds = []

                # Create non-overlapping test folds and validation folds
                for i in range(num_folds):
                    start_test_idx = i * num_test
                    end_test_idx = start_test_idx + np.min([num_test, len(fold_sessions) - start_test_idx])

                    test_fold = fold_sessions[start_test_idx:end_test_idx]
                    #print('test_fold:', test_fold)
                        
                    # Identify all the participants except the participants belonging to the test_fold & shuffle them
                    remaining_participants = np.setdiff1d(fold_sessions, test_fold)
                    np.random.shuffle(remaining_participants)
                    
                    # Validation set selected from the remaining participants
                    val_fold = remaining_participants[:num_val]
                    #print('val_fold:', val_fold)
                    
                    # Identify all the participants that don't belong to 'val_fold' & 'test_fold' and assign them to the 'train_fold'
                    train_fold = np.setdiff1d(remaining_participants, val_fold)
                    #print('train_fold:', train_fold)
                    
                    # Append the participant sets to their corresponding folds
                    train_folds.append(train_fold)
                    val_folds.append(val_fold)
                    test_folds.append(test_fold)
                
                """
                End: K-Fold Cross-Validation splits
                """

                # # Create train, validation, and test sets for each fold in 'num_folds'
                # For now, do only for fold: '0'
                train_fold = train_folds[fold_no]
                val_fold = val_folds[fold_no]
                test_fold = test_folds[fold_no]
                print('folds:', train_fold,val_fold,test_fold)


              
            else:
                num_test = np.floor(0.2*len(fold_sessions))
                num_train = len(fold_sessions) - num_test

                np.random.shuffle(fold_sessions)


                # Initialize lists to store train, validation, and test participants for each fold
                train_folds = []
                test_folds = []

                # Create non-overlapping test folds and validation folds
                for i in range(num_folds):
                    start_test_idx = i * num_test
                    end_test_idx = start_test_idx + np.min([num_test, len(fold_sessions) - start_test_idx])

                    test_fold = fold_sessions[start_test_idx:end_test_idx]
                        
                    # Identify all the participants except the participants belonging to the test_fold & shuffle them
                    remaining_participants = np.setdiff1d(fold_sessions, test_fold)
                    np.random.shuffle(remaining_participants)
                    
                    
                    # Identify all the participants that don't belong to 'val_fold' & 'test_fold' and assign them to the 'train_fold'
                    train_fold = remaining_participants
                    
                    # Append the participant sets to their corresponding folds
                    train_folds.append(train_fold)
                    test_folds.append(test_fold)
                
                """
                End: K-Fold Cross-Validation splits
                """

                # # Create train, validation, and test sets for each fold in 'num_folds'
                # For now, do only for fold: '0'
                train_fold = train_folds[fold_no]
                test_fold = test_folds[fold_no]

                print('folds:', train_fold,test_fold)


            # Split the data into train, validation, and test sets
            train_indices = df[df['participant_id'].isin(train_fold)].index
            #print(train_indices)
            if with_val == 1:
                val_indices = df[df['participant_id'].isin(val_fold)].index
            test_indices = df[df['participant_id'].isin(test_fold)].index

            X_train = features.loc[train_indices]
            y_train = target_class[train_indices]
            session_train = sessions[train_indices]
            if with_val == 1:
                X_val = features.loc[val_indices]
                y_val = target_class[val_indices]
                session_val = sessions[val_indices]

            X_test = features.loc[test_indices]
            y_test = target_class[test_indices]
            session_test = sessions[test_indices]

            #one-hot encode the target class
            #y_train = pd.get_dummies(y_train).values
            #if with_val == 1:
            #    y_val = pd.get_dummies(y_val).values
            #y_test = pd.get_dummies(y_test).values


            #print size of all sets
            print(X_train.shape, y_train.shape)
            if with_val == 1:
                print(X_val.shape, y_val.shape)
            print(X_test.shape, y_test.shape)

            #reset indexes
            X_train = X_train.reset_index(drop=True)
            if with_val == 1:
                X_val = X_val.reset_index(drop=True)
            X_test = X_test.reset_index(drop=True)

            print('got here')

            # Create sequences for LSTM
            X_train_sequences, y_train_sequences = create_sequences(X_train.values, y_train, session_train, sequence_length, hot_encode=hot_encode)
            if with_val == 1:
                X_val_sequences, y_val_sequences = create_sequences(X_val.values, y_val, session_val, sequence_length, hot_encode=hot_encode)
            else:
                X_val_sequences = None
                y_val_sequences = None
                X_val = None
                y_val = None
                
            X_test_sequences, y_test_sequences = create_sequences(X_test.values, y_test, session_test, sequence_length,hot_encode=hot_encode)

            #


        return num_classes, sessions, X_train, X_val, X_test, y_train, y_val, y_test, X_train_sequences, y_train_sequences, X_val_sequences, y_val_sequences, X_test_sequences, y_test_sequences
    
    except Exception as e:
        print(f"An error occurred: {e}")




def createDataSplitsTransformer(df, fold_no, with_val = 1, label_column = 'class_label',num_folds = 5, results_directory= '../logs/', seed_value = 42, sequence_length = 1):

    """
    Split the data into train, validation, and test sets for each fold.
    df - dataframe containing the data
    fold_no - number of fold to get the data for
    with_val - whether to include validation set or not
    label_column - column name for the target class
    fold_column - column name for the fold number
    results_directory - directory to store the results
    seed_value - seed value for reproducibility
    sequence_length - length of the sequence for RNNs or transformers
    
    """

    try:

        # # Set seed
        random.seed(seed_value)
        np.random.seed(seed_value)
        #tf.random.set_seed(seed_value)
        #torch.manual_seed(seed_value)
        #torch.cuda.manual_seed_all(seed_value)
    

        # get features and target class
        features = df.iloc[:,5:]
        target_class = df[label_column].values
        target_class = target_class.astype('int')
        sessions = df['participant_id'].values


        #get number of classes
        num_classes = len(np.unique(target_class))
        print('NUM CLASSES:', num_classes)
        #if fold number is none
        if num_folds is None:
            train_indices = df.index
            X_train = features.loc[train_indices]
            y_train = target_class[train_indices]
            session_train = sessions[train_indices]
            X_train = X_train.reset_index(drop=True)
            X_train_sequences, y_train_sequences = create_sequences_nooh(X_train.values, y_train, session_train, sequence_length)
            X_val = None
            y_val = None
            X_val_sequences = None
            y_val_sequences = None
            X_test = None
            y_test = None
            X_test_sequences = None
            y_test_sequences = None


        else:
            
            fold_sessions = df['participant_id'].unique()
            print('FOLD SESSIONS:', fold_sessions)
            num_fold_sessions = len(fold_sessions)

            #get number of participants per fold


            if with_val == 1:
                num_val = int(np.floor(0.2*len(fold_sessions)))
                num_test = int(np.ceil(0.1*len(fold_sessions)))
                num_train = int(len(fold_sessions) - num_test - num_val)

       
                np.random.shuffle(fold_sessions)


                # Initialize lists to store train, validation, and test participants for each fold
                train_folds = []
                val_folds = []
                test_folds = []

                # Create non-overlapping test folds and validation folds
                for i in range(num_folds):
                    # For validation set, take consecutive 20% slice, wrap around if needed
                    val_start = (i * num_val) % num_fold_sessions
                    if val_start + num_val <= num_fold_sessions:
                        val_fold = fold_sessions[val_start:val_start + num_val]
                    else:
                        # Wrap around for validation set
                        val_fold = np.concatenate((
                            fold_sessions[val_start:],
                            fold_sessions[:num_val - (num_fold_sessions - val_start)]
                        ))
                    
                    # For test set, take next consecutive 10% slice, wrap around if needed
                    test_start = (val_start + num_val) % num_fold_sessions
                    if test_start + num_test <= num_fold_sessions:
                        test_fold = fold_sessions[test_start:test_start + num_test]
                    else:
                        # Wrap around for test set
                        test_fold = np.concatenate((
                            fold_sessions[test_start:],
                            fold_sessions[:num_test - (num_fold_sessions - test_start)]
                        ))
                    
                    val_folds.append(val_fold)
                    test_folds.append(test_fold)

                print('folds', val_folds, test_folds)
                
                
                """
                End: K-Fold Cross-Validation splits
                """

                # # Create train, validation, and test sets for each fold in 'num_folds'
                # For now, do only for fold: '0'
                #train_fold = train_folds[fold_no]
                val_fold = val_folds[fold_no]
                test_fold = test_folds[fold_no]
                #print('folds:', train_fold,val_fold,test_fold)
                 # Train participants are all remaining participants
                train_fold = np.setdiff1d(len(fold_sessions), 
                                                np.concatenate((val_fold, test_fold)))



              
            else:
                num_test = np.floor(0.2*len(fold_sessions))
                num_train = len(fold_sessions) - num_test

                np.random.shuffle(fold_sessions)


                # Initialize lists to store train, validation, and test participants for each fold
                train_folds = []
                test_folds = []

                # Create non-overlapping test folds and validation folds
                for i in range(num_folds):
                    start_test_idx = i * num_test
                    end_test_idx = start_test_idx + np.min([num_test, len(fold_sessions) - start_test_idx])

                    test_fold = fold_sessions[start_test_idx:end_test_idx]
                        
                    # Identify all the participants except the participants belonging to the test_fold & shuffle them
                    remaining_participants = np.setdiff1d(fold_sessions, test_fold)
                    np.random.shuffle(remaining_participants)
                    
                    
                    # Identify all the participants that don't belong to 'val_fold' & 'test_fold' and assign them to the 'train_fold'
                    train_fold = remaining_participants
                    
                    # Append the participant sets to their corresponding folds
                    train_folds.append(train_fold)
                    test_folds.append(test_fold)
                
                """
                End: K-Fold Cross-Validation splits
                """

                # # Create train, validation, and test sets for each fold in 'num_folds'
                # For now, do only for fold: '0'
                train_fold = train_folds[fold_no]
                test_fold = test_folds[fold_no]

                print('folds:', train_fold,test_fold)


            # Split the data into train, validation, and test sets
            train_indices = df[df['participant_id'].isin(train_fold)].index
            #print(train_indices)
            if with_val == 1:
                val_indices = df[df['participant_id'].isin(val_fold)].index
            test_indices = df[df['participant_id'].isin(test_fold)].index

            X_train = features.loc[train_indices]
            y_train = target_class[train_indices]
            session_train = sessions[train_indices]
            if with_val == 1:
                X_val = features.loc[val_indices]
                y_val = target_class[val_indices]
                session_val = sessions[val_indices]

            X_test = features.loc[test_indices]
            y_test = target_class[test_indices]
            session_test = sessions[test_indices]

            #one-hot encode the target class
            #y_train = pd.get_dummies(y_train).values
            #if with_val == 1:
            #    y_val = pd.get_dummies(y_val).values
            #y_test = pd.get_dummies(y_test).values


            #print size of all sets
            print(X_train.shape, y_train.shape)
            if with_val == 1:
                print(X_val.shape, y_val.shape)
            print(X_test.shape, y_test.shape)

            #reset indexes
            X_train = X_train.reset_index(drop=True)
            if with_val == 1:
                X_val = X_val.reset_index(drop=True)
            X_test = X_test.reset_index(drop=True)

            print('got here')

            
            # Create sequences for LSTM
            X_train_sequences, y_train_sequences = create_sequences_nooh(X_train.values, y_train, session_train, sequence_length)
            if with_val == 1:
                X_val_sequences, y_val_sequences = create_sequences_nooh(X_val.values, y_val, session_val, sequence_length)
            else:
                X_val_sequences = None
                y_val_sequences = None
                X_val = None
                y_val = None
                
            X_test_sequences, y_test_sequences = create_sequences_nooh(X_test.values, y_test, session_test, sequence_length)

            #make categorical
            y_train= to_categorical(y_train, num_classes=2)
            y_val= to_categorical(y_train, num_classes=2)
            y_test= to_categorical(y_train, num_classes=2)

            #
            print('X_train_sequences:', X_train_sequences.shape, y_train_sequences.shape)
            print('X_val_sequences:', X_val_sequences.shape, y_val_sequences.shape)
            print('X_test_sequences:', X_test_sequences.shape, y_test_sequences.shape)
            print('num_classes:', num_classes, 'sessions:', sessions)

        return num_classes, sessions, X_train, X_val, X_test, y_train, y_val, y_test, X_train_sequences, y_train_sequences, X_val_sequences, y_val_sequences, X_test_sequences, y_test_sequences
    
    except Exception as e:
        print(f"An error occurred: {e}")

        



        

def createDataSplitsCNN(participants, with_val = 1, num_folds = 5, results_directory= '../training_outputs/', seed_value = 42):

    """
    Split the data into train, validation, and test sets for each fold.
    df - dataframe containing the data
    fold_no - number of fold to get the data for
    with_val - whether to include validation set or not
    label_column - column name for the target class
    fold_column - column name for the fold number
    results_directory - directory to store the results
    seed_value - seed value for reproducibility
    sequence_length - length of the sequence for RNNs or transformers
    
    """

    try:
        ## Set seed
        random.seed(seed_value)
        np.random.seed(seed_value)

        number_of_participants = len(participants)

        ## Shuffle the list of participants

        fold_sessions = participants

        if with_val == 1:
                num_test = int(np.floor(0.2*len(fold_sessions)))
                num_val = int(np.ceil(0.1*len(fold_sessions)))
                num_train = int(len(fold_sessions) - num_test - num_val)

       
                np.random.shuffle(fold_sessions)


                # Initialize lists to store train, validation, and test participants for each fold
                train_folds = []
                val_folds = []
                test_folds = []

                # Create non-overlapping test folds and validation folds
                for i in range(num_folds):
                    start_test_idx = i * num_test
                    end_test_idx = start_test_idx + np.min([num_test, len(fold_sessions) - start_test_idx])

                    test_fold = fold_sessions[start_test_idx:end_test_idx]
                    #print('test_fold:', test_fold)
                        
                    # Identify all the participants except the participants belonging to the test_fold & shuffle them
                    remaining_participants = np.setdiff1d(fold_sessions, test_fold)
                    np.random.shuffle(remaining_participants)
                    
                    # Validation set selected from the remaining participants
                    val_fold = remaining_participants[:num_val]
                    #print('val_fold:', val_fold)
                    
                    # Identify all the participants that don't belong to 'val_fold' & 'test_fold' and assign them to the 'train_fold'
                    train_fold = np.setdiff1d(remaining_participants, val_fold)
                    #print('train_fold:', train_fold)
                    
                    # Append the participant sets to their corresponding folds
                    train_folds.append(train_fold)
                    val_folds.append(val_fold)
                    test_folds.append(test_fold)


        else:
                num_test = np.floor(0.2*len(fold_sessions))
                num_train = len(fold_sessions) - num_test

                np.random.shuffle(fold_sessions)


                # Initialize lists to store train, validation, and test participants for each fold
                train_folds = []
                test_folds = []
                val_folds = []

                # Create non-overlapping test folds and validation folds
                for i in range(num_folds):
                    start_test_idx = i * num_test
                    end_test_idx = start_test_idx + np.min([num_test, len(fold_sessions) - start_test_idx])

                    test_fold = fold_sessions[start_test_idx:end_test_idx]
                        
                    # Identify all the participants except the participants belonging to the test_fold & shuffle them
                    remaining_participants = np.setdiff1d(fold_sessions, test_fold)
                    np.random.shuffle(remaining_participants)
                    
                    
                    # Identify all the participants that don't belong to 'val_fold' & 'test_fold' and assign them to the 'train_fold'
                    train_fold = remaining_participants
                    
                    # Append the participant sets to their corresponding folds
                    train_folds.append(train_fold)
                    test_folds.append(test_fold)


        return train_folds, val_folds, test_folds
    except Exception as e:
        with open(f"{results_directory}/results_output_log.txt", "a") as results_log_file:
            results_log_file.write(
                f"Exception {e} thrown during splitting dataset for:- \n"
            )
    