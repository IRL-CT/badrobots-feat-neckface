#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BADNet Robot Behavior Classification

This module implements a CNN model to classify robot behavior as 'good' or 'bad'
based on human bystander reactions captured via webcam.

Part of an HRI social affective computing project.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, cohen_kappa_score, confusion_matrix
)
import wandb


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#cudnn.benchmark = True
#plt.ion()   # interactive mode

# check if CUDA is available
#train_on_gpu = torch.cuda.is_available()
#class_labels = [0,1]

# Set a random seed for CPU
#seed = 42
#torch.manual_seed(seed)
#np.random.seed(seed)
#random.seed(seed)

# Set a random seed for CUDA (if available)
#if train_on_gpu:
#    torch.cuda.manual_seed(seed)

#print('DEVICE:', device)

#######################################################################
# TODO SECTION: FUTURE ENHANCEMENTS
#######################################################################
# TODO: Implement a ResNet version of the model architecture
#  - Create a new function called `create_resnet_model(input_shape, hyperparams)` 
#  - Add residual connections between layers
#  - Implement identity and convolutional blocks following ResNet paper pattern
#  - Consider deeper architecture with 18, 34, or 50 layers
#  - Add option in main() to choose between standard CNN and ResNet

# TODO: Implement data augmentation pipeline
#  - Add real-time augmentation to DataGenerator and DirectoryDataGenerator
#  - Include standard augmentations: rotation, flipping, brightness/contrast, zoom
#  - Create an AugmentationConfig class to control augmentation parameters
#  - Add options to enable/disable specific augmentations
#  - Consider using tf.image or imgaug library for augmentations
#  - Implement test-time augmentation for evaluation improvement

# TODO: Add hyperparameter tuning capability
#  - Implement grid or random search for optimal hyperparameters
#  - Track experiments with TensorBoard or external experiment tracking
#  - Test different learning rates, kernel sizes, filter counts, and dropout rates
#  - Consider using Keras Tuner or Optuna for automated hyperparameter search
#  - Save and compare results across different parameter combinations
#######################################################################


def load_data(data_path, fold, filenames=None):
    """
    Load training and testing data from numpy files.
    
    Args:
        data_path: Directory containing the data files
        filenames: Dict with keys 'x_train', 'x_test', 'y_train', 'y_test' and values
                  as corresponding filenames
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    if filenames is None:
        filenames = {
            'x_train': f'XTRAIN_{fold}.npy',
            'x_test': f'XTEST_{fold}.npy',
            'y_train': f'YTRAIN_{fold}.npy',
            'y_test': f'YTEST_{fold}.npy',
            'x_val': f'XVAL_{fold}.npy',
            'y_val': f'YVAL_{fold}.npy'
        }
    
    # Load data files
    X_train = np.load(os.path.join(data_path, filenames['x_train']))
    X_val = np.load(os.path.join(data_path, filenames['x_val']))
    y_train = np.load(os.path.join(data_path, filenames['y_train']))
    y_val = np.load(os.path.join(data_path, filenames['y_val']))
    X_test = np.load(os.path.join(data_path, filenames['x_test']))
    y_test = np.load(os.path.join(data_path, filenames['y_test']))
    
    # Check if we need to reshape from (N, 1, 224, 224, 3) to (N, 224, 224, 3)
    if len(X_train.shape) == 5 and X_train.shape[1] == 1:
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[2], X_train.shape[3], X_train.shape[4])
    if len(X_test.shape) == 5 and X_test.shape[1] == 1:
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[2], X_test.shape[3], X_test.shape[4])
    if len(X_val.shape) == 5 and X_val.shape[1] == 1:
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[2], X_val.shape[3], X_val.shape[4])
    
    print(f"Data shapes: X_train {X_train.shape}, X_test {X_test.shape}, "
          f"y_train {y_train.shape}, y_test {y_test.shape}")
    
    # Convert labels to categorical
    y_train_cat = to_categorical(y_train, num_classes=2)
    y_test_cat = to_categorical(y_test, num_classes=2)
    y_val_cat = to_categorical(y_val, num_classes=2)
    
    return X_train,X_val, X_test, y_train_cat, y_val_cat, y_test_cat


#######################################################################
# OPTIONAL MEMORY-EFFICIENT DATA LOADER
#######################################################################
class DataGenerator(tf.keras.utils.Sequence):
    """
    Memory-efficient data generator for large datasets.
    
    This generator loads data in batches to avoid loading everything into memory.
    Useful for datasets with shape (37472, 224, 224, 3) or larger.
    """
    def __init__(self, data_path, filename, label_filename, batch_size=32, 
                 is_training=True, num_classes=2, shuffle=True):
        """
        Initialize the data generator.
        
        Args:
            data_path: Directory containing the data files
            filename: Filename of the data file (.npy)
            label_filename: Filename of the labels file (.npy)
            batch_size: Number of samples per batch
            is_training: Whether this generator is for training data
            num_classes: Number of classes for one-hot encoding
            shuffle: Whether to shuffle the data after each epoch
        """
        self.data_path = data_path
        self.filename = filename
        self.label_filename = label_filename
        self.batch_size = batch_size
        self.is_training = is_training
        self.num_classes = num_classes
        self.shuffle = shuffle
        
        # Get data shape and length without loading entire array
        data_file = np.load(os.path.join(data_path, filename), mmap_mode='r')
        self.data_shape = data_file.shape
        self.num_samples = self.data_shape[0]
        
        # Calculate length of generator
        self.indices = np.arange(self.num_samples)
        self.on_epoch_end()
    
    def __len__(self):
        """Return the number of batches per epoch."""
        return int(np.floor(self.num_samples / self.batch_size))
    
    def __getitem__(self, index):
        """Generate one batch of data."""
        # Generate indices of the batch
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Generate data
        X, y = self._load_batch(batch_indices)
        
        return X, y
    
    def on_epoch_end(self):
        """Shuffle indices after each epoch if shuffle is True."""
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def _load_batch(self, indices):
        """Load a batch of data from disk."""
        # Load data and labels
        data_file = np.load(os.path.join(self.data_path, self.filename), mmap_mode='r')
        label_file = np.load(os.path.join(self.data_path, self.label_filename), mmap_mode='r')
        
        # Extract batch data
        batch_x = np.array([data_file[i] for i in indices])
        batch_y = np.array([label_file[i] for i in indices])
        
        # Reshape if needed (handle (N, 1, 224, 224, 3) shape)
        if len(batch_x.shape) == 5 and batch_x.shape[1] == 1:
            batch_x = batch_x.reshape(batch_x.shape[0], batch_x.shape[2], batch_x.shape[3], batch_x.shape[4])
        
        # Convert labels to categorical
        batch_y_cat = to_categorical(batch_y, num_classes=self.num_classes)
        
        return batch_x, batch_y_cat
    
    @property
    def shape(self):
        """Return the shape of a single sample."""
        # Return either (224, 224, 3) or the actual shape excluding batch dimension
        if len(self.data_shape) == 5:  # (N, 1, 224, 224, 3)
            return (self.data_shape[2], self.data_shape[3], self.data_shape[4])
        else:  # (N, 224, 224, 3)
            return self.data_shape[1:]


class DirectoryDataGenerator(tf.keras.utils.Sequence):
    """
    Memory-efficient data generator for datasets stored as individual files.
    
    This generator loads data from directories where each example is stored as a separate .npy file.
    Useful when data is organized with one file per example.
    """
    def __init__(self, data_dir, label_dir, batch_size=32, 
                 input_shape=(224, 224, 3), num_classes=2, shuffle=True):
        """
        Initialize the directory-based data generator.
        
        Args:
            data_dir: Directory containing the data files (.npy)
            label_dir: Directory containing the label files (.npy)
            batch_size: Number of samples per batch
            input_shape: Shape of input images (height, width, channels)
            num_classes: Number of classes for one-hot encoding
            shuffle: Whether to shuffle the data after each epoch
        """
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.shuffle = shuffle
        
        # Get all filenames in the directory
        self.data_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npy')])
        self.num_samples = len(self.data_files)
        
        # Make sure we have labels for each data file
        # Assume matching filenames between data and label directories
        # (modify this if your naming convention is different)
        assert all(os.path.exists(os.path.join(label_dir, f)) for f in self.data_files), \
            "Missing label files for some data files"
        
        # Calculate length of generator
        self.indices = np.arange(self.num_samples)
        self.on_epoch_end()
    
    def __len__(self):
        """Return the number of batches per epoch."""
        return int(np.floor(self.num_samples / self.batch_size))
    
    def __getitem__(self, index):
        """Generate one batch of data."""
        # Generate indices of the batch
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Generate data
        X, y = self._load_batch(batch_indices)
        
        return X, y
    
    def on_epoch_end(self):
        """Shuffle indices after each epoch if shuffle is True."""
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def _load_batch(self, indices):
        """Load a batch of data from disk."""
        # Initialize batch arrays
        batch_x = np.zeros((len(indices), *self.input_shape), dtype=np.float32)
        batch_y = np.zeros(len(indices), dtype=np.int32)
        
        # Load files for this batch
        for i, idx in enumerate(indices):
            # Load data file
            file_name = self.data_files[idx]
            data = np.load(os.path.join(self.data_dir, file_name))
            
            # Handle different possible shapes
            if len(data.shape) == 4 and data.shape[0] == 1:  # (1, 224, 224, 3)
                data = data.reshape(data.shape[1:])
            
            # Ensure correct shape
            if data.shape != self.input_shape:
                data = data.reshape(self.input_shape)
            
            # Load label file (assumes same filename in label directory)
            label = np.load(os.path.join(self.label_dir, file_name))
            if isinstance(label, np.ndarray) and label.size > 1:
                label = label[0]  # Take first element if array
            
            # Store in batch arrays
            batch_x[i] = data
            batch_y[i] = label
        
        # Convert labels to categorical
        batch_y_cat = to_categorical(batch_y, num_classes=self.num_classes)
        
        return batch_x, batch_y_cat
    
    @property
    def shape(self):
        """Return the shape of a single sample."""
        return self.input_shape
#######################################################################


def create_model(input_shape, hyperparams):
    """
    Create a CNN model for robot behavior classification.
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        hyperparams: Dictionary of hyperparameters
    
    Returns:
        Compiled Keras model
    """
    # Extract hyperparameters with defaults
    activation = hyperparams.get('activation', 'relu')
    kernel_init = hyperparams.get('kernel_init', 
                                  keras.initializers.he_uniform(seed=1369))
    kernel_size = hyperparams.get('kernel_size', 4)
    base_filters = hyperparams.get('base_filters', 16)
    learning_rate = hyperparams.get('learning_rate', 0.0001)
    
    # Create sequential model
    model = keras.Sequential([
        # First Conv Block
        layers.Conv2D(
            filters=base_filters,
            kernel_size=(kernel_size, kernel_size),
            strides=(2, 2),
            activation=activation,
            kernel_initializer=kernel_init,
            padding="same",
            input_shape=input_shape,
            name="layer1"
        ),
        layers.Dropout(0.15),
        
        # Second Conv Block
        layers.Conv2D(
            filters=base_filters*2,
            kernel_size=(kernel_size, kernel_size),
            strides=(2, 2),
            activation=activation,
            kernel_initializer=kernel_init,
            padding="same",
            name="layer2"
        ),
        layers.Dropout(0.2),
        
        # Third Conv Block
        layers.Conv2D(
            filters=base_filters*4,
            kernel_size=(kernel_size, kernel_size),
            strides=(2, 2),
            activation=activation,
            kernel_initializer=kernel_init,
            padding="same",
            name="layer3"
        ),
        layers.Dropout(0.2),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(data_format='channels_last'),
        
        # Fully Connected Layers
        layers.Flatten(),
        layers.Dropout(0.6),
        layers.Dense(128, activation=activation),
        layers.Dense(2, activation="softmax")
    ])
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['categorical_accuracy']
    )
    
    return model


def train_model(model, X_train, y_train, X_test, y_test, batch_size=1, epochs=350, 
               checkpoint_dir='./checkpoints', early_stopping=True):
    """
    Train the model with the provided data.
    
    Args:
        model: Keras model to train
        X_train: Training images
        y_train: Training labels (categorical)
        X_test: Testing images
        y_test: Testing labels (categorical)
        batch_size: Batch size for training
        epochs: Number of training epochs
        checkpoint_dir: Directory to save model checkpoints
        early_stopping: Whether to use early stopping
    
    Returns:
        Training history object
    """
    # Create callbacks list
    callbacks = []
    
    # Create checkpoint directory if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Add model checkpoint to save the best model
    checkpoint_path = os.path.join(checkpoint_dir, 'model_{epoch:03d}_{val_categorical_accuracy:.4f}.h5')
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_categorical_accuracy',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='max',
        save_freq='epoch'
    )
    callbacks.append(checkpoint_callback)
    
    # Add best model checkpoint (overwrite the same file for the best model)
    best_model_path = os.path.join(checkpoint_dir, 'best_model.h5')
    best_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=best_model_path,
        monitor='val_categorical_accuracy',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='max',
        save_freq='epoch'
    )
    callbacks.append(best_checkpoint_callback)
    
    # Add early stopping if enabled
    if early_stopping:
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0.001,
            patience=20,
            verbose=1,
            mode='min',
            restore_best_weights=True
        )
        callbacks.append(early_stopping_callback)
    
    # Add a callback to log training progress
    log_callback = tf.keras.callbacks.CSVLogger(
        os.path.join(checkpoint_dir, 'training_log.csv'),
        separator=',', 
        append=False
    )
    callbacks.append(log_callback)
    
    # Train model with callbacks
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        shuffle=True,
        verbose=1,
        callbacks=callbacks
    )
    
    return history


def train_model_with_generator(model, train_generator, val_generator, epochs=350, 
                              checkpoint_dir='./checkpoints', early_stopping=True):
    """
    Train the model with data generators (memory-efficient).
    
    Args:
        model: Keras model to train
        train_generator: Training data generator
        val_generator: Validation data generator
        epochs: Number of training epochs
        checkpoint_dir: Directory to save model checkpoints
        early_stopping: Whether to use early stopping
    
    Returns:
        Training history object
    """
    # Create callbacks list
    callbacks = []
    
    # Create checkpoint directory if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Add model checkpoint to save the best model
    checkpoint_path = os.path.join(checkpoint_dir, 'model_{epoch:03d}_{val_categorical_accuracy:.4f}.h5')
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_categorical_accuracy',  # Metric to monitor
        verbose=1,
        save_best_only=True,  # Only save the best model
        save_weights_only=False,  # Save entire model
        mode='max',  # For accuracy, we want to maximize
        save_freq='epoch'  # Save once per epoch
    )
    callbacks.append(checkpoint_callback)
    
    # Add best model checkpoint (overwrite the same file for the best model)
    best_model_path = os.path.join(checkpoint_dir, 'best_model.h5')
    best_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=best_model_path,
        monitor='val_categorical_accuracy',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='max',
        save_freq='epoch'
    )
    callbacks.append(best_checkpoint_callback)
    
    # Add early stopping if enabled
    if early_stopping:
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',          # Monitor validation loss
            min_delta=0.001,             # Minimum change to count as improvement
            patience=20,                 # Number of epochs with no improvement to stop training
            verbose=1,                   # Report when early stopping is triggered
            mode='min',                  # For loss, we want to minimize
            restore_best_weights=True    # Restore model weights from the epoch with the best value
        )
        callbacks.append(early_stopping_callback)
    
    # Add a callback to log training progress
    log_callback = tf.keras.callbacks.CSVLogger(
        os.path.join(checkpoint_dir, 'training_log.csv'),
        separator=',', 
        append=False
    )
    callbacks.append(log_callback)
    
    # Train model with callbacks
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        verbose=1,
        use_multiprocessing=True,
        workers=4,  # Adjust based on your CPU cores
        callbacks=callbacks
    )
    
    return history


def plot_training_history(history):
    """
    Plot training and validation accuracy.
    
    Args:
        history: History object returned by model.fit()
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['categorical_accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_categorical_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and print performance metrics.
    
    Args:
        model: Trained keras model
        X_test: Test images
        y_test: Test labels (categorical)
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Get predictions
    y_pred_probs = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_probs, axis=-1)
    
    # Convert one-hot encoded test labels to class indices
    y_test_classes = np.argmax(y_test, axis=-1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_classes, y_pred_classes)
    precision = precision_score(y_test_classes, y_pred_classes, average='weighted')
    recall = recall_score(y_test_classes, y_pred_classes, average='weighted')
    f1 = f1_score(y_test_classes, y_pred_classes, average='weighted')
    kappa = cohen_kappa_score(y_test_classes, y_pred_classes)
    conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
    
    # Print results
    print("\nModel Evaluation Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    # Return metrics for further use if needed
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'kappa': kappa,
        'confusion_matrix': conf_matrix
    }


def evaluate_model_with_generator(model, test_generator):
    """
    Evaluate the model using a data generator and print performance metrics.
    
    Args:
        model: Trained keras model
        test_generator: Generator for test data
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Get all test data and labels from generator
    all_pred_probs = []
    all_true_labels = []
    
    for i in range(len(test_generator)):
        X_batch, y_batch = test_generator[i]
        pred_probs = model.predict(X_batch)
        all_pred_probs.append(pred_probs)
        all_true_labels.append(y_batch)
    
    # Concatenate batches
    all_pred_probs = np.vstack(all_pred_probs)
    all_true_labels = np.vstack(all_true_labels)
    
    # Convert to class indices
    y_pred_classes = np.argmax(all_pred_probs, axis=-1)
    y_test_classes = np.argmax(all_true_labels, axis=-1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_classes, y_pred_classes)
    precision = precision_score(y_test_classes, y_pred_classes, average='weighted')
    recall = recall_score(y_test_classes, y_pred_classes, average='weighted')
    f1 = f1_score(y_test_classes, y_pred_classes, average='weighted')
    kappa = cohen_kappa_score(y_test_classes, y_pred_classes)
    conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
    
    # Print results
    print("\nModel Evaluation Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    # Return metrics for further use if needed
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'kappa': kappa,
        'confusion_matrix': conf_matrix
    }


def main():
    """Main function to run the training and evaluation pipeline."""
    # Define hyperparameters that worked best during training
    hyperparams = {
        'activation': 'relu',
        'kernel_init': keras.initializers.he_uniform(seed=1369),
        'kernel_size': 4,
        'base_filters': 16,
        'learning_rate': 0.0001,
        'folds': 5,
    }

    wandb.log({'activation': hyperparams['activation']})
    wandb.log({'kernel_init': hyperparams['kernel_init']})
    wandb.log({'kernel_size': hyperparams['kernel_size']})
    wandb.log({'base_filters': hyperparams['base_filters']})
    wandb.log({'learning_rate': hyperparams['learning_rate']})
    
    # Set the data path
    data_path = './'  # Change to your data directory
    
    # Choose data loading method
    data_loading_method = "standard"  # Options: "standard", "generator", "directory"
    
    for fold in range(hyperparams['folds']):
        wandb.log({'fold': fold})
        #clear model
        tf.keras.backend.clear_session()
        #if variable model exists, delete
        if 'model' in locals():
            del model

    
        if data_loading_method == "standard":
            # Standard approach (load all data at once)
            print("Loading all data into memory")
            X_train, X_val, X_test, y_train, y_val, y_test = load_data(data_path,fold)
            
            # Define input shape for the model
            input_shape = (X_train.shape[1], X_train.shape[2], 3)
            
            # Create model
            model = create_model(input_shape, hyperparams)
            model.summary()
            
            # Start with a small batch size (adjust based on your system capabilities)
            batch_size = 1  # Start small, increase as needed (8, 16, 32, etc.)
            
            # Train model
            history = train_model(
                model, 
                X_train, y_train, 
                X_val, y_val, 
                batch_size=batch_size, 
                epochs=350,
                checkpoint_dir='./checkpoints',
                early_stopping=True
            )
            #save history.history['categorical_accuracy'], history.history['val_categorical_accuracy'] and losses to wandb
            wandb.log({
                'train_accuracy': history.history['categorical_accuracy'],
                'val_accuracy': history.history['val_categorical_accuracy'],
                'losses': history.history['loss']
            })
   
            
            # Plot training history
            plot_training_history(history)
            
            # Evaluate model
            metrics = evaluate_model(model, X_test, y_test)
            wandb.log({f"fold_{fold}_metrics": metrics})
            print(f"Fold {fold} Test Metrics:", metrics)
        
        elif data_loading_method == "generator":
            # Memory-efficient approach for large datasets in single files
            print("Using memory-efficient data generator for large .npy files")
            batch_size = 32
            train_generator = DataGenerator(
                data_path=data_path,
                filename=f'XTRAIN_{fold}.npy',
                label_filename= f'YTRAIN_{fold}.npy',
                batch_size=batch_size,
                is_training=True,
                shuffle=True
            )
            
            val_generator = DataGenerator(
                data_path=data_path,
                filename=f'XVAL_{fold}.npy',
                label_filename= f'YVAL_{fold}.npy',
                batch_size=batch_size,
                is_training=False,
                shuffle=False
            )

            test_generator = DataGenerator(
                data_path=data_path,
                filename=f'XTEST_{fold}.npy',
                label_filename= f'YTEST_{fold}.npy',
                batch_size=batch_size,
                is_training=False,
                shuffle=False
            )
            
            # Print information about the data
            print(f"Training samples: {train_generator.num_samples}")
            print(f"Validation samples: {val_generator.num_samples}")
            print(f"Data shape: {train_generator.shape}")
            
            # Create model
            input_shape = train_generator.shape
            model = create_model(input_shape, hyperparams)
            model.summary()
            
            # Train model
            history = train_model_with_generator(
                model, 
                train_generator, 
                val_generator, 
                epochs=350,
                checkpoint_dir='./checkpoints',
                early_stopping=True
            )
            #save history.history['categorical_accuracy'], history.history['val_categorical_accuracy'] and losses to wandb
            wandb.log({
                'train_accuracy': history.history['categorical_accuracy'],
                'val_accuracy': history.history['val_categorical_accuracy'],
                'losses': history.history['loss']
            })
            # Plot training history
            plot_training_history(history)
            
            # Evaluate model
            metrics = evaluate_model_with_generator(model, test_generator)
            wandb.log({f"fold_{fold}_metrics": metrics})
            print(f"Fold {fold} Test Metrics:", metrics)
        
        elif data_loading_method == "directory":
            # For datasets stored as individual files
            print("Using directory-based data generator for individual .npy files")
            batch_size = 32
            train_generator = DirectoryDataGenerator(
                data_dir=os.path.join(data_path, 'train_data'),
                label_dir=os.path.join(data_path, 'train_labels'),
                batch_size=batch_size,
                input_shape=(224, 224, 3),
                shuffle=True
            )
            
            val_generator = DirectoryDataGenerator(
                data_dir=os.path.join(data_path, 'val_data'),
                label_dir=os.path.join(data_path, 'val_labels'),
                batch_size=batch_size,
                input_shape=(224, 224, 3),
                shuffle=False
            )

            test_generator = DirectoryDataGenerator(
                data_dir=os.path.join(data_path, 'test_data'),
                label_dir=os.path.join(data_path, 'test_labels'),
                batch_size=batch_size,
                input_shape=(224, 224, 3),
                shuffle=False
            )
            
            # Print information about the data
            print(f"Training samples: {train_generator.num_samples}")
            print(f"Validation samples: {val_generator.num_samples}")
            
            # Create model
            input_shape = train_generator.shape
            model = create_model(input_shape, hyperparams)
            model.summary()
            
            # Train model
            history = train_model_with_generator(model, train_generator, val_generator, epochs=350)
            #save history.history['categorical_accuracy'], history.history['val_categorical_accuracy'] and losses to wandb
            wandb.log({
                'train_accuracy': history.history['categorical_accuracy'],
                'val_accuracy': history.history['val_categorical_accuracy'],
                'losses': history.history['loss']
            })
            
            # Plot training history
            plot_training_history(history)
            
            # Evaluate model
            metrics = evaluate_model_with_generator(model, test_generator)
            wandb.log({f"fold_{fold}_metrics": metrics})
            print(f"Fold {fold} Test Metrics:", metrics)
        
        # Optional: Save the model
        # model.save('bad_robot_classifier.h5')
        
        return model, history, metrics


if __name__ == "__main__":
    # Set TensorFlow to log only errors
    tf.get_logger().setLevel('ERROR')
    
    # Set memory growth for GPU to avoid OOM errors
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
            print(f"Memory growth enabled for GPU device: {device}")

    #run wandb to save variables
    wandb.init(project="BADNet", entity="your_entity")
    
    # Run the main function
    model, history, metrics = main()