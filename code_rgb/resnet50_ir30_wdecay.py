import os
import time
import datetime
import traceback
import torch
import torchvision
import numpy as np
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn

from PIL import Image
from torch.optim import lr_scheduler
from tempfile import TemporaryDirectory
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split

import os
os.environ["WANDB__SERVICE_WAIT"]="300"



# >>> ==================== Non Mixed Participant Changes ===================
## Library dependecies
import random
import traceback
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
# <<< ==================== Non Mixed Participant Changes ===================
#os.environ["WANDB__SERVICE_WAIT"] = "300"
#CHANGE - added wandb
import wandb
wandb.login()
#MODEL SAVE
import glob
import sys
sys.path.insert(1, '../')
#sys.path.insert(1, './')
from get_metrics import get_metrics
from createDataSplits import createDataSplitsCNN
os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


from torch.cuda.amp import autocast, GradScaler



files_to_ignore = [".DS_Store"]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
#plt.ion()   # interactive mode

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()
#class_labels = [0,1]

# Set a random seed for CPU
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Set a random seed for CUDA (if available)
if train_on_gpu:
    torch.cuda.manual_seed(seed)

print('DEVICE:', device)

class_labels = ["0", "1"]


#define the model, resnet50
class ResNet50(nn.Module):
    def __init__(self, num_classes, activation='relu'):
        super(ResNet50, self).__init__()


        self.resnet50 = models.resnet50(weights="IMAGENET1K_V1")
        in_features = self.resnet50.fc.in_features
        #add fully connected layer
        self.resnet50.fc = nn.Linear(in_features, num_classes)

        # Dictionary to select activation function
        self.activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        
        self.activation = self.activations.get(activation, nn.ReLU())

    def forward(self, x):
        return self.resnet50(x)



# >>> ==================== Non Mixed Participant Changes ===================
# A CustomDataset Class - to read in participants without having overlapping frames in a given fold's splits.

## Defining a Custom Dataset class - to read in the frames of participants belonging to a particular fold's split
## This is to make sure that during training/validation/testing - the frames of participants remain present in 1 particular split
## i.e: non-mixed participants
class CustomDataset(Dataset):
    ## Initialise the CustomDataset class object to have the labels and the paths to all the image frames
    ## of all the participants beloning to a given fold's train/val/test split.
    def __init__(self, participants, study_data_path, transform=None):
        self.participants = participants
        self.study_data_path = study_data_path
        self.transform = transform

        self.images = []
        self.image_paths = []
        self.labels = []

        #seed


        for participant in self.participants:
            participant_path = f"{study_data_path}/{participant}/frames/"
            for class_label in class_labels:
                class_path = os.path.join(participant_path, class_label)

                for image_file in os.listdir(class_path):
                    image_path = os.path.join(class_path, image_file)
                    
                    ## Append only the image paths of all the frames for the given fold split's participant
                    ## You will read in the image only in batches using this initialized frame paths
                    self.image_paths.append(image_path)
                    
                    ## Correspondingly store the class the frame belongs to
                    label = int(class_label)
                    self.labels.append(label)
        self.n_samples = len(self.labels)

    def __len__(self):
        return self.n_samples
    
    ## When reading in the data during epoch steps - load the frames in batches - based on the given indexes.
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]

        image = Image.open(image_path).convert("RGB") #checkkkkk this!!!
        if self.transform:
            #apply transformat

            image = self.transform(image)
        return image, label
## <<< ==================== Non Mixed Participant Changes ===================




def test_model(model, test_loader, criterion):
    # track test loss
    test_loss = 0.0
    class_correct = list(0. for i in range(2))
    class_total = list(0. for i in range(2))
    #MODEL SAVE
    classes = [0,1]

    ## To store all the labels for generating classification report
    true_labels = []
    pred_labels = []
    
    # Set the model to evaluation mode
    model.eval()


    with torch.no_grad():

        # iterate over test data
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(inputs)

            # calculate the batch loss
            loss = criterion(output, labels)

            # update test loss 
            test_loss += loss.item()*inputs.size(0)

            # convert output probabilities to predicted class
            _, pred = torch.max(output, 1)

            # compare predictions to true label
            correct_tensor = pred.eq(labels.data.view_as(pred))
            correct = np.squeeze(correct_tensor.cpu().numpy()) if train_on_gpu else np.squeeze(correct_tensor.numpy())

            # append true and predicted labels for the batch for classification report
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(pred.cpu().numpy())

            # calculate test accuracy for each object class
            for i in range(len(labels.data)):
                label = labels.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1

    # average test loss
    test_loss = test_loss / len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.6f}\n")

    for i in range(2):
        if class_total[i] > 0:
            test_accuracy = (class_correct[i] / class_total[i]) * 100
            correct_pred = np.sum(class_correct[i])
            class_total_sum = np.sum(class_total[i])
            print(f"Test Accuracy of {classes[i]}: {test_accuracy} ({correct_pred}/{class_total_sum})")
        else:
            print(f"Test Accuracy of {classes[i]}: N/A (no training examples)' % ({classes[i]})")

    overall_test_accuracy = (np.sum(class_correct) / np.sum(class_total)) * 100
    overall_correct_preds = np.sum(class_correct)
    overall_class_total_sum = np.sum(class_total)
    print(f'\nTest Accuracy (Overall): {overall_test_accuracy} ({overall_correct_preds}/{overall_class_total_sum})')

    return true_labels, pred_labels, (overall_test_accuracy / 100), test_loss


def train_wrapper(study_a_participants, study_a_data_path, study_a_splits, model_output, model_last_save, num_epochs, resume=0, resume_fold=0, config = None):


    def train(resume = 0, resume_fold = 0, config= None): #MODEL SAVE
        # Define data transformations for training
        

        with wandb.init(config=config):
            config = wandb.config
            loss_var = config.criterion
            optimizer = config.optimizer
            step_size = config.step_size
            gamma = config.gamma
            batch_size = config.batch_size
            activation = config.activation
            weight_decay = config.weight_decay
            wandb.log({"resume": 0})
            wandb.log({"resume_fold": resume_fold})
            wandb.log({"job_path": model_output})
            wandb.log({"MODEL": "ResNet50 WDcay"})



            hot_encode = 0
            if loss_var == 'hinge':
                criterion = torch.nn.HingeEmbeddingLoss()
                #use tanh activation
                activation = 'tanh'
        
            elif loss_var == 'binary_crossentropy':
                criterion = torch.nn.BCELoss()
                #use sigmoid activation
                activation = 'sigmoid'
            elif loss_var == 'binary_crossentropylogit':
                criterion = torch.nn.BCEWithLogitsLoss()
                #use sigmoid activation
                activation = 'sigmoid'
            elif loss_var == 'categorical_crossentropy':
                criterion = torch.nn.CrossEntropyLoss()
                hot_encode = 1 
            elif loss_var == 'mean_squared_error':
                criterion = torch.nn.MSELoss()
            #create criterion with CCC

            

            print('IN TESTING')

            train_folds, val_folds, test_folds = createDataSplitsCNN(
            participants=study_a_participants,
            results_directory=model_output, 
            #train_fold_size=study_a_splits["train_fold_size"], 
            #val_fold_size=study_a_splits["val_fold_size"], 
            #test_fold_size=study_a_splits["test_fold_size"], 
            seed_value=42
            )
            print(f"Created model splits")

            ### Sanity Checks
            print("--" * 20)

            print(f"# of Train Folds : {len(train_folds)}")
            print(f"# of Val Folds : {len(val_folds)}")
            print(f"# of Test Folds : {len(test_folds)}")

            print("--" * 20)
            print(f"Train First Fold Participants: {train_folds[0]}")
            print(f"Val First Fold Participants: {val_folds[0]}")
            print(f"Test First Fold Participants: {test_folds[0]}")

            print("--" * 20)
            print(f"# of Train Fold Participants in First Fold: {len(train_folds[0])}")
            print(f"# of Val Fold Participants in First Fold: {len(val_folds[0])}")
            print(f"# of Test Fold Participants in First Fold: {len(test_folds[0])}")

            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            # Define data transformations for validation
            transform_val = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])


            ## Define data transformations for testing
            ## Perform Data Normalization
            transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            #study_a_full_dataset = datasets.ImageFolder(root=study_a_data_path, transform=transform_train)


            # Number of folds for cross-validation
            num_folds = 5  # You can adjust this based on your needs
            # Use StratifiedKFold to maintain class distribution in each fold
            #skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

            # Assuming full_dataset is the dataset you want to perform cdu -h --max-depth 1 cross-validation on
            # You may replace it with the actual dataset variable you are using
            
            
            for fold in range(len(train_folds)):
                print("--" * 20)
                print(f"Fold : {fold + 1} / {len(train_folds)}")
                print("--" * 20)

                if fold != 0: #temp
                    break
                
                classes = [0,1]
                model = ResNet50(num_classes=len(classes), activation=activation)
                if optimizer == 'adam':
                    optimizer = optim.Adam(model.parameters(), lr=0.001)
                elif optimizer == 'sgd':
                    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
                elif optimizer == 'adadelta':
                    optimizer = optim.Adadelta(model.parameters(), lr=0.001)

                # Decay LR by a factor of 0.1 every 10 epochs
                scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        
                if torch.cuda.is_available():
                    model.cuda()

                ## Given the train and validation indices of a fold, get the train, validation and test datasets
                ## NOTE: test_dataset is created by splitting the indices within the training dataset
                ## StratifiedKFold creates train/val split in the ratio of 80-20
                ## Within this 80% of training data, 10% is split for testing.
                train_dataset = CustomDataset(participants=train_folds[fold], study_data_path=study_a_data_path, transform=transform_train)
                val_dataset = CustomDataset(participants=val_folds[fold], study_data_path=study_a_data_path, transform=transform_val)
                test_dataset = CustomDataset(participants=test_folds[fold], study_data_path=study_a_data_path, transform=transform_test)

                image_datasets = {
                    "train": train_dataset,
                    "val": val_dataset,
                    "test": test_dataset
                }

                dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val", "test"]}
                print(f"# of frames in training dataset {len(train_dataset)}")
                print(f"# of frames in validation dataset {len(val_dataset)}")
                print(f"# of frames in testing dataset {len(test_dataset)}")

                with open(f"{model_output}/results_output_log.txt", "a") as results_log_file:
                    results_log_file.write(f"# of frames in training dataset {len(train_dataset)}\n")
                    results_log_file.write(f"# of frames in validation dataset {len(val_dataset)}\n")
                    results_log_file.write(f"# of frames in testing dataset {len(test_dataset)}\n")

                since = time.time()
                ## Count variable to keep track of the number of combinations in the hyper-parameter tuning
                search_count = 0
        

                #MODEL SAVE
                if resume == 1:
                    #if resume_fold is greater than fold, then skip
                    if resume_fold > fold:
                        print(f"Skipping fold {fold} as it is not the fold to resume from")
                        continue
                    
                    checkpoint = torch.load(f"{model_last_save}model_variables_{resume_fold}.pth")
                    epoch_restart = checkpoint['epoch']
                    train_losses = checkpoint['train_losses']
                    train_accuracies = checkpoint['train_accuracies']
                    val_losses = checkpoint['val_losses']
                    val_accuracies = checkpoint['val_accuracies']
                    model.load_state_dict(torch.load(f"{model_last_save}model_{resume_fold}_{epoch_restart}.pth"))
                    print(f"Resuming from epoch {epoch_restart}")
                    print(f"Resuming from fold {fold}")
                    resume = 0
                else:
                    epoch_restart = 1
                    train_losses = []
                    train_accuracies = []
                    val_losses = []
                    val_accuracies = []
                wandb.log({"fold": fold})



                
                if num_epochs == 0:
                    continue
                if epoch_restart >= num_epochs:
                    continue


                try:
                    search_count += 1
                    
                    #CHANGE - added wandb
                    wandb.log({"epochss": num_epochs, "batch_size": batch_size})
                    #MODEL SAVE - COMMENT THIS PART
                    ## To keep track of the accuracies to obtain the accuracy plots
                    #train_accuracies = []
                    #val_accuracies = []
                    #train_losses = []
                    #val_losses = []

                    dataloaders = {
                        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val', 'test']
                    }
                    
                    # Create a temporary directory to save training checkpoints
                    tempdir = model_output  ###########CHANGE to model_output and remove "with" (indentation changes)
                        ###CHANGE deleted savings
                    best_accuracy = 0.0

                    
                    print(f"Epoch_restart = {epoch_restart}")

                    #MODEL SAVE - EPOCH RESTART
                    for epoch in range(epoch_restart, num_epochs + 1):
                        print(f"Epoch {epoch}/{num_epochs}")
                        print("-" * 10)

                        ## Each epoch has a training and validation phase
                        for phase in ["train", "val"]:
                            print(f"Phase: {phase}")
                            if phase == "train":
                                model.train() # Set model to training mode
                            else:
                                model.eval() # Set model to evaluate mode
                        
                            running_loss = 0.0
                            running_corrects = 0
                            n_batch_iterations = len(dataloaders[phase])

                            scaler = GradScaler()
                            ## Iterate over the data
                            batch_count = 1
                            for inputs, labels in dataloaders[phase]:
                                #add times
                                #print('start')
                                #start_time = time.time()
                                inputs = inputs.to(device)
                                labels = labels.to(device)
                                #end_time = time.time()
                                #print(f"Time to move to device: {end_time - start_time}")
                                #print('to device')

                                # zero the parameter gradients
                                optimizer.zero_grad(set_to_none=True)

                                # forward
                                #start_time = time.time()
                                # track history if only in train
                                #with torch.set_grad_enabled(phase == 'train'):
                                #    outputs = model(inputs)
                                #    _, preds = torch.max(outputs, 1)
                                #    loss = criterion(outputs, labels)

                                    # backward + optimize only if in training phase
                                #    if phase == 'train':
                                #        loss.backward()
                                #        optimizer.step()

                                with torch.autocast(device_type='cuda', dtype=torch.float16):
                                    outputs = model(inputs)
                                    loss = criterion(outputs, labels)
                                
                                if phase == 'train':
                                    scaler.scale(loss).backward()
                                    scaler.step(optimizer)
                                    scaler.update()

                                #total_loss += loss.item()
                                _, preds = outputs.max(1)
                                #total += labels.size(0)
                                #correct += preds.eq(labels).sum().item()



                                #end_time = time.time()
                                #print(f"Time to forward and backward pass: {end_time - start_time}")
                                
                                
                                #print('finished training')
                                    

                                # statistics
                                running_loss += loss.item() * inputs.size(0)
                                running_corrects += torch.sum(preds == labels.data)
                                
                                if batch_count % 5 == 0:
                                    print(f"Epoch: {epoch}/{num_epochs}; Step: {batch_count}/{n_batch_iterations}")
                                batch_count += 1
                                # break ## break from iterating over the batches

                            if phase == "train":
                                scheduler.step()


                            epoch_loss = running_loss / dataset_sizes[phase]
                            epoch_acc = running_corrects.double() / dataset_sizes[phase]

                            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                            # CHANGE - commented below
                            # deep copy the model
                            #if phase == 'val' and epoch_acc > best_accuracy:
                            #    best_accuracy = epoch_acc
                            #    torch.save(model.state_dict(), best_model_params_path)

                            # Store accuracy for both training and validation
                            if phase == 'train':
                                train_accuracies.append(epoch_acc)
                                train_losses.append(epoch_loss)
                                wandb.log({"epoch": epoch, "train_loss": epoch_loss, "train_accuracy": epoch_acc})
                            elif phase == 'val':
                                val_accuracies.append(epoch_acc)
                                val_losses.append(epoch_loss)
                                wandb.log({"epoch": epoch, "val_loss": epoch_loss, "val_accuracy": epoch_acc})


                            if phase == 'val':
                                true_labels, pred_labels, test_accuracy, test_loss = test_model(model, dataloaders['val'], criterion)
                                if loss_var == 'hinge':
                                    #see how many values are under 0 for y_pred
                                    #print(np.any(pred_labels.cpu().numpy() < 0))
                                    pred_labels = (pred_labels > 0).float()
                                
                                    #y_train_tensor = y_train_tensor.float()
                                    #make y_pred binary output, from float
                                    #print(np.unique(y_pred.cpu().numpy()))
                                    pred_labels = pred_labels.to(torch.int64)
                                elif loss_var == 'binary_crossentropy' or loss_var == 'binary_crossentropylogit':
                                    #see how many values are under 0 for y_pred
                                    #print(np.unique(y_pred.cpu().numpy()))
                                    pred_labels= pred_labels.to(torch.int64)



                                test_metrics = get_metrics(true_labels, pred_labels)
                                wandb.log({"Val Macro Precision": test_metrics['precision'], "Val Macro Recall": test_metrics['recall'],
                                    "Val Macro F1": test_metrics['f1'], "Val Accuracy": test_metrics['accuracy'], 
                                    
                                    "Val Accuracy Tolerant": test_metrics['accuracy_tolerant'], 
                                    "Val Precision Tolerant": test_metrics['precision_tolerant'], 
                                    "Val Recall Tolerant": test_metrics['recall_tolerant'], "Val F1 Tolerant": test_metrics['f1_tolerant'], "Epoch": epoch})
                                print(f"Val Accuracy: {test_accuracy:.4f}")
                                print(f"Val Loss: {test_loss:.4f}")
                                print(f"Val Macro Precision: {test_metrics['precision']:.4f}")
                                print(f"Val Macro Recall: {test_metrics['recall']:.4f}")
                                print(f"Val Macro F1: {test_metrics['f1']:.4f}")
                                print(f"Val Accuracy Tolerant: {test_metrics['accuracy_tolerant']:.4f}")
                                print(f"Val Precision Tolerant: {test_metrics['precision_tolerant']:.4f}")
                                print(f"Val Recall Tolerant: {test_metrics['recall_tolerant']:.4f}")
                                print(f"Val F1 Tolerant: {test_metrics['f1_tolerant']:.4f}")

                                
                            # break ## break from iterating over model training and validation 
                            
                            #CHANGE - added model saving
                            #save model every 2 epochs after validating
                            if phase == 'val' and epoch % 10 == 0:
                                model_chpt = os.path.join(tempdir, f'model_chpt_{fold}_{epoch}.pth')
                                torch.save(model.state_dict(), model_chpt)
                                checkpoint = {
                                    'epoch': epoch,
                                    'train_losses': train_losses,
                                    'train_accuracies': train_accuracies,
                                    'val_losses': val_losses,
                                    'val_accuracies': val_accuracies,
                                    'classes': classes
                                }
                                torch.save(checkpoint, os.path.join(tempdir,f'model_variables_{fold}_{epoch}.pth'))
                                #MODEL SAVE
                                #save last model in last_model_save_dir
                                torch.save(model.state_dict(), f"{model_last_save}model_{fold}_{epoch}.pth")
                                torch.save(checkpoint, f"{model_last_save}model_variables_{fold}.pth")
                                #if it exists, delete the previous model
                                if os.path.exists(f"{model_last_save}model_{fold}_{epoch-2}.pth"):
                                    os.remove(f"{model_last_save}model_{fold}_{epoch-2}.pth")
                                

                        print()
                    time_elapsed = time.time() - since
                    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
                    print(f'Best val Acc: {best_accuracy:4f}')

                    #CHANGE - changed modelsaving
                    # load best model weights
                    model.load_state_dict(torch.load(model_chpt))

                    print(f"Begin testing the model")
                    true_labels, pred_labels, test_accuracy, test_loss = test_model(model, dataloaders['test'], criterion)
                    if loss_var == 'hinge':
                        #see how many values are under 0 for y_pred
                        #print(np.any(pred_labels.cpu().numpy() < 0))
                        pred_labels = (pred_labels > 0).float()
                       
                        #y_train_tensor = y_train_tensor.float()
                        #make y_pred binary output, from float
                        #print(np.unique(y_pred.cpu().numpy()))
                        pred_labels = pred_labels.to(torch.int64)
                    elif loss_var == 'binary_crossentropy' or loss_var == 'binary_crossentropylogit':
                        #see how many values are under 0 for y_pred
                        #print(np.unique(y_pred.cpu().numpy()))
                        pred_labels= pred_labels.to(torch.int64)



                    test_metrics = get_metrics(true_labels, pred_labels)
                    wandb.log({"Test Macro Precision": test_metrics['precision'], "Test Macro Recall": test_metrics['recall'],
                        "Test Macro F1": test_metrics['f1'], "Test Accuracy": test_metrics['accuracy'],
                        "Test Accuracy Tolerant": test_metrics['accuracy_tolerant'],
                        "Test Precision Tolerant": test_metrics['precision_tolerant'],
                        "Test Recall Tolerant": test_metrics['recall_tolerant'], "Test F1 Tolerant": test_metrics['f1_tolerant'], "Epoch": epoch})


                    report = classification_report(true_labels, pred_labels, target_names=classes)
                    print(report)
                    wandb.log({"report": report, "val_loss": test_loss, "val_accuracy": test_accuracy})


                    # Calculate the confusion matrix
                    conf_matrix = confusion_matrix(true_labels, pred_labels, tolerance = 1)


                    """
                    BEGIN: Subplots for training and validation accuracy for varying batch_sizes
                    """
                    # Adjust layout and save the figure for training & validation accuracy for varying batch_sizes
                    #batch_size_results_path = model_output + 'batch_size_results/'

                    #if not os.path.exists(batch_size_results_path):
                    #    os.makedirs(batch_size_results_path)

                   ###CHANGE - indentation change ends here

                    with open(f"{model_output}/results_output_log.txt", "a") as results_log_file:
                        results_log_file.write("\n")
                        results_log_file.write(f"------------ NeckFace 30fps: BEGIN SEARCH: {search_count} ------------" + "\n")
                        results_log_file.write("------------ TYPE ------------" + "\n")

                        results_log_file.write(
                            f"Epoch: {epoch}\n"
                            f"Batch Size: {batch_size}\n"
                        )

                        results_log_file.write("------------ METRICS ------------" + "\n")
                        results_log_file.write(f'Training Loss: {train_losses[-1]:.4f}' + '\n')
                        results_log_file.write(f'Training Accuracy: {train_accuracies[-1]:.4f}' + '\n')
                        results_log_file.write(f'Validation Loss: {val_losses[-1]:.4f}' + '\n')
                        results_log_file.write(f'Validation Accuracy: {val_accuracies[-1]: .4f}' + '\n')
                        results_log_file.write(f'Test Loss: {test_loss:.4f}' + '\n')
                        results_log_file.write(f'Test Accuracy: {test_accuracy:.4f}' + '\n')

                        results_log_file.write("------------ CLASSIFICATION REPORT ------------" + "\n")
                        results_log_file.write(report + '\n')


                        results_log_file.write(f"------------ END SEARCH ------------" + "\n")
                        results_log_file.write("\n")

                    #MODEL SAVE - FINISH AND START NEW WANDB RUN IF fold is lower than 4
                    wandb.finish()
                    if fold < 4:
                        wandb.init(project="neckface_IR_30fps_v4")
                        # Config is a variable that holds and saves hyperparameters and inputs
                        config = wandb.config
                        optimized = config.optimizer
                        momentum = config.momentum
                        wandb.log({"optimized": optimized, "momentum": momentum})
                        wandb.log({"MESSAGE": "testing wandb sweep"})


                except Exception as e:
                    print('Exception thrown during model training')
                    print(e)
                    wandb.finish()
                    if fold < 4:
                        wandb.init(project="neckface_IR_30fps_v4")
                        # Config is a variable that holds and saves hyperparameters and inputs
                        config = wandb.config
                        optimized = config.optimizer
                        momentum = config.momentum
                        wandb.log({"optimized": optimized, "momentum": momentum})
                        wandb.log({"MESSAGE": "testing wandb sweep"})
                    with open(f"{model_output}/results_output_log.txt", "a") as results_log_file:
                        results_log_file.write(
                            f"Search Count: {search_count} \n"
                            f"Epoch: {epoch}\n"
                            f"Batch Size: {batch_size}\n"
                            f"Exception {e} thrown during model training :- \n"
                            f"{traceback.format_exc()}\n"
                        )


                # break ## break from batch size
                # break ## break from epochs
                #MODEL SAVE - comment break
                break ## break from folds
    return train

def main():
    
    #CHANGE - added wandb and indentations for all inside main
        # Config is a variable that holds and saves hyperparameters and inputs
        

        color_channel = "BGR" #CHECKKKKKKKKKKKKKKK
        data_frame_rate = 30
        dataset_path = "../../../../data/final_study_data_BGR_30fps"
        output_directory = "../../../../data/training_outputs/"
        #MODEL SAVE
        last_model_save_dir = output_directory + "model_data/" + "resnet50_wdecay_30fps/"

        ## Define the path for storing model outputs
        now = datetime.datetime.now()

        ## >>> ==================== Non Mixed Participant Changes ===================

        neckface_participant_data_path = dataset_path
        neckface_participants = [participant for participant in os.listdir(neckface_participant_data_path) if participant not in files_to_ignore]
        print('NECKFACE PARTICIPANTS', neckface_participants)
        
        #exclude participants:
        excluded_p = [13,17,27,29,30]
        
        neckface_participants = [participant for participant in neckface_participants if int(participant) not in excluded_p]

        ## Define the path for storing model outputs
        now = datetime.datetime.now()
        neckface_output_path = output_directory + f"resnet50_neckface_{data_frame_rate}_fps_" + now.strftime("%Y-%m-%d_%H-%M-%S") + '/'

        if not os.path.exists(neckface_output_path):
            os.makedirs(neckface_output_path)

        # 70/20/10 split - total 30 participants
        neckface_fold_splits = {
            "train_fold_size": 18,
            "val_fold_size": 4,
            "test_fold_size": 3
        }

        seed_value = 42
        # # Set seed
        random.seed(seed_value)
        np.random.seed(seed_value)
        #tf.random.set_seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

        
        #MODEL SAVE
        if not os.path.exists(last_model_save_dir):
            os.makedirs(last_model_save_dir)
            resume = 0
            resume_fold = 0
        else:
            #look at the models saved - they are in the format model_FOLD_EPOCH.pth. If epoch is less than 200, save variable "RESUME" as True. Look for each fold, start from 4 to 0
            for fold in range(4, -1, -1):
                #see if model exists, regardless of epoch (any epoch numbe is fine, look for partial string match with fold number and .pth extension)
                path_start = f"{last_model_save_dir}model_{fold}_*.pth"
                if glob.glob(path_start):
                    print('MODEL EXISTS FOR FOLD', fold)
                    #wandb.log({"resume": 1})
                    resume = 1
                    resume_fold = fold
                    #wandb.log({"resume_fold": resume_fold})
                    break
                else:
                    resume = 0
                    resume_fold = 0
                    #wandb.log({"resume": 0})
                    #wandb.log({"resume_fold": resume_fold})
                    print('MODEL DOES NOT EXIST FOR FOLD', fold)
        
       
        print("started from", resume_fold)
        print("resume is", resume)



        # Define the sweep configuration
        sweep_config = {
            "method":  "random",
            #goal is to minimize mse
            "metric": {"goal": "maximize", "name": "macro_accuracy"},
            "parameters": {
                "criterion": {"values": ['categorical_crossentropy']},
                "optimizer": {"values": ['adam', 'sgd', 'adadelta']},
                "step_size": {"values": [10,30,60]},
                "gamma": {"values": [0.01, 0.1, 0.3]},
                "batch_size": {"values": [256]},
                "activation": {"values": ['relu', 'leaky_relu', 'tanh', 'sigmoid']},
                "weight_decay": {"values": [0.001,0.01,0.1]}
            },
        }  
        

        num_epochs = 200

        sweep_id = wandb.sweep(sweep_config,project="neckface_IR_30fps_v4")

        train_func = train_wrapper(study_a_participants=neckface_participants,
                                   study_a_data_path=neckface_participant_data_path,
                                    study_a_splits=neckface_fold_splits,
                                    model_output=neckface_output_path,
                                    model_last_save=last_model_save_dir,
                                    num_epochs=num_epochs,
                                    resume=resume,
                                    resume_fold=resume_fold)

        #CHANGE - added wandb sweep
        wandb.agent(sweep_id, train_func)

if __name__ == '__main__':
    main()

