import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
import random
import os
import sys
from torch.utils.tensorboard import SummaryWriter

from resources.data_utils_pt import split_dataset, CustomDataset
from torch.utils.data import DataLoader
from resources.model_utils_pt import Model
#from resources.savers import TrainingSaver
from resources.utils import label_str_to_dec

# *** Setting global constants ***
num_classes = 15 # to change according to the dictionary size
char_list = [str(i) for i in range(1, num_classes+1)]
char2int = {char: i for i, char in enumerate(char_list)}
num_replicates = 1000
dataset_size = num_classes * num_replicates
image_dimension = (80, 80)

dataset = 'final' # name of the data set
path_final = 'noise_medium_3/preprocessed_80/' # preprocessed images directory
filename = 'noise_medium_3/labels_spot_binary.csv' # label file path

# for spliting data set
ds_list = [0, 0.5, 0.75, 0.875, 0.9375, 0.96875, 0.9875, 0.9975]
ds_dict = {i+1:round((1-j)*dataset_size*0.8) for i,j in enumerate(ds_list)}
rank = 1 # change to choose training data set size
ds = ds_dict.get(rank) # training dataset size

path_hp = dataset+'/'+str(ds)+'pytorch/' # directory where training details are saved
if not os.path.exists(path_hp):
    os.makedirs(path_hp)
save_path = path_hp + "model.pt"

# *** Load data ***
# labels
labels = []
with open(filename, "r") as csvFile:
    for row in csvFile:
        labels.append(row[:-1])
    labels = np.asarray(labels)
    main_labels = label_str_to_dec(labels[0:dataset_size], char2int)

# collecting the paths to all images in a set
image_prefix = "FImg_ID_"
image_suffix = ".jpg"
images_str = [f"{path_final}{image_prefix}{img_idx}{image_suffix}" for img_idx in range(1, dataset_size + 1)]
main_dataset = pd.DataFrame({"img_path": images_str, "label": main_labels})

# *** Training ***
'''
generation_params = {"dim": image_dimension,
                     "nb_classes": num_classes,
                     "column_img": "img_path",
                     "column_label": "label",
                    }
'''

# hyperparameters
epochs = 500
min_delta = 0.0000001
patience = 20
monitor = 'val_loss'
mode = 'min' # stop when monitor metric stops decreasing
'''
early_stopping_loss = EarlyStopping(monitor=monitor,
                                    min_delta=min_delta,
                                    patience=patience,
                                    verbose=0,
                                    mode='auto',
                                    baseline=None,
                                    restore_best_weights=True
                                   )

early_stop_callback = EarlyStopping(
   monitor=monitor,
   min_delta=min_delta,
   patience=patience,
   verbose=0,
   mode='min'
)
trainer = Trainer(callbacks=[early_stop_callback])
'''
batch_size_list = [4,8,16,27,32]
batch_size = batch_size_list[0]
lr = 0.001
optimizer_name = 'Adam'
seed = 25 # seed for random initialization

# *** Train ***


# build model
model = Model()

# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()


# Split data set
df_train, df_valid, df_test = split_dataset(data_frame=main_dataset,
                                            rank=rank,
                                            column_label="label",
                                            random_state=25
                                            )

# data loaders
train_loader = DataLoader(CustomDataset(df_train), batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(CustomDataset(df_valid), batch_size=100, shuffle=False)
test_loader = DataLoader(CustomDataset(df_test), batch_size=100, shuffle=False)


# save hyperparameters
hp_filename = path_hp + "hp_details.txt"
with open(hp_filename,"w") as f:
    f.write('Dataset: %s \n' % (dataset))
    f.write('Batch size: %i \n' % (batch_size))
    f.write('Initialization random seed: %i \n' % (seed))
    f.write('Training epochs: %i \n' % (epochs))
    f.write('Optimizer: %s (lr = %f) \n' % (optimizer_name, lr))
    f.write('Early Stopping: %s (min_delta = %.9f, patience = %i) \n' % (monitor, min_delta, patience))


# Passing data through the model for both training and testing purposes
def fwd_pass(X, y, train = False):
    if train:
        model.zero_grad()

    # compare y and model outputs for accuracy
    outputs = model(X)
    matches = [torch.argmax(i) == j for i, j in zip(outputs, y)]
    acc = matches.count(True)/len(matches)
    loss = criterion(outputs, y)

    if train:
        loss.backward()
        optimizer.step()
    return acc, loss


def test():
    X,y = valid_loader


# Early stopping
min_epochs = 10
early_stop = False
min_val_loss = np.Inf

# Track training
writer = SummaryWriter()

val_acc_history = []
val_loss_history = []

train_acc_history = []
train_loss_history = []

for epoch in range(epochs):

    epoch_training_loss = [] # for tabulating training loss for this epoch
    epoch_training_accuracy = []
    running_loss = 0.0 # for printing statistics every 2000 batches
    for i, batch in enumerate(train_loader,0):

        X, y = batch
        n_samples = X.size(0)

        # Transfer Data to GPU if available
        if torch.cuda.is_available():
            X, y = X.cuda(), y.cuda()

        # get accuracy and loss for this batch
        acc, loss = fwd_pass(X, y, train = True)
        epoch_training_loss.append(loss.item())
        epoch_training_accuracy.append(acc)

        # print statistics
        running_loss += loss.item()
        if i % 1000 == 999:    # print every 1000 batches
            print(f'[Epoch %d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 1000))
            running_loss = 0.0


    writer.add_scalar("Loss/train", np.mean(epoch_training_loss), epoch)
    writer.add_scalar("Accuracy/train", np.mean(epoch_training_accuracy), epoch)

    epoch_val_loss = []
    epoch_val_accuracy = []
    for X, y in valid_loader:
        if torch.cuda.is_available():
            X, y = X.cuda(), y.cuda()

        acc, loss = fwd_pass(X,y, train = False) # not training on validation set
        epoch_val_loss.append(loss.item())
        epoch_val_accuracy.append(acc)

    curr_val_loss = np.mean(epoch_val_loss)
    writer.add_scalar("Loss/validation", curr_val_loss, epoch)
    writer.add_scalar("Accuracy/validation", np.mean(epoch_val_accuracy), epoch)
    #val_loss_history.append(avg_val_loss)
    #train_acc_history.append(acc)
    #train_loss_history.append(loss)

    # Implement early stopping if validation loss is not improving
    if curr_val_loss < min_val_loss:
          # Save the model
          torch.save(model.state_dict(), save_path)
          epochs_no_improve = 0
          min_val_loss = curr_val_loss
    else:
        epochs_no_improve += 1

    if epoch > min_epochs and epochs_no_improve == patience:
        print('Early stopping!' )
        early_stop = True
        break
    else:
        continue
    break


writer.flush()
print('Finished Training')


# *** Save training results ***
'''
experiment_saver = TrainingSaver(path_out=path_hp,
                                 stopped_epoch=stopped_epoch,
                                 nb_classes=num_classes,
                                 df_train=df_train,
                                 df_valid=df_valid,
                                 df_test=df_test,
                                 column_label="label",
                                 model=model,
                                 prefix=prefix
                                 )
'''
