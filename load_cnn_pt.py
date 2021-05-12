import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from resources.model_utils_pt import Model
from resources.data_utils_pt import split_dataset, CustomDataset
from torch.utils.data import DataLoader
from resources.model_utils_pt import Model
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

# build model
model = Model()

# load check point
model.load_state_dict(torch.load(save_path))

batch_size_list = [4,8,16,27,32]
batch_size = batch_size_list[0]
seed = 25 # seed for random initialization

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

# Evaluate model on test data
test_acc = []
for X, y in valid_loader:
    if torch.cuda.is_available():
        X, y = X.cuda(), y.cuda()
    with torch.no_grad():
        outputs = model(X)
        matches = [torch.argmax(i) == j for i, j in zip(outputs, y)]
        acc = matches.count(True)/len(matches)
        test_acc.append(acc)
test_acc = np.round(np.mean(test_acc) * 100, 3)

# Report accuracy
print(f'Accuracy of the network on {df_test.shape[0]} test images: {test_acc}%')
