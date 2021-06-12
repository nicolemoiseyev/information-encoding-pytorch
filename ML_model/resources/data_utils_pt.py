import numpy as np
import pandas as pd
#from PIL import Image
from torchvision import transforms

import torch

from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset
import pandas as pd
import os
import cv2
#from PIL import Image
import torch

class CustomDataset(Dataset):
    """Generates data for PyTorch"""
    def __init__(self, data_frame: pd.DataFrame, column_img="img_path", column_label="label"):
        """Initialization"""
        self.data_frame = data_frame
        self.column_img = column_img
        self.column_label = column_label

    def __len__(self):
        return int(np.floor(len(self.data_frame)))

    def __getitem__(self, index):

        sample = self.data_frame.iloc[index]
        image_path = sample[self.column_img]

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = transforms.ToTensor()(img)
        img = img.view(1, 80, 80)
        y_label = torch.tensor((sample[self.column_label]))
        return img, y_label


def split_dataset(data_frame: pd.DataFrame, rank: int, column_label="label", random_state=25):

    """ Split training, test and validation sets
        ids_main_data: ids of all available data, in this case images
        labels_main_data: a list of labels in integer form for all available data
    """
    df_train, df_test, _, _ = train_test_split(data_frame, data_frame[column_label],
                                               stratify=data_frame[column_label],
                                               test_size=0.1,
                                               shuffle=True, random_state=random_state)
    if rank == 1:
        df_train, df_valid, _, _ = train_test_split(df_train,
                                                    df_train[column_label],
                                                    stratify=df_train[column_label],
                                                    test_size=0.11111,
                                                    shuffle=True,
                                                    random_state=random_state)
    else:
        df_train, df_valid, _, _ = train_test_split(df_train,
                                                    df_train[column_label],
                                                    stratify=df_train[column_label],
                                                    test_size=0.11111,
                                                    shuffle=True,
                                                    random_state=random_state)

        seg_ratio = [0.5, 0.75, 0.875, 0.9375, 0.96875, 0.9875, 0.9975]
        df_train, df_left_over, _, _ = train_test_split(df_train,
                                                        df_train[column_label],
                                                        stratify=df_train[column_label],
                                                        test_size=seg_ratio[rank - 2],
                                                        shuffle=True,
                                                        random_state=random_state)
    df_train.reset_index(drop=True, inplace=True)
    df_valid.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    return df_train, df_valid, df_test
