import random

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, sampler
from models.configurations import configurations
import os
import shutil
import torchvision
import torchvision.transforms as transforms
import glob

from sklearn.model_selection import train_test_split
from utils.utils import Merge_CSV

from dataset import Dataset

data_folder = os.path.join(os.getcwd(), "data")
TRAIN_SPLIT = 0.7
TEST_SPLIT = 0.15
VALIDATION_SPLIT = 0.15

def get_loader(dataset, batchSize=100, percentage=1,target_image_w=299,target_image_h=299):

    if dataset == "Lung_Opacity":
        CFG = configurations["datasets"]["Lung_Opacity"]

        test_image_path = os.path.join(CFG['path'], CFG['test_images'])
        train_image_path = os.path.join(CFG['path'], CFG['train_images'])
        validation_image_path = os.path.join(CFG['path'], CFG['validation_images'])
        
        metadata = os.path.join(data_folder, "Lung_Opacity.metadata.xlsx")

        os.makedirs(train_image_path, exist_ok=True)
        os.makedirs(test_image_path, exist_ok=True)

        all_files = [file for file in os.listdir(CFG['path']) if file.endswith(".png")]
        train_files, remaining_files = train_test_split(all_files, test_size=(1 - TRAIN_SPLIT))
        test_files, validation_files = train_test_split(remaining_files,
                                                        test_size=(VALIDATION_SPLIT / (TEST_SPLIT + VALIDATION_SPLIT)))

    elif dataset == 'COVID19':
        CFG = configurations["datasets"]["COVID19"]

        test_image_path = os.path.join(CFG['path'], CFG['test_images'])
        train_image_path = os.path.join(CFG['path'], CFG['train_images'])
        validation_image_path = os.path.join(CFG['path'], CFG['validation_images'])

        metadata = os.path.join(data_folder, "COVID.metadata.xlsx")

        os.makedirs(train_image_path, exist_ok=True)
        os.makedirs(test_image_path, exist_ok=True)
        os.makedirs(validation_image_path, exist_ok=True)

        all_files = [file for file in os.listdir(CFG['path']) if file.endswith(".png")]
        train_files, remaining_files = train_test_split(all_files, test_size=(1 - TRAIN_SPLIT))
        test_files, validation_files = train_test_split(remaining_files,
                                                        test_size=(VALIDATION_SPLIT / (TEST_SPLIT + VALIDATION_SPLIT)))

    elif dataset == 'Pneumonia':
        CFG = configurations["datasets"]["Pneumonia"]

        test_image_path = os.path.join(CFG['path'], CFG['test_images'])
        train_image_path = os.path.join(CFG['path'], CFG['train_images'])
        validation_image_path = os.path.join(CFG['path'], CFG['validation_images'])

        metadata = os.path.join(data_folder, "Viral Pneumonia.metadata.xlsx")

        os.makedirs(train_image_path, exist_ok=True)
        os.makedirs(test_image_path, exist_ok=True)
        os.makedirs(validation_image_path, exist_ok=True)

        all_files = [file for file in os.listdir(CFG['path']) if file.endswith(".png")]
        train_files, remaining_files = train_test_split(all_files, test_size=(1 - TRAIN_SPLIT))
        test_files, validation_files = train_test_split(remaining_files,
                                                        test_size=(VALIDATION_SPLIT / (TEST_SPLIT + VALIDATION_SPLIT)))

    # Test dataset for conditional GAN for 2 classes Lung Opacity & Normal
    elif dataset == "LungOpacity_Normal":
        CFG = configurations["datasets"]["LungOpacity_Normal"]

        test_image_path = os.path.join(CFG['path'], CFG['test_images'])
        train_image_path = os.path.join(CFG['path'], CFG['train_images'])
        validation_image_path = os.path.join(CFG['path'], CFG['validation_images'])

        # Creating new metadata for the combined dataset
        if not os.path.exists(os.path.join(data_folder, "Opacity_Normal.metadata.xlsx")):
            Merge_CSV(data_folder)


        metadata = os.path.join(data_folder, "Opacity_Normal.metadata.xlsx")

        os.makedirs(train_image_path, exist_ok=True)
        os.makedirs(test_image_path, exist_ok=True)
        os.makedirs(validation_image_path, exist_ok=True)

        all_files = [file for file in os.listdir(CFG['path']) if file.endswith(".png")]
        train_files, remaining_files = train_test_split(all_files, test_size=(1 - TRAIN_SPLIT))
        test_files, validation_files = train_test_split(remaining_files,
                                                        test_size=(VALIDATION_SPLIT / (TEST_SPLIT + VALIDATION_SPLIT)))

    else:
        raise Exception("dataset name not correct (or not implemented)")

    dst = train_image_path
    if not os.listdir(train_image_path):
        for filename in train_files:
            src = os.path.join(CFG['path'], filename)
            dst = train_image_path
            shutil.copy(src, dst)

    if not os.listdir(test_image_path):
        for filename in test_files:
            src = os.path.join(CFG['path'], filename)
            dst = test_image_path
            shutil.copy(src, dst)

    if not os.listdir(validation_image_path):
        for filename in validation_files:
            src = os.path.join(CFG['path'], filename)
            dst = validation_image_path
            shutil.copy(src, dst)

    path = pd.ExcelFile(metadata,  engine='openpyxl')
    data = pd.read_excel(path)

    files = pd.DataFrame({"FILE NAME": pd.Series(dtype="str"),
                          "FORMAT": pd.Series(dtype="str"),
                          "SIZE": pd.Series(dtype="str"),
                          "URL": pd.Series(dtype="str"),
                          "LABEL": pd.Series(dtype="int64")})

    files["FILE NAME"] = [f for f in os.listdir(dst) if os.path.isfile(os.path.join(dst, f))]

    file_names = files["FILE NAME"].tolist()
    merged_df = pd.DataFrame()
    merged_df["FILE NAME"] = file_names


    merged_df["LABEL"] = data.iloc[(np.where(merged_df["FILE NAME"] == files["FILE NAME"]))]["LABEL"]

    labels = merged_df["LABEL"].tolist()

    # Shuffle both input images and labels
    joined_list = list(zip(file_names, labels))
    random.shuffle((joined_list))

    file_names, labels = zip(*joined_list)

    data = Dataset(dst, file_names, labels, target_image_w=target_image_w, target_image_h=target_image_h,
                   transform=True)

    loader = DataLoader(data, batch_size=batchSize, collate_fn=lambda x: x)
    return loader