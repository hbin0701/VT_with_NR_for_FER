import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import pandas as pd
import os

class FerplusDataset(Dataset):

    def __init__(self, data_dir:str, mode:str, transforms=None):
        self.data_dir = data_dir
        self.transforms = transforms

        if mode == "train":
            self.label_mode = "Training"
            self.img_dir = os.path.join(self.data_dir, "train")
        elif mode == "valid":
            self.label_mode = "PrivateTest"
            self.img_dir = os.path.join(self.data_dir, "valid")
        elif mode == "test":
            self.label_mode = "PublicTest"
            self.img_dir = os.path.join(self.data_dir, "test")

        self.label_file = os.path.join(self.data_dir, "fer2013new_jin2.csv")
        self.dataframe = self.get_images(self.label_mode)


    def __getitem__(self, idx: int):
        #fname = os.path.join(self.img_dir, self.dataframe.iloc[idx]['Image Name'][:-4] + '_x8_SR.png')
        fname = os.path.join(self.img_dir, self.dataframe.iloc[idx]['Image Name'])
        img = Image.open(fname).convert('RGB')
        label = self.dataframe.iloc[idx]['emotion']

        if self.transforms is not None:
            img = self.transforms(img)

        return img, torch.tensor(label).to(torch.long)


    def __len__(self):
        return len(self.dataframe)


    def get_images(self, label_mode: str): 
        fer_meta = pd.read_csv(self.label_file)
        imgs = fer_meta.loc[fer_meta['Usage'] == label_mode][["Image Name", "emotion"]]
        return imgs


class AffectnetDataset(Dataset):

    def __init__(self, data_dir: str, mode: str, transforms=None):
        self.data_dir = data_dir
        self.transforms = transforms
        self.mode = mode

        if self.mode == "train":
            self.img_dir = os.path.join(self.data_dir, "train_set/images")
            self.label_dir = os.path.join(self.data_dir, "train_set/annotations")
        elif self.mode == "test":
            self.img_dir = os.path.join(self.data_dir, "val_set/images")
            self.label_dir = os.path.join(self.data_dir, "val_set/annotations")

        valid_emotions = []

        for label_file in os.listdir(self.label_dir):
            # 0: Neutral, 1: Happiness, 2: Sadness, 3: Surprise, 4: Fear, 5: Disgust, 6: Anger,
            # 7: Contempt, 8: None, 9: Uncertain, 10: No - Face
            if '_exp.npy' in label_file and int(np.load(os.path.join(self.label_dir, label_file))) <= 7:
                valid_emotions.append(label_file.split('_')[0])

        self.imgs = valid_emotions

    def __getitem__(self, idx):
        img_file = os.path.join(self.img_dir, f"{self.imgs[idx]}.jpg")
        img_label = os.path.join(self.label_dir, f"{self.imgs[idx]}_exp.npy")

        img = Image.open(img_file)
        label = int(np.load(img_label))

        if self.transforms is not None:
            img = self.transforms(img)

        return img, torch.tensor(label).to(torch.long)

    def __len__(self):
        return len(self.imgs)


class RafdbDataset(Dataset):

    def __init__(self, data_dir, mode="train", aligned="aligned", transforms=None):

        f = open(os.path.join(data_dir, "list_patition_label.txt"), "r")

        self.data_dir = os.path.join(data_dir, aligned)
        self.transforms = transforms
        self.mode = mode

        train = []
        test = []
        train_labels = []
        test_labels = []

        for img in os.listdir(self.data_dir):
            if "train" in img:
                train.append(img)
            elif "test" in img:
                test.append(img)

        if aligned == "aligned":
            train = sorted(train, key=lambda x: int(x.split("_")[1]))
            test = sorted(test, key=lambda x: int(x.split("_")[1]))
        
        elif aligned == "original":
            train = sorted(train, key=lambda x: int(x.split(".")[0].split("_")[-1]))
            test = sorted(test, key=lambda x: int(x.split(".")[0].split("_")[-1]))
        
        self.img_file = train if self.mode == "train" else test

        # Labels originally [ 1, 2, ..., 7] => [0, 1, ..., 6]
        for line in f:
            if "train" in line:
                train_labels.append(int(line.strip()[-1]) - 1)
            elif "test" in line:
                test_labels.append(int(line.strip()[-1]) - 1)
        
        self.imgs = train if mode == "train" else test
        self.labels = train_labels if mode == "train" else test_labels

    def __getitem__(self, idx):
        img_file = os.path.join(self.data_dir, self.imgs[idx])

        img = Image.open(img_file)
        label = self.labels[idx]

        if self.transforms is not None:
            img = self.transforms(img)

        return img, torch.tensor(label).to(torch.long)

    def __len__(self):
        return len(self.imgs)
