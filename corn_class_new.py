import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import json
import pickle


class CornDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, pkl_path, transforms=None):
        self.data_dir = data_dir
        self.transforms = transforms   
        f = open(pkl_path, 'rb')
        annotations1 = pickle.load(f)
        self.filenames = annotations1[0]
        self.annotations = annotations1[1]

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        img_path = os.path.join(self.data_dir, img_name) # TODO update path
        img = Image.open(img_path).convert("RGB")

        num_objs = len(self.annotations[img_name])
        boxes=[]
        for i in range(num_objs):
            xmin = float(self.annotations[img_name][i][0])
            ymin = float(self.annotations[img_name][i][1])
            w = float(self.annotations[img_name][i][2])
            h = float(self.annotations[img_name][i][3])
            xmax = xmin + w
            ymax = ymin + h
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.annotations)