from corn_class_new import CornDataset
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import json

from engine import train_one_epoch, evaluate
import utils
import transforms as T


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

if __name__ == "__main__":
    # use our dataset and defined transformations/ use same dataset for train and test for this
    # dataset_test = CornDataset('0707_corn/images', get_transform(train=False))
    dataset_test = CornDataset('corn_data/test_all_org',
                'corn_data/bbox_labels_test_all_org_100.pkl', get_transform(train=False))
    
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # two classes: corn and background
    num_classes = 2


    # save trained model for inference    
    # torch.save(model, './output/faster-rcnn-corn_bgr8.pt')
    model = torch.load('./output/faster-rcnn-corn_mobile_all_combine_ep99.pt')
    model.to(device)
    model.eval()
    evaluate(model, data_loader_test, device=device)
