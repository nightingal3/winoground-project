import clip
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms
import numpy as np

from data.mscoco import COCO, COCODataset

import pdb

COCO_PATH="/projects/tir1/corpora/COCO/"
BATCH_SIZE=32

if __name__ == "__main__":
    default_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = COCODataset(root=COCO_PATH, caption_year="2017", image_transforms=None)
    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model, preprocess = clip.load("ViT-B/32")
    model = model.float().cuda()

    loss = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters()) 
    
    for i in range(1000):
        optimizer.zero_grad()
        inds = np.random.choice(len(dataset), BATCH_SIZE)
        train_data = [dataset[i] for i in inds]
        #train_images = [train_data['image_0'] + train_data['image_1']]
        train_images = [train_data[i]["image"] for i in range(len(train_data))]
        train_text = [train_data[i]["caption"] for i in range(len(train_data))]

        train_images = [preprocess(img) for img in train_images]
        train_images = torch.stack(train_images, dim=0).cuda()
        train_text = clip.tokenize(train_text).cuda()
        
        logits_per_image, logits_per_text = model(train_images, train_text)
        
        loss_i = loss(logits_per_image, torch.arange(BATCH_SIZE).cuda())
        loss_t = loss(logits_per_text, torch.arange(BATCH_SIZE).cuda())
        
        total_loss = (loss_i + loss_t) / 2
        total_loss.backward()
        print(total_loss)



