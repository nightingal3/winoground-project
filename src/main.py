import clip
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np

from data.mscoco import COCO, COCODataset

import pdb

COCO_PATH="/projects/tir1/corpora/COCO/"

if __name__ == "__main__":
    dataset = COCODataset(root=COCO_PATH, caption_year="2017")
    train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model, preprocess = clip.load("ViT-B/32")
    model = model.float().cuda()

    loss = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters()) 

    for i in range(1000):
        optimizer.zero_grad()
        train_data = dataset[np.random.choice(400, size=32, replace=False)]
        train_images = [train_data['image_0'] + train_data['image_1']]
        train_images = [preprocess(img) for img in train_images[0]]
        train_images = torch.stack(train_images, dim=0).cuda()
        train_text = clip.tokenize(train_data['caption_0'] + train_data['caption_1']).cuda()
        
        logits_per_image, logits_per_text = model(train_images, train_text)
        
        loss_i = loss(logits_per_image, torch.arange(64).cuda())
        loss_t = loss(logits_per_text, torch.arange(64).cuda())
        
        total_loss = (loss_i + loss_t) / 2
        total_loss.backward()
        print(total_loss)



