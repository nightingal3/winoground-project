import clip
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms
import numpy as np
import argparse

from data.mscoco import COCO, COCODataset
from VLMOrder.perturbtextorder import nounadjshuf
from VLMOrder.perturbimage import RandomPatch
import pdb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ViT-B/32')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--use_distractors', action='store_true')
    parser.add_argument('--coco_path', type=str, default="/projects/tir1/corpora/COCO/")
    parser.add_argument('--caption_year', type=str, default="2017")
    args = parser.parse_args()

    dataset = COCODataset(root=args.coco_path, caption_year=args.caption_year, image_transforms=None)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model, preprocess = clip.load(args.model)
    model = model.float().cuda()

    loss_image = nn.CrossEntropyLoss()
    loss_text = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters()) 
    
    for i in range(1000):
        optimizer.zero_grad()
        inds = np.random.choice(len(dataset), args.batch_size)
        train_data = [dataset[i] for i in inds]
        #train_images = [train_data['image_0'] + train_data['image_1']]
        train_images = [train_data[i]["image"] for i in range(len(train_data))]
        train_text = [train_data[i]["caption"] for i in range(len(train_data))]

        if args.use_distractors: # TODO: more sophisticated distractors
            distractor_text = [nounadjshuf(train_data[i]["caption"]) for i in range(len(train_data))]
            train_text = train_text + distractor_text

            distractor_images = [RandomPatch(train_data[i]["image"], 4) for i in range(len(train_data))]
            train_images = train_images + distractor_images

        train_images = [preprocess(img) for img in train_images]
        train_images = torch.stack(train_images, dim=0).cuda()
        train_text = clip.tokenize(train_text).cuda()
        
        logits_per_image, logits_per_text = model(train_images, train_text)
        
        if args.use_distractors: # distractors have no corresponding label
            logits_per_image = logits_per_image[:args.batch_size, :args.batch_size]
            logits_per_text = logits_per_text[:args.batch_size, :args.batch_size]

        loss_i = loss_image(logits_per_image, torch.arange(args.batch_size).cuda())
        loss_t = loss_text(logits_per_text, torch.arange(args.batch_size).cuda())

        total_loss = (loss_i + loss_t) / 2
        total_loss.backward()
        print(total_loss)



