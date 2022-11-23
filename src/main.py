import clip
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms
import numpy as np
import argparse

from data import get_dataset
import pdb

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ViT-B/32')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--use_distractors', action='store_true')
    parser.add_argument('--coco_path', type=str, default="/projects/tir1/corpora/COCO/")
    parser.add_argument('--caption_year', type=str, default="2017")
    parser.add_argument('--train_dataset', type=str, default="coco")
    parser.add_argument('--test_dataset', type=str, default="winoground")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    return args

# TODO: Make a models folder and move this into there
def load_model(args):
    model, preprocess = clip.load(args.model)
    model = model.float().cuda()
    return model

# TODO: separate into train/test functions
def main(args):
    train_dataset, val_dataset, test_dataset = get_dataset(args)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = load_model(args)

    # freeze layers
    for name, param in model.named_parameters():
        if "11" in name or "ln_post" in name or "ln_final" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    loss_image = nn.CrossEntropyLoss()
    loss_text = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    for epoch in range(args.epochs):
        train_loss = 0
        train_correct_image = 0
        train_correct_text = 0
        train_total = 0
        model.train()
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            image = batch['image'].cuda()
            text = batch['text'].cuda().squeeze(1)

            if args.use_distractors:
                image = torch.cat((image, batch['distractor_image'].cuda()), dim=0)
                distractor_text = batch['distractor_text'].cuda().squeeze(1)
                text = torch.cat((text, distractor_text), dim=0)
        
            logits_per_image, logits_per_text = model(image, text)
            
            if args.use_distractors: # distractors have no corresponding label
                logits_per_image = logits_per_image[:args.batch_size, :args.batch_size]
                logits_per_text = logits_per_text[:args.batch_size, :args.batch_size]

            loss_i = loss_image(logits_per_image, torch.arange(args.batch_size).cuda())
            loss_t = loss_text(logits_per_text, torch.arange(args.batch_size).cuda())

            total_loss = (loss_i + loss_t) / 2
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()
            train_correct_image += (logits_per_image.argmax(dim=1) == torch.arange(args.batch_size).cuda()).sum().item()
            train_correct_text += (logits_per_text.argmax(dim=1) == torch.arange(args.batch_size).cuda()).sum().item()
            train_total += args.batch_size

            if i % 100 == 0:
                print(f"Epoch {epoch} Batch {i}, train loss: {train_loss / train_total}, train acc image: {train_correct_image / train_total}, train acc text: {train_correct_text / train_total}")
                train_total = 0
                train_loss = 0
                train_correct_image = 0
                train_correct_text = 0
        
        model.eval()
        val_loss = 0
        val_correct_image = 0
        val_correct_text = 0
        val_total = 0
        with torch.no_grad():
            for i, batch in enumerate(val_dataloader):
                image = batch['image'].cuda()
                text = batch['text'].cuda()

                logits_per_image, logits_per_text = model(image, text)
                loss_i = loss_image(logits_per_image, torch.arange(args.batch_size).cuda())
                loss_t = loss_text(logits_per_text, torch.arange(args.batch_size).cuda())

                total_loss = (loss_i + loss_t) / 2
                val_loss += total_loss.item()
                val_total += len(batch)
                val_correct_image += (logits_per_image.argmax(dim=1) == torch.arange(args.batch_size).cuda()).sum().item()
                val_correct_text += (logits_per_text.argmax(dim=1) == torch.arange(args.batch_size).cuda()).sum().item()
        print(f"Epoch {epoch}, val loss: {val_loss / val_total}, val acc image: {val_correct_image / val_total}, val acc text: {val_correct_text / val_total}")

        torch.save(model.state_dict(), f"../save/{epoch}.pt")

    model.eval()
    test_loss = 0
    test_correct_image = 0
    test_correct_text = 0
    test_total = 0
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            image = batch['image'].cuda()
            text = batch['text'].cuda()

            logits_per_image, logits_per_text = model(image, text)
            loss_i = loss_image(logits_per_image, torch.arange(args.batch_size).cuda())
            loss_t = loss_text(logits_per_text, torch.arange(args.batch_size).cuda())

            total_loss = (loss_i + loss_t) / 2
            test_loss += total_loss.item()
            test_total += len(batch)
            test_correct_image += (logits_per_image.argmax(dim=1) == torch.arange(args.batch_size).cuda()).sum().item()
            test_correct_text += (logits_per_text.argmax(dim=1) == torch.arange(args.batch_size).cuda()).sum().item()
    print(f"Test loss: {test_loss / test_total}, test acc image: {test_correct_image / test_total}, test acc text: {test_correct_text / test_total}")

if __name__ == "__main__":
    
    args = get_args()
    main(args)
    


