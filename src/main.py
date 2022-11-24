import clip
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
import os

from data import get_dataset

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

def train_epoch(dataloader, model, optimizer, loss_image, loss_text, args, epoch):
    train_loss = 0
    train_correct_image = 0
    train_correct_text = 0
    train_total = 0
    model.train()
    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()
        image = batch['image'].cuda()
        text = batch['text'].cuda().squeeze(1)

        if args.use_distractors:
            image = torch.cat((image, batch['distractor_image'].cuda()), dim=0)
            distractor_text = batch['distractor_text'].cuda().squeeze(1)
            text = torch.cat((text, distractor_text), dim=0)
    
        logits_per_image, logits_per_text = model(image, text)
        
        if args.use_distractors: # distractors have no corresponding label
            logits_per_image = logits_per_image[:len(batch['image']), :len(batch['image'])]
            logits_per_text = logits_per_text[:len(batch['image']), :len(batch['image'])]

        loss_i = loss_image(logits_per_image, torch.arange(len(batch['image'])).cuda())
        loss_t = loss_text(logits_per_text, torch.arange(len(batch['image'])).cuda())

        total_loss = (loss_i + loss_t) / 2
        total_loss.backward()
        optimizer.step()

        train_loss += total_loss.item()
        train_correct_image += (logits_per_image.argmax(dim=1) == torch.arange(len(batch['image'])).cuda()).sum().item()
        train_correct_text += (logits_per_text.argmax(dim=1) == torch.arange(len(batch['image'])).cuda()).sum().item()
        train_total += len(batch['image'])
        if i % 100 == 0:
            print(f"Epoch {epoch} Batch {i}, train loss: {train_loss / train_total}, train acc image: {train_correct_image / train_total}, train acc text: {train_correct_text / train_total}")
            train_total = 0
            train_loss = 0
            train_correct_image = 0
            train_correct_text = 0

def eval(dataloader, model, loss_image, loss_text, args, epoch=0, test=False):
    model.eval()
    val_loss = 0
    val_correct_image = 0
    val_correct_text = 0
    val_total = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            image = batch['image'].cuda()
            text = batch['text'].cuda().squeeze(1)

            logits_per_image, logits_per_text = model(image, text)
            loss_i = loss_image(logits_per_image, torch.arange(len(batch['image'])).cuda())
            loss_t = loss_text(logits_per_text, torch.arange(len(batch['image'])).cuda())

            total_loss = (loss_i + loss_t) / 2
            val_loss += total_loss.item()
            val_total += len(batch['image'])

            val_correct_image += ((logits_per_image.argmax(dim=1) == torch.arange(len(batch['image'])).cuda()).sum().item() == len(batch['image']))*len(batch['image'])
            val_correct_text += ((logits_per_text.argmax(dim=1) == torch.arange(len(batch['image'])).cuda()).sum().item() == len(batch['image']))*len(batch['image'])
            
    name = "test" if test else "val"
    print(f"Epoch {epoch}, {name} loss: {val_loss / val_total}, {name} acc image: {val_correct_image / val_total}, {name} acc text: {val_correct_text / val_total}")

def main(args):
    train_dataset, val_dataset, test_dataset = get_dataset(args)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)

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
    epoch = 0
    eval(val_dataloader, model, loss_image, loss_text, args)
    eval(test_dataloader, model, loss_image, loss_text, args, epoch, test=True)
    for epoch in range(args.epochs):
        train_epoch(train_dataloader, model, optimizer, loss_image, loss_text, args, epoch)
        eval(val_dataloader, model, loss_image, loss_text, args, epoch)
        os.makedirs("../save", exist_ok=True)
        torch.save(model.state_dict(), f"../save/{epoch}.pt")
        eval(test_dataloader, model, loss_image, loss_text, args, epoch, test=True)

if __name__ == "__main__":
    
    args = get_args()
    main(args)
    


