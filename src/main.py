import clip
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
import os
import wandb
import logging
import pdb

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
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--wd', type=float, default=0.2)
    parser.add_argument('--betas', type=float, nargs=2, default=(0.9, 0.98))
    parser.add_argument('--eps', type=float, default=1e-7)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    return args

# TODO: Make a models folder and move this into there
def load_model(args):
    model, preprocess = clip.load(args.model)
    model = model.float()
    if torch.cuda.device_count() > 1:
        logging.info()
        model = nn.DataParallel(model)
    elif torch.cuda.is_available():
        model = model.cuda()
    
    return model

def train_epoch(dataloader, model, optimizer, loss_image, loss_text, args, epoch):
    train_loss = 0
    train_correct_image = 0
    train_correct_text = 0
    train_total = 0
    model.train()
    logging.info("Training for epoch {}".format(epoch))
    for i, batch in enumerate(dataloader):
        #if i > 10:
            #break
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
        train_correct_text += (logits_per_image.argmax(dim=1) == torch.arange(len(batch['image'])).cuda()).sum().item()
        train_correct_image += (logits_per_text.argmax(dim=1) == torch.arange(len(batch['image'])).cuda()).sum().item()
        train_total += len(batch['image'])
        if i % 100 == 0:
            logging.info(f"Epoch {epoch} Batch {i}, train loss: {train_loss / train_total}, train acc image: {train_correct_image / train_total}, train acc text: {train_correct_text / train_total}")
            wandb.log({"train_step": i, "train_loss": train_loss / train_total, "train_acc_image": train_correct_image / train_total, "train_acc_text": train_correct_text / train_total})
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
    logging.info(f"Evaluation for epoch {epoch}, on test? {test}")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            image = batch['image'].cuda()
            text = batch['text'].cuda().squeeze(1)
            #pdb.set_trace()

            logits_per_image, logits_per_text = model(image, text)
            loss_i = loss_image(logits_per_image, torch.arange(len(batch['image'])).cuda())
            loss_t = loss_text(logits_per_text, torch.arange(len(batch['image'])).cuda())

            total_loss = (loss_i + loss_t) / 2
            val_loss += total_loss.item()
            val_total += len(batch['image'])

            val_correct_text += ((logits_per_image.argmax(dim=1) == torch.arange(len(batch['image'])).cuda()).sum().item() == len(batch['image']))*len(batch['image'])
            val_correct_image += ((logits_per_text.argmax(dim=1) == torch.arange(len(batch['image'])).cuda()).sum().item() == len(batch['image']))*len(batch['image'])
            
    name = "test" if test else "val"
    logging.info(f"Epoch {epoch}, {name} loss: {val_loss / val_total}, {name} acc image: {val_correct_image / val_total}, {name} acc text: {val_correct_text / val_total}")
    return val_loss, val_correct_image, val_correct_text, val_total

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
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=args.betas, eps=args.eps)
    epoch = 0
    eval(val_dataloader, model, loss_image, loss_text, args)
    eval(test_dataloader, model, loss_image, loss_text, args, epoch, test=True)
    logging.info("Starting training")
    for epoch in range(args.epochs):
        train_epoch(train_dataloader, model, optimizer, loss_image, loss_text, args, epoch)
        val_loss, val_correct_img, val_correct_text, val_total = eval(val_dataloader, model, loss_image, loss_text, args, epoch)
        wandb.log({"epoch": epoch, "val_loss": val_loss / val_total, "val_acc_img": val_correct_img / val_total, "val_acc_text": val_correct_text / val_total})
        os.makedirs("../save", exist_ok=True)
        torch.save(model.state_dict(), f"../save/{epoch}.pt")
        test_loss, test_correct_img, test_correct_text, test_total = eval(test_dataloader, model, loss_image, loss_text, args, epoch, test=True)
        wandb.log({"epoch": epoch, "test_loss": test_loss / test_total, "test_acc_img": test_correct_img / test_total, "test_acc_text": test_correct_text / test_total})

if __name__ == "__main__":
    args = get_args()
    logging.basicConfig(level=logging.INFO)
    wandb.init(project="winoground-pretraining")
    wandb.config.lr = args.lr
    wandb.config.wd = args.wd
    wandb.config.batch_size = args.batch_size
    wandb.config.epochs = args.epochs
    wandb.config.use_distractors = args.use_distractors
    wandb.run.name = f"lr_{args.lr}_wd_{args.wd}_bs_{args.batch_size}_epochs_{args.epochs}_distractors_{args.use_distractors}"

    main(args)
    


