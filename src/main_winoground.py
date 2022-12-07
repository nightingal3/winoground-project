import clip
from clip.model import CLIP
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
import os
import wandb
import logging
from utils.losses import ContrastiveLoss
from data import get_dataset

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ViT-B/32')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--distractor_text', choices=["nounadjshuf", "nonnounadjshuf", "trigramshuf", "nounshuf", "advshuf", "adjshuf", "verbshuf"])
    parser.add_argument('--unfreeze', nargs="*", type=str, default=['11', 'ln_post', 'ln_final'])
    parser.add_argument('--distractor_image', choices=["random_patch"])
    parser.add_argument('--coco_path', type=str, default="/scratch/samuelyu/mscoco/")
    parser.add_argument('--caption_year', type=str, default="2014")
    parser.add_argument('--is_contrastive', action='store_true')
    parser.add_argument('--l1', type=float, default=0.5)
    parser.add_argument('--l2', type=float, default=0.5)
    parser.add_argument('--l3', type=float, default=0.0)
    parser.add_argument('--l4', type=float, default=0.0)
    parser.add_argument('--c', type=float, default=-0.1)
    parser.add_argument('--r1', type=float, default=0.25)
    parser.add_argument('--r2', type=float, default=0.25)
    parser.add_argument('--train_dataset', type=str, default="winoground")
    parser.add_argument('--test_dataset', type=str, default="winoground")
    parser.add_argument('--ratio', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--wd', type=float, default=1e-5)
    parser.add_argument('--betas', type=float, nargs=2, default=(0.9, 0.98))
    parser.add_argument('--eps', type=float, default=1e-7)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    using_distractors = args.distractor_text is not None or args.distractor_image is not None
    return args

def new_forward(self, image, text):
    image_features = self.encode_image(image)
    text_features = self.encode_text(text)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return image_features, text_features

CLIP.forward = new_forward

# TODO: Make a models folder and move this into there
def load_model(args):
    model, preprocess = clip.load(args.model)
    model = model.float()
    
    if torch.cuda.device_count() > 1:
        logging.info()
        model = nn.DataParallel(model)
    elif torch.cuda.is_available():
        model = model.cuda()
    else:
        logging.info("using CPU only")
    return model

def train_epoch(dataloader, model, optimizer, loss_image, loss_text, args, epoch, device=torch.device("cuda"), closs=None):
    train_loss = 0
    train_correct_image = 0
    train_correct_text = 0
    train_image_score = 0
    train_text_score = 0
    train_total = 0
    train_group_score = 0

    model.train()
    logging.info("Training for epoch {}".format(epoch))
    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()
        img0 = batch['image_0'].to(device)
        img1 = batch['image_1'].to(device)
        text0 = batch['text_0'].to(device).squeeze(1)
        text1 = batch['text_1'].to(device).squeeze(1)
        num = len(batch['image_0'])

        i0,t0 = model(img0, text0)
        i1,t1 = model(img1, text1)

        image_features = torch.cat([i0, i1], dim=0)
        text_features = torch.cat([t0, t1], dim=0)

        if args.distractor_image or args.distractor_text:
            dimg0 = batch['distractor_image_0'].to(device)
            dimg1 = batch['distractor_image_1'].to(device)
            dtext0 = batch['distractor_text_0'].to(device).squeeze(1)
            dtext1 = batch['distractor_text_1'].to(device).squeeze(1)
            di0, dt0 = model(dimg0, dtext0)
            di1, dt1 = model(dimg1, dtext1)
            image_features = torch.cat([image_features, di0, di1], dim=0)
            text_features = torch.cat([text_features, dt0, dt1], dim=0)

        logits_per_image = 100.0 * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        if args.distractor_image or args.distractor_text:
            logits_per_image = logits_per_image[:num*2]
            logits_per_text = logits_per_text[:num*2]
        
        loss_i = loss_image(logits_per_image, torch.arange(num*2).to(device))
        loss_t = loss_text(logits_per_text, torch.arange(num*2).to(device))
        
        total_loss = args.r1*loss_i + args.r2*loss_t

        if args.is_contrastive:
            total_loss += (1-args.r1-args.r2)*closs(t0, i0, t1, i1)

        total_loss.backward()
        train_loss += total_loss.item()

        temp1 = (logits_per_image.argmax(dim=1) == torch.arange(num*2).to(device))
        temp2 = (logits_per_text.argmax(dim=1) == torch.arange(num*2).to(device))
        
        train_correct_text += temp1.sum().item()
        train_correct_image += temp2.sum().item()
        train_text_score += (temp1[::2] & temp1[1::2]).sum().item()
        train_image_score += (temp2[::2] & temp2[1::2]).sum().item()
        train_group_score += (temp1[::2] & temp1[1::2] & temp2[::2] & temp2[1::2]).sum().item()

        optimizer.step()

        train_total += num*2
        if i % 100 == 99 or i == len(dataloader) - 1:
            logging.info(f"Epoch {epoch} Batch {i}, train loss: {train_loss / train_total}, train acc image: {train_correct_image / train_total}, train acc text: {train_correct_text / train_total}, train_image_score: {train_image_score / train_total*2}, train_text_score: {train_text_score / train_total*2}, train_group_score: {train_group_score / train_total*2}")
            wandb.log({"train_step": i, "train_loss": train_loss / train_total, "train_acc_image": train_correct_image / train_total, "train_acc_text": train_correct_text / train_total, "train_image_score": train_image_score / train_total*2, "train_text_score": train_text_score / train_total*2, "train_group_score": train_group_score / train_total*2})
            train_total = 0
            train_loss = 0
            train_correct_image = 0
            train_correct_text = 0
            train_image_score = 0
            train_text_score = 0

def eval(dataloader, model, loss_image, loss_text, args, epoch=0, test=False, device=torch.device("cuda"), closs=None):
    model.eval()
    val_loss = 0
    val_correct_image = 0
    val_correct_text = 0
    val_total = 0
    val_group_score = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            img0 = batch['image_0'].to(device)
            img1 = batch['image_1'].to(device)
            text0 = batch['text_0'].to(device).squeeze(1)
            text1 = batch['text_1'].to(device).squeeze(1)
            num = len(batch['image_0'])

            i0,t0 = model(img0, text0)
            i1,t1 = model(img1, text1)

            image_features = torch.cat([i0, i1], dim=0)
            text_features = torch.cat([t0, t1], dim=0)

            logits_per_image = 100.0 * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()
            
            loss_i = loss_image(logits_per_image, torch.arange(num*2).to(device))
            loss_t = loss_text(logits_per_text, torch.arange(num*2).to(device))

            total_loss = args.r1*loss_i + args.r2*loss_t
            if args.is_contrastive:
                total_loss += (1-args.r1-args.r2)*closs(t0, i0, t1, i1)
            val_loss += total_loss.item()
            val_total += 1
            temp1 = (logits_per_image.argmax(dim=1) == torch.arange(num*2).to(device))
            temp2 = (logits_per_text.argmax(dim=1) == torch.arange(num*2).to(device))
            val_correct_text += (temp1[::2] & temp1[1::2]).sum().item()
            val_correct_image += (temp2[::2] & temp2[1::2]).sum().item()
            val_group_score += (temp1[::2] & temp1[1::2] & temp2[::2] & temp2[1::2]).sum().item()
            
    name = "test" if test else "val"
    logging.info(f"Epoch {epoch}, {name} loss: {val_loss / val_total}, {name} acc image: {val_correct_image / val_total}, {name} acc text: {val_correct_text / val_total}, {name} group score: {val_group_score / val_total}")
    return val_loss, val_correct_image, val_correct_text, val_total, val_group_score

def main(args):
    train_dataset, val_dataset, test_dataset = get_dataset(args)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args)

    print("unfreezing:", args.unfreeze)
    # freeze layers
    for name, param in model.named_parameters():
        if any(n in name for n in args.unfreeze):
            param.requires_grad = True
        else:
            param.requires_grad = False

    loss_image = nn.CrossEntropyLoss()
    loss_text = nn.CrossEntropyLoss()
    closs = ContrastiveLoss(lamb1=args.l1, lamb2=args.l2, lamb3=args.l3, c=args.c)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=args.betas, eps=args.eps)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-8)
    epoch = 0
    # val_loss, val_correct_img, val_correct_text, val_total = eval(val_dataloader, model, loss_image, loss_text, args, closs=closs)
    # wandb.log({"epoch": -1, "val_loss": val_loss / val_total, "val_acc_img": val_correct_img / val_total, "val_acc_text": val_correct_text / val_total})
    test_loss, test_correct_img, test_correct_text, test_total, test_group_score = eval(test_dataloader, model, loss_image, loss_text, args, epoch, test=True, closs=closs)
    wandb.log({"epoch": -1, "test_loss": test_loss / test_total, "test_acc_img": test_correct_img / test_total, "test_acc_text": test_correct_text / test_total, "test_group_score": test_group_score / test_total})
    logging.info("Starting training")
    for epoch in range(args.epochs):
        print("LR: ", optimizer.param_groups[0]['lr'])
        train_epoch(train_dataloader, model, optimizer, loss_image, loss_text, args, epoch, closs=closs)
        scheduler.step()
        # val_loss, val_correct_img, val_correct_text, val_total = eval(val_dataloader, model, loss_image, loss_text, args, epoch, closs=closs)
        # wandb.log({"epoch": epoch, "val_loss": val_loss / val_total, "val_acc_img": val_correct_img / val_total, "val_acc_text": val_correct_text / val_total})
        
        # os.makedirs("../save", exist_ok=True)
        # torch.save(model.state_dict(), f"../save/{epoch}.pt")
        test_loss, test_correct_img, test_correct_text, test_total, test_group_score = eval(test_dataloader, model, loss_image, loss_text, args, epoch, test=True, device=device, closs=closs)
        wandb.log({"epoch": epoch, "test_loss": test_loss / test_total, "test_acc_img": test_correct_img / test_total, "test_acc_text": test_correct_text / test_total, "test_group_score": test_group_score / test_total})

if __name__ == "__main__":
    args = get_args()
    logging.basicConfig(level=logging.INFO)
    wandb.init(project="winoground-pretraining")
    use_distractors = args.distractor_text is not None and args.distractor_image is not None
    wandb.config.update(args)
    wandb.run.name = f"lr_{args.lr}_wd_{args.wd}_bs_{args.batch_size}_epochs_{args.epochs}_distractors_{use_distractors}_t_{args.distractor_text}_i_{args.distractor_image}_contrastive_{args.is_contrastive}"

    main(args)
    


