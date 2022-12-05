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
    parser.add_argument('--distractor_image', choices=["random_patch"])
    parser.add_argument('--coco_path', type=str, default="/scratch/samuelyu/mscoco/")
    parser.add_argument('--caption_year', type=str, default="2014")
    parser.add_argument('--is_contrastive', action='store_true')
    parser.add_argument('--l1', type=float, default=0.5)
    parser.add_argument('--l2', type=float, default=0.5)
    parser.add_argument('--l3', type=float, default=0.0)
    parser.add_argument('--l4', type=float, default=0.0)
    parser.add_argument('--c', type=float, default=-0.1)
    parser.add_argument('--train_dataset', type=str, default="coco")
    parser.add_argument('--test_dataset', type=str, default="winoground")
    parser.add_argument('--ratio', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--wd', type=float, default=0.2)
    parser.add_argument('--betas', type=float, nargs=2, default=(0.9, 0.98))
    parser.add_argument('--eps', type=float, default=1e-7)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    using_distractors = args.distractor_text is not None or args.distractor_image is not None
    if args.is_contrastive and not using_distractors:
        raise Exception("use_distractors must be True if is_contrastive is True")

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

def train_epoch(dataloader, model, optimizer, loss_image, loss_text, args, epoch, device=torch.device("cuda")):
    train_loss = 0
    train_correct_image = 0
    train_correct_text = 0
    train_total = 0
    model.train()
    if args.is_contrastive:
        print("Using contrastive loss")
        closs = ContrastiveLoss(lamb1=args.l1, lamb2=args.l2, lamb3=args.l3, c=args.c)  # default params
    logging.info("Training for epoch {}".format(epoch))
    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()
        image = batch['image'].to(device)
        text = batch['text'].to(device).squeeze(1)
        num_image = len(batch['image'])
        if args.distractor_text or args.distractor_image:
            image = torch.cat((image, batch['distractor_image'].to(device)), dim=0)
            distractor_text = batch['distractor_text'].to(device).squeeze(1)
            text = torch.cat((text, distractor_text), dim=0)

        image_features, text_features = model(image, text)

        if not args.is_contrastive:
            logits_per_image = 100.0 * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()
            
            if args.distractor_text or args.distractor_image: # distractors have no corresponding label
                logits_per_image = logits_per_image[:len(batch['image']), :len(batch['image'])]
                logits_per_text = logits_per_text[:len(batch['image']), :len(batch['image'])]

            loss_i = loss_image(logits_per_image, torch.arange(len(batch['image'])).to(device))
            loss_t = loss_text(logits_per_text, torch.arange(len(batch['image'])).to(device))

            total_loss = (loss_i + loss_t) / 2
            total_loss.backward()
            train_loss += total_loss.item()
        else:
            
            # since use_distractors must be true, 0-num_image are original images and num_image-end are distractors
            i0 = image_features[:num_image]
            i1 = image_features[num_image:]
            c0 = text_features[:num_image]
            c1 = text_features[num_image:]

            total_loss = closs(c0, i0, c1, i1)
            total_loss.backward()
            train_loss += total_loss.item()

            logit_scale = 100.0
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()
            logits_per_image = logits_per_image[:len(batch['image']), :len(batch['image'])]
            logits_per_text = logits_per_text[:len(batch['image']), :len(batch['image'])]

        train_correct_text += (logits_per_image.argmax(dim=1) == torch.arange(len(batch['image'])).to(device)).sum().item()
        train_correct_image += (logits_per_text.argmax(dim=1) == torch.arange(len(batch['image'])).to(device)).sum().item()
        optimizer.step()

        train_total += len(batch['image'])
        if i % 100 == 0:
            logging.info(f"Epoch {epoch} Batch {i}, train loss: {train_loss / train_total}, train acc image: {train_correct_image / train_total}, train acc text: {train_correct_text / train_total}")
            wandb.log({"train_step": i, "train_loss": train_loss / train_total, "train_acc_image": train_correct_image / train_total, "train_acc_text": train_correct_text / train_total})
            train_total = 0
            train_loss = 0
            train_correct_image = 0
            train_correct_text = 0

def eval(dataloader, model, loss_image, loss_text, args, epoch=0, test=False, device=torch.device("cuda")):
    model.eval()
    val_loss = 0
    val_correct_image = 0
    val_correct_text = 0
    val_total = 0
    # logging.info(f"Evaluation for epoch {epoch}, on test? {test}")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            image = batch['image'].to(device)
            text = batch['text'].to(device).squeeze(1)

            image_features, text_features = model(image, text)
            logits_per_image = 100.0 * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()
            
            loss_i = loss_image(logits_per_image, torch.arange(len(batch['image'])).to(device))
            loss_t = loss_text(logits_per_text, torch.arange(len(batch['image'])).to(device))

            total_loss = (loss_i + loss_t) / 2
            val_loss += total_loss.item()
            val_total += len(batch['image'])

            val_correct_text += ((logits_per_image.argmax(dim=1) == torch.arange(len(batch['image'])).to(device)).sum().item() == len(batch['image']))*len(batch['image'])
            val_correct_image += ((logits_per_text.argmax(dim=1) == torch.arange(len(batch['image'])).to(device)).sum().item() == len(batch['image']))*len(batch['image'])
            
    name = "test" if test else "val"
    logging.info(f"Epoch {epoch}, {name} loss: {val_loss / val_total}, {name} acc image: {val_correct_image / val_total}, {name} acc text: {val_correct_text / val_total}")
    return val_loss, val_correct_image, val_correct_text, val_total

def main(args):
    train_dataset, val_dataset, test_dataset = get_dataset(args)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.train_dataset != "winoground")
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args)

    # freeze layers
    for name, param in model.named_parameters():
        if "11" in name or "ln_post" in name or "ln_final" in name:
        # if "ln_post" in name or "ln_final" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    loss_image = nn.CrossEntropyLoss()
    loss_text = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=args.betas, eps=args.eps)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-8)
    epoch = 0
    # val_loss, val_correct_img, val_correct_text, val_total = eval(val_dataloader, model, loss_image, loss_text, args)
    # wandb.log({"epoch": -1, "val_loss": val_loss / val_total, "val_acc_img": val_correct_img / val_total, "val_acc_text": val_correct_text / val_total})
    # test_loss, test_correct_img, test_correct_text, test_total = eval(test_dataloader, model, loss_image, loss_text, args, epoch, test=True)
    # wandb.log({"epoch": -1, "test_loss": test_loss / test_total, "test_acc_img": test_correct_img / test_total, "test_acc_text": test_correct_text / test_total})
    logging.info("Starting training")
    for epoch in range(args.epochs):
        print("LR: ", optimizer.param_groups[0]['lr'])
        train_epoch(train_dataloader, model, optimizer, loss_image, loss_text, args, epoch)
        scheduler.step()
        val_loss, val_correct_img, val_correct_text, val_total = eval(val_dataloader, model, loss_image, loss_text, args, epoch)
        wandb.log({"epoch": epoch, "val_loss": val_loss / val_total, "val_acc_img": val_correct_img / val_total, "val_acc_text": val_correct_text / val_total})
        os.makedirs("../save", exist_ok=True)
        torch.save(model.state_dict(), f"../save/{epoch}.pt")
        test_loss, test_correct_img, test_correct_text, test_total = eval(test_dataloader, model, loss_image, loss_text, args, epoch, test=True, device=device)
        wandb.log({"epoch": epoch, "test_loss": test_loss / test_total, "test_acc_img": test_correct_img / test_total, "test_acc_text": test_correct_text / test_total})

if __name__ == "__main__":
    args = get_args()
    logging.basicConfig(level=logging.INFO)
    wandb.init(project="winoground-pretraining")
    use_distractors = args.distractor_text is not None and args.distractor_image is not None
    wandb.config.update(args)
    wandb.run.name = f"lr_{args.lr}_wd_{args.wd}_bs_{args.batch_size}_epochs_{args.epochs}_distractors_{use_distractors}_t_{args.distractor_text}_i_{args.distractor_image}_contrastive_{args.is_contrastive}"

    main(args)
    


