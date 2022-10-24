import argparse
import os
import pandas as pd
import logging
from tqdm import tqdm
from pathlib import Path
import numpy as np
from datasets import load_dataset
from transformers import (
    CLIPProcessor,
    CLIPModel,
    BertTokenizer,
    BertModel,
    RobertaTokenizer,
    RobertaModel,
    OpenAIGPTTokenizer,
    OpenAIGPTLMHeadModel,
)
import torch
import pdb

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def model_init(model_string: str, fast=False):
    if model_string == "clip":
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    elif model_string == "bert":
        processor = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained(
            "bert-base-uncased", output_hidden_states=True
        )
    elif model_string == "roberta":
        processor = RobertaTokenizer.from_pretrained("roberta-base")
        model = RobertaModel.from_pretrained("roberta-base", output_hidden_states=True)
    else:
        processor = OpenAIGPTTokenizer.from_pretrained(model_string)
        model = OpenAIGPTLMHeadModel.from_pretrained(model_string)

    model.eval()
    model.to(DEVICE)

    return model, processor

def get_one_vec(phrase, model, tokenizer, emb_type: str = "CLS", model_type: str = "encoder", layer: int = 12):
    if model_type == "gpt":
        phrase = phrase + " " + tokenizer.eos_token

    encoded_phrase = tokenizer.encode(phrase, return_tensors="pt")
    if DEVICE == "cuda":
        encoded_phrase = encoded_phrase.to("cuda")

    outputs = model(encoded_phrase)
    if emb_type == "CLS":
        if layer == 12:
            if model_type == "encoder":
                emb_phrase = outputs[0][:, 0, :]
            else:
                emb_phrase = outputs[0][:, -1, :]
        else:
            hidden = outputs["hidden_states"][layer]
            if model_type == "encoder":
                emb_phrase = hidden[:, 0, :]
            else:
                emb_phrase = hidden[:, -1, :]

    elif emb_type == "avg":
        if layer == 12:
            token_embeddings = outputs["last_hidden_state"]
            emb_phrase = token_embeddings.squeeze(0).mean(dim=0)
        else:
            token_embeddings = outputs["hidden_states"][layer]
            emb_phrase = token_embeddings.squeeze(0).mean(dim=0)


    return emb_phrase

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract winoground embeddings and data for analysis"
    )
    parser.add_argument(
        "-m",
        "--model",
        help="model to extract embs from",
        choices=["clip", "roberta"],
        required=True,
    )
    parser.add_argument(
        "-l",
        "--layer",
        help="layer to extract embs from (doesn't apply to all models)",
        default=12,
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        help="directory to output embeddings and winoground info to",
        default="./data",
    )
    parser.add_argument(
        "--normalize", help="normalize the embeddings", action="store_true"
    )
    args = parser.parse_args()
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

    winoground_hf = load_dataset("facebook/winoground", use_auth_token=True)["test"]
    winoground_lang_hf = winoground_hf.remove_columns(["image_0", "image_1"])

    lang_csv_output_path = os.path.join(args.output_dir, "winoground_langdata.csv")
    emb_output_dir = os.path.join(args.output_dir, f"{args.model}_embs/")
    img_output_dir = os.path.join(args.output_dir, "imgs/")
    
    if not os.path.exists(lang_csv_output_path):
        logging.info(
            f"Summary csv for language doesn't exist yet, writing it to {lang_csv_output_path}"
        )
        df_lang = winoground_lang_hf.to_pandas()
        df_lang.to_csv(lang_csv_output_path, index=False)
    
    if not os.path.exists(img_output_dir):
        logging.info(
            f"Image embeddings don't exist yet, writing them to {img_output_dir}"
        )
        os.makedirs(img_output_dir)
        for i, example in enumerate(tqdm(winoground_hf)):
            img_0 = example["image_0"]
            img_1 = example["image_1"]
            img_0.save(os.path.join(img_output_dir, f"{i}_0.png"))
            img_1.save(os.path.join(img_output_dir, f"{i}_1.png"))

    if not os.path.isdir(emb_output_dir) or len(os.listdir(emb_output_dir)) == 0:
        Path(emb_output_dir).mkdir(parents=False, exist_ok=True)
        logging.info(f"Embeddings not produced yet, writing them to {emb_output_dir}")

        logging.info(f"Loading model...")
        model, processor = model_init(args.model)
        logging.info("Model loaded")

        # TODO: also support RoBERTa
        all_image_embs = []
        all_text_embs = []
        for i, sample in enumerate(tqdm(winoground_hf)):
            if args.model == "clip":
                input_c0_i0 = processor(
                    text=[sample["caption_0"]],
                    images=[sample["image_0"].convert("RGB")],
                    return_tensors="pt",
                )
                input_i0 = processor(
                    text=None,
                    images=[sample["image_0"].convert("RGB")],
                    return_tensors="pt",
                )["pixel_values"].to(DEVICE)
                input_i1 = processor(
                    text=None,
                    images=[sample["image_1"].convert("RGB")],
                    return_tensors="pt",
                )["pixel_values"].to(DEVICE)
                input_c0 = processor(
                    text=[sample["caption_0"]], images=None, return_tensors="pt"
                ).to(DEVICE)
                input_c1 = processor(
                    text=[sample["caption_1"]], images=None, return_tensors="pt"
                ).to(DEVICE)
                if DEVICE == "cuda":
                    for key in input_c0_i0:
                        input_c0_i0[key] = input_c0_i0[key].cuda()
                output_i0 = model.get_image_features(input_i0)
                output_i1 = model.get_image_features(input_i1)
                output_c0 = model.get_text_features(**input_c0)
                output_c1 = model.get_text_features(**input_c1)
                if args.normalize:
                    output_i0 /= output_i0.norm(dim=-1, keepdim=True)
                    output_i1 /= output_i1.norm(dim=-1, keepdim=True)
                    output_c0 /= output_c0.norm(dim=-1, keepdim=True)
                    output_c1 /= output_c1.norm(dim=-1, keepdim=True)

                all_image_embs.append(torch.stack([output_i0, output_i1]).transpose(0, 1).cpu().detach().numpy())
                all_text_embs.append(torch.stack([output_c0, output_c1]).transpose(0, 1).cpu().detach().numpy())

            elif args.model == "roberta":
                input_c0 = sample["caption_0"]
                input_c1 = sample["caption_1"]
                c0_emb = get_one_vec(input_c0, model, processor, layer=args.layer)
                c1_emb = get_one_vec(input_c1, model, processor, layer=args.layer)
                all_text_embs.append(torch.stack([c0_emb, c1_emb]).transpose(0, 1).cpu().detach().numpy())

        all_text_embs = np.concatenate(all_text_embs, axis=0)

        if args.model == "clip":
            all_image_embs = np.concatenate(all_image_embs, axis=0)

        # Will be saved as (N x 2 x 512) tensor
        with open(f"{emb_output_dir}/text.npy", "wb") as f:
            np.save(f, all_text_embs)
        if args.model == "clip":
            with open(f"{emb_output_dir}/image.npy", "wb") as f:
                np.save(f, all_image_embs)
        logging.info(f"Saved {args.model} embeddings")
