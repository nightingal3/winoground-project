from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import torch
from perturbtextorder import *
from perturbimage import *
from PIL import Image

# slightly modified version of CLIP_Implementation.ipynb that writes out the relevant information to a file
# that we can use for later analysis

auth_token = "" # use your own
winoground = load_dataset("facebook/winoground", use_auth_token=auth_token)["test"]

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = clip_model.to(device)
print("Original Scores")
# Functions to calculate Text, Image and Group Scores

def image_correct(result):
    return result["c0_i"] > result["c1_i"] 


#Calculate the image-caption score for all examples in winoground
winoground_clip_scores = []
for example in tqdm(winoground):
  input_c0_i0 = clip_processor(text=[example["caption_0"]], images=[example["image_0"].convert("RGB")], return_tensors="pt").to(device)
  input_c1_i0 = clip_processor(text=[example["caption_1"]], images=[example["image_0"].convert("RGB")], return_tensors="pt").to(device)
  input_c0_i1 = clip_processor(text=[example["caption_0"]], images=[example["image_1"].convert("RGB")], return_tensors="pt").to(device)
  input_c1_i1 = clip_processor(text=[example["caption_1"]], images=[example["image_1"].convert("RGB")], return_tensors="pt").to(device)
  output_c0_i0 = clip_model(**input_c0_i0)
  output_c1_i0 = clip_model(**input_c1_i0)
  output_c0_i1 = clip_model(**input_c0_i1)
  output_c1_i1 = clip_model(**input_c1_i1)
  clip_score_c0_i0 = output_c0_i0.logits_per_image.item()
  clip_score_c1_i0 = output_c1_i0.logits_per_image.item()
  clip_score_c0_i1 = output_c0_i1.logits_per_image.item()
  clip_score_c1_i1 = output_c1_i1.logits_per_image.item()
  winoground_clip_scores.append({"id" : example["id"], "c0_i": clip_score_c0_i0, "c1_i": clip_score_c1_i0})
  winoground_clip_scores.append({"id" : example["id"], "c0_i": clip_score_c1_i1, "c1_i": clip_score_c0_i1})


image_correct_count = 0
for result in winoground_clip_scores:
  image_correct_count += 1 if image_correct(result) else 0

denominator = len(winoground_clip_scores)
print("image score:", image_correct_count/denominator)

#Functions to calculate Text, Image and Group Scores
def text_correct_pert(result):
    return result["c_i"] > result["cpert_i"]

def image_correct_pert(result):
    return result["c_i"] > result["c_ipert"]


winoground_clip_scores = []
print("Image Patch Shuffling")
for example in tqdm(winoground):
  text0 = example["caption_0"]
  text1 = example["caption_1"]
  # If the perturbation return the same text, don't use it.
  # Becuase Winoground is symmetrical in captions with just one different word, if one text can't be perturbed then the second one can't be either
  input_c0_i0 = clip_processor(text=[text0], images=[example["image_0"].convert("RGB")], return_tensors="pt").to(device)
  input_c0_i0pert = clip_processor(text=[text0], images=[RandomPatch(example["image_0"].convert("RGB"),7*7)], return_tensors="pt").to(device)
  input_c1_i1 = clip_processor(text=[text1], images=[example["image_1"].convert("RGB")], return_tensors="pt").to(device)
  input_c1_i1pert = clip_processor(text=[text1], images=[RandomPatch(example["image_1"].convert("RGB"),7*7)], return_tensors="pt").to(device)
  output_c0_i0 = clip_model(**input_c0_i0)
  output_c0_i0pert = clip_model(**input_c0_i0pert)
  output_c1_i1 = clip_model(**input_c1_i1)
  output_c1_i1pert = clip_model(**input_c1_i1pert)
  clip_score_c0_i0 = output_c0_i0.logits_per_image.item()
  clip_score_c0_i0pert = output_c0_i0pert.logits_per_image.item()
  clip_score_c1_i1pert = output_c1_i1pert.logits_per_image.item()
  clip_score_c1_i1 = output_c1_i1.logits_per_image.item()
  winoground_clip_scores.append({"id" : example["id"], "c_i": clip_score_c0_i0, "c_ipert": clip_score_c0_i0pert})
  winoground_clip_scores.append({"id" : example["id"], "c_i": clip_score_c1_i1, "c_ipert": clip_score_c1_i1pert})

image_correct_count = 0
for result in winoground_clip_scores:
  image_correct_count += 1 if image_correct_pert(result) else 0

denominator = len(winoground_clip_scores)
print("Image Score:", image_correct_count/denominator)

