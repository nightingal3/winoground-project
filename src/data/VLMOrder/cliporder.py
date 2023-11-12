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
def text_correct(result):
    return result["c_i0"] > result["c_i1"]

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
  winoground_clip_scores.append({"id" : example["id"], "c_i0": clip_score_c0_i0, "c_i1": clip_score_c0_i1})
  winoground_clip_scores.append({"id" : example["id"], "c_i0": clip_score_c1_i1, "c_i1": clip_score_c1_i0})

#Calculate the Text, Image and Group Scores from Winoground's image-caption score
text_correct_count = 0
for result in winoground_clip_scores:
  text_correct_count += 1 if text_correct(result) else 0

denominator = len(winoground_clip_scores)
print("text score:", text_correct_count/denominator)


#Functions to calculate Text, Image and Group Scores
def text_correct_pert(result):
    return result["c_i"] > result["cpert_i"]

def image_correct_pert(result):
    return result["c_i"] > result["c_ipert"]


winoground_clip_scores = []
print("Noun Shuffling")
for example in tqdm(winoground):
  text0 = example["caption_0"]
  text1 = example["caption_1"]
  text0pert = nounshuf(text0)
  text1pert = nounshuf(text1)
  # If the perturbation return the same text, don't use it.
  # Becuase Winoground is symmetrical in captions with just one different word, if one text can't be perturbed then the second one can't be either
  if(text0==text0pert): 
    continue
  input_c0_i0 = clip_processor(text=[text0], images=[example["image_0"].convert("RGB")], return_tensors="pt").to(device)
  input_c0pert_i0 = clip_processor(text=[text0pert], images=[example["image_0"].convert("RGB")], return_tensors="pt").to(device)
  input_c1_i1 = clip_processor(text=[text1], images=[example["image_1"].convert("RGB")], return_tensors="pt").to(device)
  input_c1pert_i1 = clip_processor(text=[text1pert], images=[example["image_1"].convert("RGB")], return_tensors="pt").to(device)
  output_c0_i0 = clip_model(**input_c0_i0)
  output_c0pert_i0 = clip_model(**input_c0pert_i0)
  output_c1_i1 = clip_model(**input_c1_i1)
  output_c1pert_i1 = clip_model(**input_c1pert_i1)
  clip_score_c0_i0 = output_c0_i0.logits_per_image.item()
  clip_score_c0pert_i0 = output_c0pert_i0.logits_per_image.item()
  clip_score_c1pert_i1 = output_c1pert_i1.logits_per_image.item()
  clip_score_c1_i1 = output_c1_i1.logits_per_image.item()
  winoground_clip_scores.append({"id" : example["id"], "c_i": clip_score_c0_i0, "cpert_i": clip_score_c0pert_i0})
  winoground_clip_scores.append({"id" : example["id"], "c_i": clip_score_c1_i1, "cpert_i": clip_score_c1pert_i1})

text_correct_count = 0
for result in winoground_clip_scores:
  text_correct_count += 1 if text_correct_pert(result) else 0

denominator = len(winoground_clip_scores)
print("text score:", text_correct_count/denominator)


winoground_clip_scores = []
print("Adverb Shuffling")
for example in tqdm(winoground):
  text0 = example["caption_0"]
  text1 = example["caption_1"]
  text0pert = advshuf(text0)
  text1pert = advshuf(text1)
  # If the perturbation return the same text, don't use it.
  # Becuase Winoground is symmetrical in captions with just one different word, if one text can't be perturbed then the second one can't be either
  if(text0==text0pert): 
    continue
  input_c0_i0 = clip_processor(text=[text0], images=[example["image_0"].convert("RGB")], return_tensors="pt").to(device)
  input_c0pert_i0 = clip_processor(text=[text0pert], images=[example["image_0"].convert("RGB")], return_tensors="pt").to(device)
  input_c1_i1 = clip_processor(text=[text1], images=[example["image_1"].convert("RGB")], return_tensors="pt").to(device)
  input_c1pert_i1 = clip_processor(text=[text1pert], images=[example["image_1"].convert("RGB")], return_tensors="pt").to(device)
  output_c0_i0 = clip_model(**input_c0_i0)
  output_c0pert_i0 = clip_model(**input_c0pert_i0)
  output_c1_i1 = clip_model(**input_c1_i1)
  output_c1pert_i1 = clip_model(**input_c1pert_i1)
  clip_score_c0_i0 = output_c0_i0.logits_per_image.item()
  clip_score_c0pert_i0 = output_c0pert_i0.logits_per_image.item()
  clip_score_c1pert_i1 = output_c1pert_i1.logits_per_image.item()
  clip_score_c1_i1 = output_c1_i1.logits_per_image.item()
  winoground_clip_scores.append({"id" : example["id"], "c_i": clip_score_c0_i0, "cpert_i": clip_score_c0pert_i0})
  winoground_clip_scores.append({"id" : example["id"], "c_i": clip_score_c1_i1, "cpert_i": clip_score_c1pert_i1})

text_correct_count = 0
for result in winoground_clip_scores:
  text_correct_count += 1 if text_correct_pert(result) else 0

denominator = len(winoground_clip_scores)
print("text score:", text_correct_count/denominator)

winoground_clip_scores = []
print("Adjective Shuffling")
for example in tqdm(winoground):
  text0 = example["caption_0"]
  text1 = example["caption_1"]
  text0pert = adjshuf(text0)
  text1pert = adjshuf(text1)
  # If the perturbation return the same text, don't use it.
  # Becuase Winoground is symmetrical in captions with just one different word, if one text can't be perturbed then the second one can't be either
  if(text0==text0pert): 
    continue
  input_c0_i0 = clip_processor(text=[text0], images=[example["image_0"].convert("RGB")], return_tensors="pt").to(device)
  input_c0pert_i0 = clip_processor(text=[text0pert], images=[example["image_0"].convert("RGB")], return_tensors="pt").to(device)
  input_c1_i1 = clip_processor(text=[text1], images=[example["image_1"].convert("RGB")], return_tensors="pt").to(device)
  input_c1pert_i1 = clip_processor(text=[text1pert], images=[example["image_1"].convert("RGB")], return_tensors="pt").to(device)
  output_c0_i0 = clip_model(**input_c0_i0)
  output_c0pert_i0 = clip_model(**input_c0pert_i0)
  output_c1_i1 = clip_model(**input_c1_i1)
  output_c1pert_i1 = clip_model(**input_c1pert_i1)
  clip_score_c0_i0 = output_c0_i0.logits_per_image.item()
  clip_score_c0pert_i0 = output_c0pert_i0.logits_per_image.item()
  clip_score_c1pert_i1 = output_c1pert_i1.logits_per_image.item()
  clip_score_c1_i1 = output_c1_i1.logits_per_image.item()
  winoground_clip_scores.append({"id" : example["id"], "c_i": clip_score_c0_i0, "cpert_i": clip_score_c0pert_i0})
  winoground_clip_scores.append({"id" : example["id"], "c_i": clip_score_c1_i1, "cpert_i": clip_score_c1pert_i1})

text_correct_count = 0
for result in winoground_clip_scores:
  text_correct_count += 1 if text_correct_pert(result) else 0

denominator = len(winoground_clip_scores)
print("text score:", text_correct_count/denominator)

winoground_clip_scores = []
print("Verb Shuffling")
for example in tqdm(winoground):
  text0 = example["caption_0"]
  text1 = example["caption_1"]
  text0pert = verbshuf(text0)
  text1pert = verbshuf(text1)
  # If the perturbation return the same text, don't use it.
  # Becuase Winoground is symmetrical in captions with just one different word, if one text can't be perturbed then the second one can't be either
  if(text0==text0pert): 
    continue
  input_c0_i0 = clip_processor(text=[text0], images=[example["image_0"].convert("RGB")], return_tensors="pt").to(device)
  input_c0pert_i0 = clip_processor(text=[text0pert], images=[example["image_0"].convert("RGB")], return_tensors="pt").to(device)
  input_c1_i1 = clip_processor(text=[text1], images=[example["image_1"].convert("RGB")], return_tensors="pt").to(device)
  input_c1pert_i1 = clip_processor(text=[text1pert], images=[example["image_1"].convert("RGB")], return_tensors="pt").to(device)
  output_c0_i0 = clip_model(**input_c0_i0)
  output_c0pert_i0 = clip_model(**input_c0pert_i0)
  output_c1_i1 = clip_model(**input_c1_i1)
  output_c1pert_i1 = clip_model(**input_c1pert_i1)
  clip_score_c0_i0 = output_c0_i0.logits_per_image.item()
  clip_score_c0pert_i0 = output_c0pert_i0.logits_per_image.item()
  clip_score_c1pert_i1 = output_c1pert_i1.logits_per_image.item()
  clip_score_c1_i1 = output_c1_i1.logits_per_image.item()
  winoground_clip_scores.append({"id" : example["id"], "c_i": clip_score_c0_i0, "cpert_i": clip_score_c0pert_i0})
  winoground_clip_scores.append({"id" : example["id"], "c_i": clip_score_c1_i1, "cpert_i": clip_score_c1pert_i1})

text_correct_count = 0
for result in winoground_clip_scores:
  text_correct_count += 1 if text_correct_pert(result) else 0

denominator = len(winoground_clip_scores)
print("text score:", text_correct_count/denominator)



winoground_clip_scores = []
print("Noun Chunk Shuffling")
for example in tqdm(winoground):
  text0 = example["caption_0"]
  text1 = example["caption_1"]
  text0pert = nounchunkshuf(text0)
  text1pert = nounchunkshuf(text1)
  # If the perturbation return the same text, don't use it.
  # Becuase Winoground is symmetrical in captions with just one different word, if one text can't be perturbed then the second one can't be either
  if(text0==text0pert): 
    continue
  input_c0_i0 = clip_processor(text=[text0], images=[example["image_0"].convert("RGB")], return_tensors="pt").to(device)
  input_c0pert_i0 = clip_processor(text=[text0pert], images=[example["image_0"].convert("RGB")], return_tensors="pt").to(device)
  input_c1_i1 = clip_processor(text=[text1], images=[example["image_1"].convert("RGB")], return_tensors="pt").to(device)
  input_c1pert_i1 = clip_processor(text=[text1pert], images=[example["image_1"].convert("RGB")], return_tensors="pt").to(device)
  output_c0_i0 = clip_model(**input_c0_i0)
  output_c0pert_i0 = clip_model(**input_c0pert_i0)
  output_c1_i1 = clip_model(**input_c1_i1)
  output_c1pert_i1 = clip_model(**input_c1pert_i1)
  clip_score_c0_i0 = output_c0_i0.logits_per_image.item()
  clip_score_c0pert_i0 = output_c0pert_i0.logits_per_image.item()
  clip_score_c1pert_i1 = output_c1pert_i1.logits_per_image.item()
  clip_score_c1_i1 = output_c1_i1.logits_per_image.item()
  winoground_clip_scores.append({"id" : example["id"], "c_i": clip_score_c0_i0, "cpert_i": clip_score_c0pert_i0})
  winoground_clip_scores.append({"id" : example["id"], "c_i": clip_score_c1_i1, "cpert_i": clip_score_c1pert_i1})

text_correct_count = 0
for result in winoground_clip_scores:
  text_correct_count += 1 if text_correct_pert(result) else 0

denominator = len(winoground_clip_scores)
print("text score:", text_correct_count/denominator)

