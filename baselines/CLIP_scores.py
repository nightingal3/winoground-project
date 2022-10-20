from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel

# slightly modified version of CLIP_Implementation.ipynb that writes out the relevant information to a file
# that we can use for later analysis

auth_token = "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"
winoground = load_dataset("facebook/winoground", use_auth_token=auth_token)["test"]

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

f = open("CLIP_scores.txt", "w")
f.write("ID\xa0tag\xa0secondary_tag\xa0num_main_preds\xa0collapsed_tag\xa0C0I0\xa0C0I1\xa0C1I0\xa0C1I1\n")
print(winoground[4])

for example in winoground:
  input_c0_i0 = clip_processor(text=[example["caption_0"]], images=[example["image_0"].convert("RGB")], return_tensors="pt")
  input_c1_i0 = clip_processor(text=[example["caption_1"]], images=[example["image_0"].convert("RGB")], return_tensors="pt")
  input_c0_i1 = clip_processor(text=[example["caption_0"]], images=[example["image_1"].convert("RGB")], return_tensors="pt")
  input_c1_i1 = clip_processor(text=[example["caption_1"]], images=[example["image_1"].convert("RGB")], return_tensors="pt")
  output_c0_i0 = clip_model(**input_c0_i0)
  output_c1_i0 = clip_model(**input_c1_i0)
  output_c0_i1 = clip_model(**input_c0_i1)
  output_c1_i1 = clip_model(**input_c1_i1)
  clip_score_c0_i0 = output_c0_i0.logits_per_image.item()
  clip_score_c1_i0 = output_c1_i0.logits_per_image.item()
  clip_score_c0_i1 = output_c0_i1.logits_per_image.item()
  clip_score_c1_i1 = output_c1_i1.logits_per_image.item()
  row = [str(example["id"]), example["tag"], example["secondary_tag"], str(example["num_main_preds"]), example["collapsed_tag"], str(clip_score_c0_i0), str(clip_score_c0_i1), str(clip_score_c1_i0), str(clip_score_c1_i1)]
  f.write("\xa0".join(row) + "\n")


