from datasets import load_dataset
from transformers import FlavaProcessor, FlavaForPreTraining
import torch

# slightly modified version of FLAVA_Implementation.ipynb that writes out the relevant information to a file
# that we can use for later analysis

auth_token = "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" # use your own
winoground = load_dataset("facebook/winoground", use_auth_token=auth_token)["test"]

flava_model = FlavaForPreTraining.from_pretrained("facebook/flava-full").eval().to("cpu")
flava_processor = FlavaProcessor.from_pretrained("facebook/flava-full")

i = open("FLAVA_itm_scores.txt", "w")
i.write("ID\xa0tag\xa0secondary_tag\xa0num_main_preds\xa0collapsed_tag\xa0C0I0\xa0C0I1\xa0C1I0\xa0C1I1\n")
c = open("FLAVA_contrastive_scores.txt", "w")
c.write("ID\xa0tag\xa0secondary_tag\xa0num_main_preds\xa0collapsed_tag\xa0C0I0\xa0C0I1\xa0C1I0\xa0C1I1\n")

winoground_flava_contrastive_scores = []
winoground_flava_itm_scores = []

for example in winoground:
    # Note that some images in winoground are RGBA and some are RGB. Need to convert all to RGB with .convert('RGB')
    inputs_c0_i0 = flava_processor(text=[example["caption_0"]], images=[example["image_0"].convert("RGB")],
                                   return_tensors="pt", padding="max_length", max_length=77,
                                   return_codebook_pixels=True, return_image_mask=True)
    inputs_c1_i0 = flava_processor(text=[example["caption_1"]], images=[example["image_0"].convert("RGB")],
                                   return_tensors="pt", padding="max_length", max_length=77,
                                   return_codebook_pixels=True, return_image_mask=True)
    inputs_c0_i1 = flava_processor(text=[example["caption_0"]], images=[example["image_1"].convert("RGB")],
                                   return_tensors="pt", padding="max_length", max_length=77,
                                   return_codebook_pixels=True, return_image_mask=True)
    inputs_c1_i1 = flava_processor(text=[example["caption_1"]], images=[example["image_1"].convert("RGB")],
                                   return_tensors="pt", padding="max_length", max_length=77,
                                   return_codebook_pixels=True, return_image_mask=True)

    inputs_c0_i0["input_ids_masked"] = inputs_c0_i0["input_ids"].detach().clone()
    inputs_c1_i0["input_ids_masked"] = inputs_c1_i0["input_ids"].detach().clone()
    inputs_c0_i1["input_ids_masked"] = inputs_c0_i1["input_ids"].detach().clone()
    inputs_c1_i1["input_ids_masked"] = inputs_c1_i1["input_ids"].detach().clone()

    inputs_c0_i0["bool_masked_pos"] = torch.zeros_like(inputs_c0_i0["bool_masked_pos"])
    inputs_c1_i0["bool_masked_pos"] = torch.zeros_like(inputs_c1_i0["bool_masked_pos"])
    inputs_c0_i1["bool_masked_pos"] = torch.zeros_like(inputs_c0_i1["bool_masked_pos"])
    inputs_c1_i1["bool_masked_pos"] = torch.zeros_like(inputs_c1_i1["bool_masked_pos"])

    outputs_c0_i0 = flava_model(**inputs_c0_i0)
    outputs_c1_i0 = flava_model(**inputs_c1_i0)
    outputs_c0_i1 = flava_model(**inputs_c0_i1)
    outputs_c1_i1 = flava_model(**inputs_c1_i1)

    flava_contrastive_scores_c0_i0 = outputs_c0_i0.contrastive_logits_per_image.item()
    flava_contrastive_scores_c1_i0 = outputs_c1_i0.contrastive_logits_per_image.item()
    flava_contrastive_scores_c0_i1 = outputs_c0_i1.contrastive_logits_per_image.item()
    flava_contrastive_scores_c1_i1 = outputs_c1_i1.contrastive_logits_per_image.item()
    row = [str(example["id"]), example["tag"], example["secondary_tag"], str(example["num_main_preds"]),
           example["collapsed_tag"], str(flava_contrastive_scores_c0_i0), str(flava_contrastive_scores_c0_i1), str(flava_contrastive_scores_c1_i0),
           str(flava_contrastive_scores_c1_i1)]
    c.write("\xa0".join(row) + "\n")

    flava_itm_scores_c0_i0 = torch.nn.functional.softmax(outputs_c0_i0.itm_logits)[0][1].item()
    flava_itm_scores_c1_i0 = torch.nn.functional.softmax(outputs_c1_i0.itm_logits)[0][1].item()
    flava_itm_scores_c0_i1 = torch.nn.functional.softmax(outputs_c0_i1.itm_logits)[0][1].item()
    flava_itm_scores_c1_i1 = torch.nn.functional.softmax(outputs_c1_i1.itm_logits)[0][1].item()
    row = [str(example["id"]), example["tag"], example["secondary_tag"], str(example["num_main_preds"]),
           example["collapsed_tag"], str(flava_itm_scores_c0_i0), str(flava_itm_scores_c0_i1),
           str(flava_itm_scores_c1_i0),
           str(flava_itm_scores_c1_i1)]
    i.write("\xa0".join(row) + "\n")