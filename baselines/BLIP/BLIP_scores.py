import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from blip_itm import blip_itm
from datasets import load_dataset

# adapted from https://colab.research.google.com/github/salesforce/BLIP/blob/main/demo.ipynb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocess(image, image_size, device):
    # just take the preprocessing method from their demo for now
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    return transform(image).unsqueeze(0).to(device)

def score(output):
    return torch.nn.functional.softmax(output, dim=1)[:, 1].item()

image_size = 384 # default for BLIP
model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth'

model = blip_itm(pretrained=model_url, image_size=image_size, vit='base')
model.eval()
model = model.to(device=device)

auth_token = "" # use your own
winoground = load_dataset("facebook/winoground", use_auth_token=auth_token)["test"]

f = open("BLIP_scores.txt", "w")
f.write("ID\xa0tag\xa0secondary_tag\xa0num_main_preds\xa0collapsed_tag\xa0C0I0\xa0C0I1\xa0C1I0\xa0C1I1\n")

for example in winoground:
  i0 = preprocess(example["image_0"].convert("RGB"), image_size, device)
  i1 = preprocess(example["image_1"].convert("RGB"), image_size, device)
  c0 = example["caption_0"]
  c1 = example["caption_1"]
  output_c0_i0 = model(i0, c0, match_head="itm")
  output_c1_i0 = model(i0, c1, match_head="itm")
  output_c0_i1 = model(i1, c0, match_head="itm")
  output_c1_i1 = model(i1, c1, match_head="itm")
  score_c0_i0 = score(output_c0_i0)
  score_c1_i0 = score(output_c1_i0)
  score_c0_i1 = score(output_c0_i1)
  score_c1_i1 = score(output_c1_i1)
  row = [str(example["id"]), example["tag"], example["secondary_tag"], str(example["num_main_preds"]),
         example["collapsed_tag"], str(score_c0_i0), str(score_c0_i1), str(score_c1_i0),
         str(score_c1_i1)]
  f.write("\xa0".join(row) + "\n")