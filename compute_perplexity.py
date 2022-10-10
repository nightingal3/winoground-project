from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import numpy as np
import time
import json

model_name = 'bert-base-uncased'
model = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
preprocess = True
separator = "\t"

# scoring function taken from https://stackoverflow.com/questions/70464428/how-to-calculate-perplexity-of-a-sentence-using-huggingface-masked-language-mode
# see https://arxiv.org/pdf/1910.14659.pdf for how to estimate perplexity for MLM
def score(sentence, model=model, tokenizer=tokenizer):
    tensor_input = tokenizer.encode(sentence, return_tensors='pt')
    repeat_input = tensor_input.repeat(tensor_input.size(-1)-2, 1)
    mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2]
    masked_input = repeat_input.masked_fill(mask == 1, tokenizer.mask_token_id)
    labels = repeat_input.masked_fill( masked_input != tokenizer.mask_token_id, -100)
    with torch.inference_mode():
        loss = model(masked_input, labels=labels).loss
    return np.exp(loss.item())

f = open("data/winoground_captions.jsonl")
lines = f.readlines()
output = open("outputs/perplexity/winoground_perplexities.txt", "w")
for line in lines:
    d = json.loads(line.strip())
    s1 = d["caption_0"]
    s2 = d["caption_1"]

    # bert has a seizure if the sentence doesn't end with a period
    if preprocess:
        s1 = s1 + "."
        s2 = s2 + "."

    p1 = score(s1)
    p2 = score(s2)
    output.write(s1 + separator + s2 + separator + str(p1) + separator + str(p2) + "\n")
f.close()
output.close()