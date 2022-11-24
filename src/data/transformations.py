from torchvision import transforms
from clip import tokenize
from .VLMOrder.perturbtextorder import nounadjshuf
from .VLMOrder.perturbimage import RandomPatch

# TODO: Actually make this take in args and stuff, and not just be hard coded. Also we must add more transformations.
def train_transforms(data):
    image = data['image']
    text = data['text']
    distractor_image = RandomPatch(image, 4)
    distractor_text = nounadjshuf(text)

    train_image_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
    ])

    image = train_image_transforms(image)
    text = tokenize(text)
    distractor_image = train_image_transforms(distractor_image)
    distractor_text = tokenize(distractor_text)

    data['image'] = image
    data['text'] = text
    data['distractor_image'] = distractor_image
    data['distractor_text'] = distractor_text

    return data

def val_transforms(data):
    image = data['image'].convert('RGB')
    text = data['text']
    val_image_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
    ])

    image = val_image_transforms(image)
    text = tokenize(text)
    data['image'] = image
    data['text'] = text

    return data

def load_transforms(args):
    return train_transforms, val_transforms, val_transforms