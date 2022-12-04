from torchvision import transforms
from clip import tokenize
from .VLMOrder.perturbtextorder import nounadjshuf, nonnounadjshuf, trigramshuf, nounshuf, advshuf, adjshuf, verbshuf
from .VLMOrder.perturbimage import RandomPatch
import copy

transform_names = {
    "nounadjshuf": nounadjshuf,
    "nonnounadjshuf": nonnounadjshuf,
    "trigramshuf": trigramshuf,
    "nounshuf": nounshuf,
    "advshuf": advshuf,
    "adjshuf": adjshuf,
    "verbshuf": verbshuf,
}
# TODO: Actually make this take in args and stuff, and not just be hard coded. Also we must add more transformations.
def make_train_transforms(distractor_type_text, distractor_type_image):
    def train_transforms(data):
        nonlocal distractor_type_text
        nonlocal distractor_type_image
        data = copy.deepcopy(data)
        image = data['image']
        text = data['text']
        if distractor_type_text is not None:
            text_transform = transform_names[distractor_type_text]
        else:
            text_transform = lambda x: x

        distractor_image = RandomPatch(image, 4) # there's only one img transform
        distractor_text = text_transform(text)

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

    return train_transforms

def val_transforms(data):
    data = copy.deepcopy(data)
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
    return make_train_transforms(distractor_type_text=args.distractor_text, distractor_type_image=args.distractor_image), val_transforms, val_transforms