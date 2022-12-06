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
        image0 = data['image_0']
        # image1 = data['image_1']
        text0 = data['text_0']
        # text1 = data['text_1']
        if distractor_type_text is not None:
            text_transform = transform_names[distractor_type_text]
        else:
            text_transform = lambda x: x

        distractor_image0 = RandomPatch(image0, 4) # there's only one img transform
        distractor_text0 = text_transform(text0)
        # distractor_image1 = RandomPatch(image1, 4) # there's only one img transform
        # distractor_text1 = text_transform(text1)

        train_image_transforms = transforms.Compose([
            transforms.RandomResizedCrop((224,224), scale=(0.8,1.2), ratio=(0.95,1.05)),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        ])

        image0 = train_image_transforms(image0)
        # image1 = train_image_transforms(image1)
        text0 = tokenize(text0)
        # text1 = tokenize(text1)
        data['image_0'] = image0
        # data['image_1'] = image1
        data['text_0'] = text0
        # data['text_1'] = text1

        distractor_image0 = train_image_transforms(distractor_image0)
        distractor_text0 = tokenize(distractor_text0)
        # distractor_image1 = train_image_transforms(distractor_image1)
        # distractor_text1 = tokenize(distractor_text1)

        data['distractor_image_0'] = distractor_image0
        data['distractor_text_0'] = distractor_text0
        # data['distractor_image_1'] = distractor_image1
        # data['distractor_text_1'] = distractor_text1

        return data

    return train_transforms

def val_transforms(data):
    data = copy.deepcopy(data)
    image0 = data['image_0']
    image1 = data['image_1']
    text0 = data['text_0']
    text1 = data['text_1']
    val_image_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
    ])

    image0 = val_image_transforms(image0)
    image1 = val_image_transforms(image1)
    text0 = tokenize(text0)
    text1 = tokenize(text1)
    data['image_0'] = image0
    data['image_1'] = image1
    data['text_0'] = text0
    data['text_1'] = text1

    return data

def load_transforms(args):
    return make_train_transforms(distractor_type_text=args.distractor_text, distractor_type_image=args.distractor_image), val_transforms, val_transforms