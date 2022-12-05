from .mscoco import COCO, COCODataset
from .winoground import WinogroundDataset
from .transformations import load_transforms

def get_dataset(args):
    train_transforms, val_transforms, test_transforms = load_transforms(args)

    train_dataset = None
    val_dataset = None    
    if args.train_dataset == 'coco':
        train_dataset = COCODataset(transform=train_transforms, root=args.coco_path, caption_year=args.caption_year)
        val_dataset = COCODataset(transform=val_transforms, root=args.coco_path, split='val', caption_year=args.caption_year)
    elif args.train_dataset == 'winoground':
        train_dataset = WinogroundDataset(transform=val_transforms, split="train", ratio=args.ratio)
        val_dataset = WinogroundDataset(transform=val_transforms, split='test', ratio=args.ratio)
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    test_dataset = None
    if args.test_dataset == 'coco':
        test_dataset = COCODataset(transform=test_transforms, split='test')
    elif args.test_dataset == 'winoground':
        test_dataset = WinogroundDataset(transform=test_transforms)
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    return train_dataset, val_dataset, test_dataset