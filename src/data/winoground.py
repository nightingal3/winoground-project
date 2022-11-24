from torch.utils.data import Dataset
from datasets import load_dataset
import numpy as np

class WinogroundDataset(Dataset):
    def __init__(self, auth_token="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", seed=0, split="test", ratio=1.0, transform=None):
        self.split = split
        self.ratio = ratio
        self.seed = seed
        winoground =  load_dataset("facebook/winoground", use_auth_token=auth_token)['test']
        self.data = self.process_data(winoground)
        self.transform = transform
        
    def process_data(self, winoground):
        indices = np.arange(len(winoground))
        np.random.seed(self.seed)
        np.random.shuffle(indices)
        if self.split == 'test':
            indices = indices[:int(len(indices)*self.ratio)]
        else:
            indices = indices[int(len(indices)*self.ratio):]
        winoground = winoground.select(indices)
        
        data = []
        for example in winoground:
            e0 = {"image": example['image_0'], 'text': example['caption_0'], 'tag': example['tag'], 'secondary_tag': example['secondary_tag'], 'num_main_preds': example['num_main_preds'], 'collapsed_tag': example['collapsed_tag'], 'pair': len(data)+1}
            data.append(e0)
            e1 = {"image": example['image_1'], 'text': example['caption_1'], 'tag': example['tag'], 'secondary_tag': example['secondary_tag'], 'num_main_preds': example['num_main_preds'], 'collapsed_tag': example['collapsed_tag'], 'pair': len(data)-1}
            data.append(e1)
            
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]

        if self.transform is not None:
            return self.transform(data)
        else:
            return data