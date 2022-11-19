from torch.utils.data import Dataset
from datasets import load_dataset
import numpy as np

class WinogroundDataset(Dataset):
    def __init__(self, auth_token="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", seed=0, split="test", ratio=1.0):
        self.split = split
        self.ratio = ratio
        self.seed = seed
        winoground =  load_dataset("facebook/winoground", use_auth_token=auth_token)['test']
        self.data = self.process_data(winoground)
        
    def process_data(self, winoground):
        indices = np.arange(len(winoground))
        np.random.seed(self.seed)
        np.random.shuffle(indices)
        if self.split == 'test':
            indices = indices[:int(len(indices)*self.ratio)]
        else:
            indices = indices[-int(len(indices)*self.ratio):]
        winoground = winoground.select(indices)
        
        data = []
        for example in winoground:
            data.append(example)
            
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]