
from torch.utils.data import Dataset, DataLoader
import json
import random
from PIL import Image

class ImageTextDataset(Dataset):
    def __init__(self, json_path, processor, prompt = " " , train=True , max_length = 256):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.entries = list(self.data.values())
        
        self.processor = processor
        self.prompt = prompt
        self.train = train
        self.max_length = max_length
    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        image = Image.open(entry['image_path']).convert("RGB")
        caption = entry['caption']
        #print(caption)
        processed = self.processor(
            images=image, 
            padding="max_length",
            max_length = self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        if self.train : 
            processed = {k: v.squeeze() for k, v in processed.items()}
            processed["text"] = caption
            return processed
        
        elif not self.train : 
            return {
                "pixel_values": processed["pixel_values"].squeeze(0),
            }