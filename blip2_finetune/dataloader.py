
from torch.utils.data import Dataset, DataLoader
import json
import random
from PIL import Image

class ImageTextDataset(Dataset):
    def __init__(self, json_path, processor, prompt = " " , train=True):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        all_entries = list(self.data.values())
        
        split_idx = int(len(all_entries) * 0.999)
        if train:
            self.entries = all_entries[:split_idx]
        else:
            self.entries = all_entries[split_idx:]

        self.processor = processor
        self.prompt = prompt
        self.train = train
    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        image = Image.open(entry['image_path']).convert("RGB")
        caption = entry['caption']

        processed = self.processor(
            images=image, 
            text= self.prompt, 
            padding="max_length", 
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        if self.train : 
            label_encoding = self.processor.tokenizer(
                caption,
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="pt",
            )
            labels = label_encoding["input_ids"].squeeze(0)
            labels[labels == self.processor.tokenizer.pad_token_id] = -100  # Padding 무시
            return {
                "input_ids": processed["input_ids"].squeeze(0),
                "attention_mask": processed["attention_mask"].squeeze(0),
                "pixel_values": processed["pixel_values"].squeeze(0),
                "labels": labels  # caption을 label로
            }

        elif not self.train : 
            return {
                "input_ids": processed["input_ids"].squeeze(0),
                "attention_mask": processed["attention_mask"].squeeze(0),
                "pixel_values": processed["pixel_values"].squeeze(0),
            }