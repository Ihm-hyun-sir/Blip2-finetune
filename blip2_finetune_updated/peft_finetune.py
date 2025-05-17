from datasets import load_dataset

from torch.utils.data import Dataset, DataLoader
import json
from PIL import Image
from dataloader import ImageTextDataset

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, choices=["fusecap", "sharegpt", "coco"])
parser.add_argument("--max_length", type=int, default=256)
parser.add_argument("--epoch", type=int, default=10)
parser.add_argument("--lr", type=float, default=1e-5)
args = parser.parse_args()

DATASET = args.dataset
MAX_LENGTH = args.max_length # max_length 설정
EPOCH = args.epoch
LR = args.lr

def collate_fn(batch):
    processed_batch = {}
    for key in batch[0].keys():
        if key != "text":
            # pixel_values 같은 항목 처리
            processed_batch[key] = torch.stack([example[key] for example in batch])
        else:
            # text → tokenizer로 변환
            text_inputs = processor.tokenizer(
                [example["text"] for example in batch],
                padding='max_length',
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            )
            input_ids = text_inputs["input_ids"]
            attention_mask = text_inputs["attention_mask"]

            # ✅ labels 생성: padding 위치를 -100으로 마스킹 (loss에서 무시되도록)
            labels = input_ids.clone()
            labels[input_ids == processor.tokenizer.pad_token_id] = -100

            # 추가
            processed_batch["input_ids"] = input_ids
            processed_batch["attention_mask"] = attention_mask
            processed_batch["labels"] = labels  # ✅ 핵심 추가!
    return processed_batch



from transformers import AutoProcessor, Blip2ForConditionalGeneration, BitsAndBytesConfig

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("ybelkada/blip2-opt-2.7b-fp16-sharded",  device_map={"": 0})

from peft import LoraConfig, get_peft_model

# Let's define the LoraConfig
config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj"]
)

model = get_peft_model(model, config)
model.print_trainable_parameters()

train_dataset = ImageTextDataset( f'./caption-train-data-20250509T105717Z-1-001/caption-train-data/{DATASET}_train_id2entry.json' , processor, train=True , max_length = MAX_LENGTH)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=2, collate_fn=collate_fn)

val_dataset = ImageTextDataset( f'./caption-train-data-20250509T105717Z-1-001/caption-train-data/{DATASET}_val_id2entry.json' , processor, train=False , max_length = MAX_LENGTH,)
val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=2, collate_fn=collate_fn)

test_dataset = ImageTextDataset( f'./caption-train-data-20250509T105717Z-1-001/caption-train-data/{DATASET}_test_id2entry.json' , processor, train=False, max_length = MAX_LENGTH,)
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=1 , collate_fn=collate_fn)

import torch
from tqdm import tqdm
from peft import get_peft_model_state_dict

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

device = "cuda" if torch.cuda.is_available() else "cpu"


for epoch in range(EPOCH):
    print("Epoch:", epoch)
    model.train()
    prog_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}")
    
    for idx, batch in enumerate(prog_bar):
        input_ids = batch.pop("input_ids").to(device)
        pixel_values = batch.pop("pixel_values").to(device, torch.float16)
        labels = batch.pop("labels").to(device)
        attn_mask = batch.pop("attention_mask").to(device)

        outputs = model(input_ids=input_ids,
                        pixel_values=pixel_values,
                        labels=labels,
                        attention_mask=attn_mask)

        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # 🔽 tqdm에 실시간 loss 표시
        prog_bar.set_postfix(loss=loss.item())
    
    
    torch.save(get_peft_model_state_dict(model), f"./peft_{DATASET}_epoch{epoch}.pth")
    model.eval()
    val_loss_sum = 0
    val_steps = 0
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch.pop("input_ids").to(device)
            pixel_values = batch.pop("pixel_values").to(device, torch.float16)
            labels = batch.pop("labels").to(device)
            attn_mask = batch.pop("attention_mask").to(device)

            outputs = model(input_ids=input_ids,
                            pixel_values=pixel_values,
                            labels=labels,
                            attention_mask=attn_mask)
            val_loss_sum += outputs.loss.item()
            val_steps += 1
    avg_val_loss = val_loss_sum / val_steps
    print(f"[Epoch {epoch}] Validation Loss: {avg_val_loss:.4f}")


@torch.no_grad()
def generate_captions(loader):
    model.eval()
    captions = []
    for i,batch in enumerate(loader):
        # 배치를 GPU로 이동
        pixel_values = batch['pixel_values'].to(device)
        generated_ids = model.generate(
            pixel_values=pixel_values, max_new_tokens= int(MAX_LENGTH * 1.1)
        )
        generated_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
        print(i)
        print(generated_text)
        captions.append(generated_text)
    return captions

model.eval()
print("Let's Test")
with torch.no_grad():
    generate_captions(test_dataloader)
print()  # 줄바꿈



