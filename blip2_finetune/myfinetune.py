# ================================================================
# 0. 기본 세팅 -----------------------------------------------------
import torch, os
from datasets import load_dataset
from transformers import AutoProcessor, Blip2ForConditionalGeneration , default_data_collator
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import requests
DEVICE = "cuda:0"            # 단일 GPU
DTYPE  = torch.float16      # fp16 사용 (VRAM 24 GB 이상 권장)
EPOCHS = 30
BATCH  = 3
LR     = 1e-5

@torch.no_grad()
def generate_captions(loader, max_new_tokens=30, num_beams=3):
    model.eval()
    captions = []
    for batch in loader:
        generated_ids = model.generate(
            **batch,max_new_tokens=20
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        print(generated_text)

    return captions
# ================================================================
# 1. 데이터셋 -----------------------------------------------------
ds_train = load_dataset("ybelkada/football-dataset", split="train")

class TrainDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset, self.processor = dataset, processor
    def __len__(self):  return len(self.dataset)
    def __getitem__(self, idx):
        item = self.dataset[idx]
        enc = self.processor(
            images=item["image"],
            text = " ",
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )

        label_encoding = self.processor.tokenizer(
            item['text'],
            padding="max_length",
            max_length=256,
            truncation=True,
            return_tensors="pt",
        )
        labels = label_encoding["input_ids"].squeeze(0)
        labels[labels == processor.tokenizer.pad_token_id] = -100  # Padding 무시
        return {
        "pixel_values": enc["pixel_values"].squeeze(0),
        "input_ids": enc["input_ids"].squeeze(0),
        "attention_mask": enc["attention_mask"].squeeze(0),
        "labels": labels
        }

class InferDataset(Dataset):
    """이미지만 넘겨서 <image> 토큰을 자동 생성하게 함"""
    def __init__(self, dataset, processor):
        self.dataset, self.processor = dataset, processor
    def __len__(self):  return len(self.dataset)
    def __getitem__(self, idx):
        item = self.dataset[idx]
        enc = self.processor(item["image"] ,' Describe this image' ,return_tensors="pt")

        return {k: v.squeeze(0) for k, v in enc.items()}

def infer_collate(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])\
                         .to(DEVICE, dtype=DTYPE, non_blocking=True)
    return {"pixel_values": pixel_values}

# ================================================================
# 2. 모델 로드 (단일 GPU) -----------------------------------------
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=DTYPE).to(DEVICE)

# Q-Former만 학습
for n, p in model.named_parameters():
    p.requires_grad = n.startswith("qformer.")

opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                        lr=LR, weight_decay=0.01)

# ================================================================
# 3. 데이터로더 ---------------------------------------------------
dl_train = DataLoader(TrainDataset(ds_train, processor),
                      batch_size=BATCH, shuffle=True)
dl_infer = DataLoader(InferDataset(ds_train, processor),
                      batch_size=1, shuffle=False,
                      collate_fn=infer_collate)
model.eval()
generate_captions(dl_infer)
# ================================================================
# 4. 학습 ---------------------------------------------------------
model.train()
for epoch in range(EPOCHS):
    for step, batch in enumerate(dl_train):
        batch = {k: v.to(DEVICE, dtype=DTYPE if v.dtype==torch.float32 else v.dtype)
                  for k, v in batch.items()}
        loss = model(**batch).loss
        loss.backward()
        opt.step(); opt.zero_grad()
        print(f"Epoch {epoch} | step {step} | loss {loss.item():.4f}")

# ================================================================
# 5. 추론 (학습 이미지 → 캡션) ------------------------------------
model.eval()
generate_captions(dl_infer)

