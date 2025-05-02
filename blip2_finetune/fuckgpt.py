import os
import json
import random
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoProcessor, Blip2ForConditionalGeneration

# ================================================================
# 0. 기본 세팅 -----------------------------------------------------
DEVICE      = "cuda:0"       # 단일 GPU
DTYPE       = torch.float16  # fp16 사용 (VRAM 24 GB 이상 권장)
EPOCHS      = 30
BATCH       = 4
LR          = 1e-5
SAMPLE_SIZE = 1000           # JSON에서 뽑아 쓸 샘플 개수
SEED        = 42
JSON_PATH   = "sharegpt_train_id2entry_gender_0430_updated.json"

# ================================================================
# 1. 캡션 생성 함수 -----------------------------------------------
@torch.no_grad()
def generate_captions(loader, max_new_tokens=30, num_beams=3):
    model.eval()
    for batch in loader:
        generated_ids = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams
        )
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        print(text[0])

# ================================================================
# 2. Dataset 정의 (이미지 → 캡션 학습용) ---------------------------
class ImageTextDataset(Dataset):
    def __init__(self, json_path, processor, sample_size=None, seed=SEED):
        # 1) JSON 로드
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        entries = list(data.values())
        
        # 2) 샘플링
        if sample_size is not None and sample_size < len(entries):
            random.seed(seed)
            entries = random.sample(entries, sample_size)
        
        self.entries = entries
        self.processor = processor

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry   = self.entries[idx]
        image   = Image.open(entry["image_path"]).convert("RGB")
        caption = entry["caption"]

        # ① Vision encoder 입력
        vision_inputs = self.processor(
            images=image,
            return_tensors="pt"
        )
        pixel_values = vision_inputs["pixel_values"].squeeze(0)

        # ② 캡션(label) 토크나이즈
        label_enc = self.processor.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        labels = label_enc["input_ids"].squeeze(0)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels":       labels,
        }

# ================================================================
# 3. Collate 함수 (배치 생성) --------------------------------------
def image_caption_collate(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    labels       = torch.stack([b["labels"]       for b in batch])
    return {"pixel_values": pixel_values, "labels": labels}

# ================================================================
# 4. Dataset 정의 (추론용: 이미지 → prompt 없이 캡션 생성) ----------
class InferDataset(Dataset):
    def __init__(self, json_path, processor, sample_size=None, seed=SEED):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        entries = list(data.values())
        if sample_size is not None and sample_size < len(entries):
            random.seed(seed)
            entries = random.sample(entries, sample_size)
        self.entries  = entries
        self.processor = processor

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        image = Image.open(entry["image_path"]).convert("RGB")
        enc = self.processor(
            images=image,
            return_tensors="pt"
        )
        return {k: v.squeeze(0) for k, v in enc.items()}

def infer_collate(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch]) \
                        .to(DEVICE, dtype=DTYPE, non_blocking=True)
    return {"pixel_values": pixel_values}

# ================================================================
# 5. 모델 로드 및 옵티마이저 ---------------------------------------
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model     = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b",
                torch_dtype=DTYPE
            ).to(DEVICE)

# Q-Former만 학습
for n, p in model.named_parameters():
    p.requires_grad = n.startswith("qformer.")

opt = torch.optim.AdamW(
    (p for p in model.parameters() if p.requires_grad),
    lr=LR,
    weight_decay=0.01
)

# ================================================================
# 6. DataLoader 생성 ---------------------------------------------
train_dataset = ImageTextDataset(
    json_path=JSON_PATH,
    processor=processor,
    sample_size=SAMPLE_SIZE,
    seed=SEED
)
dl_train = DataLoader(
    train_dataset,
    batch_size=BATCH,
    shuffle=True,
    collate_fn=image_caption_collate
)

infer_dataset = InferDataset(
    json_path=JSON_PATH,
    processor=processor,
    sample_size=SAMPLE_SIZE,
    seed=SEED
)
dl_infer = DataLoader(
    infer_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=infer_collate
)

# ================================================================
# 7. 초기 추론 (학습 전) ------------------------------------------
#generate_captions(dl_infer)

# ================================================================
# 8. 학습 루프 ----------------------------------------------------
model.train()
for epoch in range(EPOCHS):
    for step, batch in enumerate(dl_train):
        # GPU & dtype 옮기기
        batch = {
            k: v.to(DEVICE, dtype=DTYPE if v.dtype==torch.float32 else v.dtype)
            for k, v in batch.items()
        }
        loss = model(**batch).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        print(f"Epoch {epoch} | step {step} | loss {loss.item():.4f}")

# ================================================================
# 9. 학습 후 추론 -----------------------------------------------
generate_captions(dl_infer)
