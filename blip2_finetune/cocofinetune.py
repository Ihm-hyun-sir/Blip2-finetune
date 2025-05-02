# ================================================================
# 0. 기본 세팅 -----------------------------------------------------
import torch, os
from datasets import load_dataset
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from dataloader import ImageTextDataset
from tqdm import tqdm   # 추가

DEVICE = "cuda:0"            # 단일 GPU
DTYPE  = torch.float16      # fp16 사용 (VRAM 24 GB 이상 권장)
EPOCHS = 3
BATCH  = 4
LR     = 1e-5

@torch.no_grad()
def generate_captions(loader, max_new_tokens=256, num_beams=3):
    model.eval()
    captions = []
    for i,batch in enumerate(loader):
        # 배치를 GPU로 이동
        batch = {
            k: v.to(DEVICE, dtype=DTYPE if v.dtype == torch.float32 else v.dtype)
            for k, v in batch.items()
        }
        generated_ids = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams
        )
        generated_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0].strip()
        print(i)
        print(generated_text)
        captions.append(generated_text)
    return captions

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
dl_train = DataLoader( ImageTextDataset(
    json_path="sharegpt_train_id2entry_gender_0430_updated.json" , processor=processor , train=True), batch_size=BATCH, shuffle=True)

dl_val = DataLoader( ImageTextDataset(
    json_path="sharegpt_train_id2entry_gender_0430_updated.json" , processor=processor, train=False), batch_size=1, shuffle=False)
# model.eval()
# generate_captions(dl_infer)
# ================================================================
# 4. 학습 ---------------------------------------------------------
model.train()
for epoch in range(EPOCHS):
    epoch_iter = tqdm(dl_train, desc=f"[Epoch {epoch+1}/{EPOCHS}]", leave=False)
    for step, batch in enumerate(epoch_iter):
        # 장치 이동 및 dtype 변환
        batch = {
            k: v.to(DEVICE, dtype=DTYPE if v.dtype == torch.float32 else v.dtype)
            for k, v in batch.items()
        }
        loss = model(**batch).loss
        loss.backward()
        opt.step()
        opt.zero_grad()

        # tqdm에 현재 loss 표시
        epoch_iter.set_postfix(loss=f"{loss.item():.4f}")

    # (원한다면) 매 epoch 끝나고 체크포인트 저장
    # torch.save(model.qformer.state_dict(), f"qformer_epoch{epoch+1}.pth")
# ================================================================
# 4. 학습 후 Q-Former만 저장 ---------------------------------------
qformer_path = "qformer_final.pth"
torch.save(model.qformer.state_dict(), qformer_path)
print(f"✅ Q-Former state_dict saved to `{qformer_path}`")

# ================================================================
# 5. 추론 (학습 이미지 → 캡션) ------------------------------------
model.eval()
print("Let's Validate")
with torch.no_grad():
    generate_captions(dl_val, max_new_tokens=20, num_beams=3)
print()  # 줄바꿈