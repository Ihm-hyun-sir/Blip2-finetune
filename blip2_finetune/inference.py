import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from torch.utils.data import DataLoader
from dataloader import ImageTextDataset
from PIL import Image
from tqdm import tqdm   # 진행 상태 표시용

# ================================================================
# 0. 환경 설정
DEVICE        = "cuda:0"
DTYPE         = torch.float16       # fp16 사용
QFORMER_PATH  = "/home2/sir2000/blip2_finetune/qformer_final.pth"
BATCH_SIZE    = 1                   # validation 배치 크기
MAX_TOKENS    = 128
NUM_BEAMS     = 3

# ================================================================
# 1. Processor & Model 초기화
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=DTYPE
).to(DEVICE)

# ================================================================
# 2. Q-Former 파라미터 로드
# qformer_state = torch.load(QFORMER_PATH, map_location=DEVICE)
# model.qformer.load_state_dict(qformer_state)

# 모든 파라미터 동결
for param in model.parameters():
    param.requires_grad = False
model.eval()

# ================================================================
# 3. 캡션 생성 함수 정의
@torch.no_grad()
def generate_captions(loader, max_new_tokens=MAX_TOKENS, num_beams=NUM_BEAMS):
    captions = []
    for i,batch in enumerate(tqdm(loader, desc="Inference")):
        # GPU 및 dtype 이동
        batch = {
            k: v.to(DEVICE, dtype=DTYPE if v.dtype == torch.float32 else v.dtype)
            for k, v in batch.items()
        }
        # generate
        output_ids = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            temperature=0.7,
        )
        texts = processor.batch_decode(output_ids, skip_special_tokens=True)
        captions.extend([t.strip() for t in texts])
    return captions

# ================================================================
# 4. DataLoader 준비 (Validation 용)
dl_val = DataLoader(
    ImageTextDataset(
        json_path="sharegpt_train_id2entry_gender_0430_updated.json",
        processor=processor,
        train=False,
    ),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ================================================================
# 5. 배치 단위 추론 실행 및 출력
caps = generate_captions(dl_val)
for idx, cap in enumerate(caps, 1):
    print(f"[{idx:03d}] {cap}")

# ================================================================
# 6. 단일 이미지 직접 추론 예시
def infer_single_image(image_path: str):
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(DEVICE)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_TOKENS,
        num_beams=NUM_BEAMS
    )
    caption = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(caption)
    return caption

# 사용 예:
# print(infer_single_image("/path/to/your/image.jpg"))
