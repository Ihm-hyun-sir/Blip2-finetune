import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from dataloader import ImageTextDataset
from PIL import Image
from tqdm import tqdm
import json

# ================================================================
# 0. 환경 설정
DEVICE        = "cuda:0"
DTYPE         = torch.float16
QFORMER_PATH  = "/home2/sir2000/blip2_finetune/fuse_cap_pth/peft_fusecap_epoch4.pth"
BATCH_SIZE    = 3
MAX_LENGTH    = 240
NUM_BEAMS     = 3
device = "cuda" if torch.cuda.is_available() else "cpu"

# ================================================================
# 1. Processor & Base Model 초기화
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=DTYPE
).to(DEVICE)

# ================================================================
#2. PEFT 모델 구성 및 weight 로드
config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj"]
)

from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
model = get_peft_model(model, config)
state_dict = torch.load(
    QFORMER_PATH, 
    map_location="cpu"                # GPU 메모리 절약
)
set_peft_model_state_dict(model, state_dict)



# freeze 모든 파라미터 (추론용)
for p in model.parameters():
    p.requires_grad = False
model.eval()

# ================================================================
# 3. 캡션 생성 함수 정의

# ================================================================
# 4. DataLoader 준비 (Validation용)
test_dataloader = DataLoader(
    ImageTextDataset(
        json_path="/home2/sir2000/blip2_finetune/caption-train-data-20250509T105717Z-1-001/caption-train-data/coco_test_id2entry.json",
        processor=processor,
        train=False,
    ),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ================================================================
# 5. 배치 단위 추론 실행 및 출력
def generate_captions(loader):
    model.eval()
    captions = []
    for i, batch in enumerate(tqdm(loader, desc="Generating captions")):
        # 배치를 GPU로 이동
        pixel_values = batch['pixel_values'].to(device)
        generated_ids = model.generate(
            pixel_values=pixel_values,
            max_new_tokens=MAX_LENGTH
        )
        generated_texts = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )
        for text in generated_texts:
            entry = {
                'predicted_caption': text
            }
            captions.append(entry)
    return captions

model.eval()
print("Let's Test")
with torch.no_grad():
    results = generate_captions(test_dataloader)
save_path =  './fusecap_predicted.json'
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n✅ 결과가 {save_path} 에 저장되었습니다.")
