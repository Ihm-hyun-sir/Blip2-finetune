import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from dataloader import ImageTextDataset
from PIL import Image
from tqdm import tqdm

# ================================================================
# 0. 환경 설정
DEVICE        = "cuda:0"
DTYPE         = torch.float16
QFORMER_PATH  = "/home2/sir2000/blip2_finetune/peft_qformer_real_epoch9.pth"
BATCH_SIZE    = 3
MAX_TOKENS    = 256
NUM_BEAMS     = 3

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
    "/home2/sir2000/blip2_finetune/peft_qformer_real_epoch4.pth", 
    map_location="cpu"                # GPU 메모리 절약
)
set_peft_model_state_dict(model, state_dict)



# freeze 모든 파라미터 (추론용)
for p in model.parameters():
    p.requires_grad = False
model.eval()

# ================================================================
# 3. 캡션 생성 함수 정의
@torch.no_grad()
def generate_captions(loader):
    captions = []
    for i, batch in enumerate(tqdm(loader, desc="Inference")):
        pixel_values = batch['pixel_values'].to(DEVICE)

        output_ids = model.generate(
            pixel_values=pixel_values,
            max_length = 256
        )
        texts = processor.batch_decode(output_ids, skip_special_tokens=True)
        captions.extend([t.strip() for t in texts])
    return captions

# ================================================================
# 4. DataLoader 준비 (Validation용)
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

