import json
import numpy as np
from transformers import AutoTokenizer

# 1) JSON 파일 경로 지정
json_path = "/home2/sir2000/blip2_finetune/sharegpt_train_id2entry_gender_0430_updated.json"

# 2) 파일 로드
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 3) 모든 caption 추출
captions = [entry["caption"] for entry in data.values()]

# 4) 토크나이저 로드 (BLIP-2용)
tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip2-opt-2.7b")

# 5) 캡션별 토큰 길이 계산
lengths = [len(tokenizer(c, add_special_tokens=True)["input_ids"]) for c in captions]

# 6) 통계 출력
print(f"전체 캡션 수: {len(lengths)}")
print(f"평균 토큰 길이    : {np.mean(lengths):.2f}")
print(f"최댓값 토큰 길이  : {np.max(lengths)}")
print(f"95퍼센타일 토큰 길이: {np.percentile(lengths, 95):.0f}")