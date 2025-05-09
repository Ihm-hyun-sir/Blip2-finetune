## 📦 Script Options

```bash
python peft_finetune.py --dataset DATASET --max_length MAX_LEN --epoch N --lr LR
```

### ✅ Arguments

| Option         | Type   | Description                                  | Default     |
|----------------|--------|----------------------------------------------|-------------|
| `--dataset`    | string | 선택할 데이터셋: `fusecap`, `sharegpt`, `coco` | (required)  |
| `--max_length` | int    | 토큰 최대 길이                               | `256`       |
| `--epoch`      | int    | 학습 에폭 수                                  | `10`        |
| `--lr`         | float  | 학습률 (learning rate)                        | `1e-5`      |

### 🧪 Example

```bash
python peft_finetune.py --dataset fusecap --epoch 20 --lr 5e-5
```

if error occurs
```bash
LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH python peft_finetune.py --dataset sharegpt python peft_finetune.py --dataset fusecap --epoch 20 --lr 5e-5
```
