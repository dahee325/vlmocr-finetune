## 멀티 GPU로 학습이 가능한가?

> Qwen3.5-9B VLM 모델은 단일 GPU에서 학습하기에는 모델 weight, activation memory, vision encoder, projector, 고해상도 이미지 token 처리로 인한 VRAM 부담이 크다.
> 
> 
> 따라서 **멀티 GPU 학습은 가능**하며, 단순 DDP(Distributed Data Parallel) 보다는 **Accelerate + DeepSpeed ZeRO(Zero Redundancy Optimizer) 기반 학습 구성이 적합**하다.
> 


Qwen3.5-9B VLM 모델은 멀티 GPU 학습이 가능하다.
다만 단순 DDP는 각 GPU에 모델 전체를 복사하기 때문에, 9B급 VLM 모델에서는 GPU 한 장의 VRAM 한계에 걸릴 수 있다.

따라서 초기 구현은 LoRA fine-tuning을 유지하면서 Accelerate + DeepSpeed ZeRO-2 기반으로 구성하는 것이 적절하다.
OOM이 발생하거나 모델이 단일 GPU에 올라가지 않는 경우 ZeRO-3 또는 FSDP를 추가 검토한다.



https://www.youngju.dev/blog/gpu-cuda/multi_gpu_training_setup#1-%EC%99%9C-multi-gpu-%ED%95%99%EC%8A%B5%EC%9D%B4-%ED%95%84%EC%9A%94%ED%95%9C%EA%B0%80

- **DDP** : Data Parallelism을 구현하는 PyTorch의 공식 모듈
    - 각 GPU에 모델 전체를 복사하고 데이터만 나눠 학습
    - 내부 동작 원리
        1. **초기화 단계** : Rank 0 프로세스의 모델 state를 ‘broadcast’하여 모든 프로세스가 동일한 초기 상태에서 시작
        2. **Forward Pass** : 각 프로세스가 자신의 data shard에 대해 독립적으로 forward pass를 수행
        3. **Backward Pass** : Backward pass 중에 gradient를 bucket단위로 ‘AllReduce’하여 동기화 함(Bucket 크기는 ‘bucket_cap_mb’파라미터로 조정 가능, 기본값:25MB)
        4. **Optimizer Step** : 모든 프로세스가 동기화된 동일한 gradient로 optimizer step을 수행하므로, 모델 파라미터가 항상 동일하게 유지
    - 한계 : 각 GPU에 모델 전체가 올라가야 하므로, 9B VLMOCR 모델에서는 GPU 한 장의 VRAM 한계가 문제가 될 수 있음
- **Accelerate** : HuggingFace에서 개발한 라이브러리, 통합 분산 학습 인터페이스
    - PyTorch 위에 얇은 래퍼(thin wrapper)를 제공
    - 새로운 프레임워크를 배울 필요 없이 기존 학습 루프를 거의 그대로 유지하면서 분산 학습에 적용 가능
    - 멀티 GPU, mix precision, DeepSpeed 연동 등을 관리하는 역할
    - 전체 API가 ‘Accelerator’ 하나의 클래스에 집중되어 있음
- **DeepSpeed ZeRO** : DeepSpeed는 Microsoft에서 개발한 분산 학습 라이브러리,  zeRO를 통해 메모리 효율적인 학습을 가능하게  함
    - optimizer state, gradient, parameter 등을 GPU 여러 장에 나눠 저장하여 메모리 중복을 줄
    1. **ZeRO Stage 1** : Optimizer State만 분산
    2. **ZeRO Stage 2** : Optimizer State + Gradient 분산
    3. **ZeRO Stage 3** : Optimizer State + Gradient + Parameter까지 분산

## 학습 설정

<strong>기본 전략</strong>

Qwen3.5-9B + LoRA + bf16 + flash_attention_2 + gradient checkpointing + Accelerate + DeepSpeed ZeRO-2

| 설정 | 역할 |
| --- | --- |
| LoRA | 전체 모델이 아니라 adapter 중심으로 학습하여 메모리 사용량 절감 |
| bf16 | FP32보다 메모리 사용량을 줄이고, FP16보다 학습 안정성이 높음 |
| flash_attention_2 | attention 연산 메모리 사용량과 속도 최적화 |
| gradient checkpointing | activation memory를 줄여 큰 모델 학습 시 OOM 가능성을 낮춤 |
| DeepSpeed ZeRO-2 | optimizer state와 gradient를 여러 GPU에 분산 |
| Accelerate | 멀티 GPU 실행 및 DeepSpeed 연동 관리 |

### DDP보다 DeepSpeed를 추천하는 이유

| 방식 | 설명 | Qwen3.5-9B 학습에서의 적합성 | 추천 여부 |
| --- | --- | --- | --- |
| DDP | 각 GPU에 모델 전체를 복사하고 데이터만 나눠 학습 | 모델이 GPU 한 장에 올라가야 해서 VRAM 한계가 발생할 수 있음 | 가능하지만 9B VLM에서는 우선순위 낮음 |
| DeepSpeed ZeRO-2 | optimizer state와 gradient를 GPU 여러 장에 분산 | LoRA finetuning 초기 구성에 현실적  | 1차 추천 |
| DeepSpeed ZeRO-3 | optimizer state, gradient, parameter까지 GPU 여러 장에 분산 | 메모리 절약 효과는 크지만 설정과 저장 구조가 복잡해질 수 있음 | ZeRO-2에서 OOM 발생 시 검토 |
| FSDP | PyTorch 기반 parameter sharding 방식 | 가능하지만 초기 구현 난이도와 디버깅 부담이 있음 | 가능하지만 초기 구현 난이도 높음 |
- 핵심 차이
    - DDP : GPU마다 모델 전체를 올림 → 모델이 GPU 한 장에 올라가야 함
    - DeepSpeed ZeRO : 학습에 필요한 optimizer state, gradient, parameter 등을 나눠 저장 → 큰 모델 학습에 더 유리

⇒ DDP도 멀티 GPU 학습은 가능하지만, 각 GPU에 모델 전체가 복사되므로 Qwen3.5-9B처럼 모델 크기가 큰 경우에는 GPU 한 장의 VRAM 한계가 문제가 될 수 있음

⇒ 따라서, 초기 구현은 Accelerate + Deepspeed ZeRO-2를 우선 적용하고, 메모리 부족이 발생하면ZeRO-3 또는 FSDP를 검토함

### `.env` 학습 설정

```python
# =========================
# Model
# =========================
MODEL_NAME=Qwen/Qwen3.5-9B
PROCESSOR_NAME=Qwen/Qwen3.5-9B

# =========================
# Precision / Memory
# =========================
TORCH_DTYPE=bfloat16
USE_FLASH_ATTENTION=true
ATTN_IMPLEMENTATION=flash_attention_2
GRADIENT_CHECKPOINTING=true

# =========================
# LoRA
# =========================
USE_LORA=true
LORA_RANK=8
LORA_ALPHA=16
LORA_DROPOUT=0.05
LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj

# =========================
# Training
# =========================
LEARNING_RATE=5e-6
WEIGHT_DECAY=0.1
WARMUP_RATIO=0.15
NUM_TRAIN_EPOCHS=1
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8

# =========================
# Image / Data
# =========================
TARGET_LONGEST_IMAGE_DIM=1288
MAX_SEQ_LENGTH=4096
NUM_WORKERS=2

# =========================
# Multi GPU
# =========================
USE_DEEPSPEED=true
DEEPSPEED_CONFIG=./configs/deepspeed_zero2.json
NUM_GPUS=4

# =========================
# Test
# =========================
SMOKE_TEST=true
SMOKE_TEST_STEPS=10
```

| 값 | 추천 초기값 | 이유 |
| --- | --- | --- |
| `PER_DEVICE_TRAIN_BATCH_SIZE` | `1` | GPU당 batch를 작게 잡아 OOM 방지 |
| `GRADIENT_ACCUMULATION_STEPS` | `8` | 작은 batch를 누적해서 실질 batch size 확보 |
| `TARGET_LONGEST_IMAGE_DIM` | `1288` | 1536보다 메모리 부담이 낮아 초기 테스트에 적합 |
| `MAX_SEQ_LENGTH` | `4096` | 긴 문서 처리 가능성을 남기되, 초기에는 과도하게 크게 잡지 않음 |
| `LORA_RANK` | `8` | 처음에는 메모리 안정성과 학습 가능성 확인을 우선 |
| `TORCH_DTYPE` | `bfloat16` | 메모리 절약 + 학습 안정성 확보 |
| `USE_DEEPSPEED` | `true` | 멀티 GPU 분산 학습 사용 |
| `NUM_GPUS` | `4` | 기록 및 effective batch size 계산용 값 |

⇒ 단, `.env` 의 `NUM_GPUS=4` 는 기록 및 batch size 계산용에 가깝고, 실제로 몇 개의 GPU를 사용할지는 실행 명령어의 `--num_processes 4` 또는 launcher 설정이 결정함

⇒ 실제 적용 시에는 사용 가능한 GPU 개수, VRAM 용량, CUDA/flash-attn 설치 여부에 따라 값 조정 필요

## DeepSpeed 환경 설정

- `configs/deepspeed_zero2.json`

```json
{
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "none"
    },
    "allgather_partitions": true,
    "reduce_scatter": true,
    "overlap_comm": true,
    "contiguous_gradients": true
  },
  "gradient_accumulation_steps": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_clipping": "auto"
}
```

⇒ Deepspeed ZeRO-2를 사용해 optimizer state와 gradient를 여러 GPU에 분산

⇒ bf16을 활성화해 메모리 사용량을 줄이고, gradient accumulation과 micro batch size는 Trainer 또는 Accelerate 설정과 연동되도록 `auto`로 둠

⇒ ZeRO-2는 parameter 자체까지 나누는 방식은 아니므로, 모델이 GPU 한 장에 전혀 올라가지 않는 경우에는 ZeRO-3를 검토해야 함

## 멀티 GPU batch size 계산

`effective batch size = per_device_train_batch_size × GPU 개수 × gradient_accumulation_steps`

- ex) GPU 4장일 때 `1 × 4 × 8 = 32`
    - 즉, `.env` 가 아래와 같이 되어있으면
    
    ```python
    PER_DEVICE_TRAIN_BATCH_SIZE=1
    NUM_GPUS=4
    GRADIENT_ACCUMULATION_STEPS=8
    ```
    
    ⇒ 실질 batch size는 32임
    
- 주의사항 : 단일 GPU에서 사용하던 gradient_accumulation_steps를 멀티 GPU에서 그대로 사용하면 effective batch size가 GPU 개수만큼 커짐
    
    ⇒ 따라서 멀티 GPU 전환 시 accumulation step을 함께 조정해야 함
    
    ⇒ effective batch size가 커지면 learning rate, warmup ratio, 학습 안정성에도 영향을 줄 수 있으므로 함께 점검해야 함
    
- ex ) 단일 GPU : `1 × 1 × 8 = 8` / 멀티 GPU 4장 : `1 × 4 × 8 = 32`
    - 같은 설정을 그대로 사용해도 멀티 GPU에서는 실제로 batch size가 4배 커질 수 있음
    - 따라서 같은 멀티 GPU  전환 후에는 loss 변화, 학습 속도, gradient accumulation step, learning rate를 함께 확인해야 함

## 실행 명령어

### 단일 GPU smoke test

```python
python train.py
# or
accelerate launch --num_processes 1 train.py
```

### 멀티 GPU smoke test

```python
accelerate launch --num_processes 4 --deepspeed_config_file configs/deepspeed_zero2.json train.py
# PowerShell
accelerate launch --num_processes 4 --deepspeed_config_file configs/deepspeed_zero2.json train.py
```

- `.env`의 NUM_GPUS=4는 기록/계산용에 가깝고, 실제로 몇 개 GPU를 쓸지는 --num_processes 4가 결정
    
    ⇒ 단, 실제 실행 명령어는 현재 학습 콛, 구조에 따라 달라질 수 있음
    
- `Trainer` 기반 코드라면 `TrainingArguments` 에서 `deepspeed` config를 읽을 수 있고, 순수 `Accelerate` 기반 코드라면 `accelerate config` 또는 launch 명령어에서 config 파일을 지정할 수 있음

⇒ 따라서 최종 실행 방식은 현재 [`train.py`](http://train.py) 가 DeepSpeed 설정을 어디서 읽는지 확인한 뒤 결정해야 함

## 검증 순서

<aside>

1. 모델/processor 로딩 확인
2. LoRA adapter 적용 확인
3. trainable parameter 수 출력
4. 샘플 1개 forward pass 확인
5. 단일 GPU 10 step smoke test
6. 멀티 GPU 10 step smoke test
7. nvidia-smi로 GPU별 메모리 사용 및 프로세스 분산 확인
8. checkpoint / LoRA adapter 저장 확인
9. 실제 학습 진행
</aside>

⇒ 특히 `model.print_trainable_parameters()` 출력에서 `trainable params` 가 0이면 LoRA가 제대로 붙지 않은 것

⇒ 멀티 GPU 실행 시에는 `nvidia-smi` 에서 여러 GPU 프로세스가 분산되어 올라갔는지 확인해야 함

⇒ 단일 GPU에서는 정상인데 멀티 GPU에서 오류가 난다면, DeepSpeed confg, process 수, device_map 설정, batch size, gradient accumulation 설정을 우선 확인해야 

## 최종 정리

> Qwen3.5-9B 모델은 멀티 GPU 학습이 가능하다.
>
> 다만 단순 DDP는 각 GPU에 모델 전체를 복사하기 때문에, 모델 weight와 activation memory, vision encoder, projector, 고해상도 이미지 token 처리로 인해 GPU 한 장의 VRAM 한계에 걸릴 수 있다.
>
> 따라서 초기 학습 구성은 전체 모델을 full fine-tuning하기보다 LoRA fin-tuning을 유지하고, `bf16`, `flash_attention_2`, `gradient checkointing`을 함께 적용하는 것이 적절하다.
>
> 분산 학습 방식은 `Accelerate + DeepSpeed ZeRO-2`를 1차로 적용하고, ZeRO-2에서도 OOM이 발생하거나 모델이 단일 GPU에 올라가지 않는 경우 ZeRO-3 또는 FSDP를 추가 검토한다.
> 멀티 GPU 전환 시에는 `per_device_train_batch_size × GPU 개수 × gradient_accumulation_steps`로 effective batch size가 증가하므로, 기존 단일 GPU 설정을 그대로 사용하면 실제 batch size가 과도하게 커질 수 있다.
>
> 따라서 `gradient accumulation step`, `learning rate`, `image resolution`, `max sequence length`를 함께 조정해야 한다.
>