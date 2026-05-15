# Qwen3.5-9B 아키텍처 분석 및 튜닝 시 주의사항

> **Qwen3.5-9B**는 Vision Encoder와 Language Model을 결합한 multimodal 구조 사용
긴 multimodal context 처리를 위해 GQA 및 Hybrid Attention 구조를 사용하는 모델
> 

## 1. Qwen3.5-9B config 기반 아키텍처 특징

### Qwen3.5-9B config.json

https://huggingface.co/Qwen/Qwen3.5-9B-Base/blob/main/config.json

| 항목 | 값 | 의미 | 영향 |
| --- | --- | --- | --- |
| `text_config.hidden_size` | 4096 | LLM token embedding 크기 | 커질수록 weight 및 VRAM 사용량 증가 |
| `text_config.num_hidden_layers`  | 32 | Transformer layer 수 | layer 증가 시 학습 속도 저하 및 메모리 증가 |
| `text_config.num_attention_heads` | 16 | attention head 수 |  `num_key_value_heads`와 함께 GQA 구조 판단 |
| `text_config.num_key_value_heads` | 4 | key/value head 수 | 여러 query head가 key/value head를 공유하는 GQA 구조 |
| `vision_config.hidden_size` | 1152 | vision encoder 내부 feature 크기 | 이미지 feature 표현 크기 |
| `vision_config.depth` | 27 | vision encoder layer 깊이 | vision encoder 연산량 증가 가능 |
| `vision_config.patch_size` | 16 | 이미지를 16×16 patch로 분할 | 작은 글씨 인식 유리, 대신image token 증가 가능 |
| `vision_config.spatial_merge_size` | 2 | vision token 공간 병합 | token 수 감소 목적 |
| `vision_config.out_hidden_size` | 4096 | vision output 크기 | language model hidden size와 연결 |

### Vision Encoder ↔ Language Model 연결 구조

```python
vision_config.hidden_size = 1152
vision_config.out_hidden_size = 4096
text_config.hidden_size = 4096
```

⇒ Vision Encoder가 생성한 image feature를 `out_hidden_size=4096` 으로 projection한 뒤 Language Model hidden size와 연결하는 구조로 볼 수 있음

⇒ 기존 Qwen2.5-VL-7B 기반 OCR/VLM 파이프라인에 Qwen3.5-9B를 적용할 경우,
hidden size, attention 구조, vision-text projection 구조 차이로 인해 기존 processor·memory 구조·학습 설정에 영향을 줄 가능성이 있음

## 2. GQA(Grouped Query Attention) 구조

```python
num_attention_heads = 16
num_key_value_heads = 4
```

⇒ 여러 query head가 더 적은 수의 key/value head를 공유하는 GQA 구조로 볼 수 있음

- 특징
    - KV cache 메모리 절약 가능
    - 긴 OCR 문서 처리에 유리
    - 긴 multimodal context 처리 효율 증가 가능

## 3. Hybrid Attention 구조

```python
"layer_types": [
  "linear_attention",
  "linear_attention",
  "linear_attention",
  "full_attention",
  ...
]
```

- `linear_attention` 와 `full_attention`으로 구성된 Hybrid Attention 구조 사용
- 특징
    - linear_attention : 긴 sequence 처리 효율 증가
    - full_attention : 문맥 이해 성능 유지

### OCR/VLM 환경에서의 영향

- OCR/VLM 환경에서는 `image token + text token` 이 함께 sequence에 포함되므로 일반 텍스트 모델보다 attention memory 사용량이 더 크게 증가할 수 있음
    
    ⇒ 따라서 Qwen3.5-9B는 대부분 layer에서는 linear attention을 사용해 긴 context 처리 효율 증가, 일정 간격마다 full attention을 사용해 전체 문맥 이해 성능 유지하는 구조로 볼 수 있음
    

```python
full_attention_interval = 4
```

⇒ 대략 4개 layer마다 full attention layer가 배치되는 설정으로 해석 가능

## 4. Qwen3.5-9B Modeling 구조 분석

https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py?utm_source=chatgpt.com

### (1) Vision Model + Text Model  구조

- visual : 이미지 처리 모델
- language_model : 텍스트 생성 모델

```python
self.visual = Qwen3VLVisionModel._from_config(config.vision_config)
self.language_model = Qwen3VLTextModel._from_config(config.text_config)
```

⇒ Qwen3VLModel은 내부적으로 Vision Model과 Text Model로 구성

### (2) 이미지가 Vision Model을 통과하는 부분

- `pixel_values` → `self.visual(...)` → `image_embeds` 생성
    
    ⇒ OCR 문서 이미지가 image embedding으로 변환되는 부분
    

```python
vision_output = self.visual(
    pixel_values,
    grid_thw=image_grid_thw,
    return_dict=True,
    **kwargs
)

image_embeds = vision_output.pooler_output
```

### (3) Image Token 위치 찾기

```python
image_mask, _ = self.get_placeholder_mask(
    input_ids,
    inputs_embeds=inputs_embeds,
    image_features=image_embeds
)
```

⇒ text token 내부 `<image>` 위치를 찾는 부분

### (4) Text Embedding 내부에 Image Embedding 삽입

```python
inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
```

⇒ image embedding을 text embedding 내부 image token 위치에 삽입하는 구조

---

### 전체 흐름

<aside>
💡

input_ids
→ self.get_input_embeddings()(input_ids)
→ inputs_embeds 생성

pixel_values
→ self.get_image_features()
→ self.visual(...)
→ image_embeds 생성

input_ids 안의 image_token 위치 찾기
→ get_placeholder_mask()

해당 위치에 image_embeds 삽입
→ inputs_embeds.masked_scatter(image_mask, image_embeds)

최종 inputs_embeds를 language_model에 전달
→ self.language_model(...)

</aside>

---

### (5) Language Model에 전달되는 구조

- `inputs_embeds` 내부에는 이미지 정보 포함

```python
outputs = self.language_model(
    input_ids=None,
    position_ids=position_ids,
    attention_mask=attention_mask,
    past_key_values=past_key_values,
    inputs_embeds=inputs_embeds,
    visual_pos_masks=visual_pos_masks,
    deepstack_visual_embeds=deepstack_visual_embeds,
    **kwargs,
)
```

⇒ 최종적으로 `text token + image embedding`이 결합된 형태로 Language Model에 전달되어 답변 생성에 사용됨

## 5. 튜닝 시 주의해야 할 점

### (1) 모델 구조 직접 수정 금지

```python
hidden_size 변경
layer 수 변경
```

- **위험요소**
    - pretrained weight shape mismatch 가능
    - projector/embedding 차원 불일치 가능
    - multimodal alignment 문제 가능
- **이유**
    - Qwen3.5-9B는 `Vision Encoder <-> Language Model` 구조가 연결되어 있음
        
        → 변경시 `vision output <-> text embedding` 차원 불일치 문제 발생할 수 있음
        
        ⇒ 따라서 기본 아키텍처를 유지한 상태에서 fine-tuning 수행이 더 안전함
        

### (2) VRAM 사용량 증가

- Qwen3.5-9B는 hidden_size 증가, 긴 sequence, image token 증가 때문에 메모리 사용량이 매우 커질 수 있음
- **특히 OCR/VLM 환경에서 문제되는 이유**
    - OCR문서는 `고해상도 이미지 + 긴 text token` 구조가 많음
        
        → `image token + text token` 길이가 크게 증가할 수 있음
        
        ⇒ attention memory 사용량 증가 가
        
- **대응 방법**
    
    
    | 방법 | 목적 |
    | --- | --- |
    | gradient checkpointing | 메모리 절약 |
    | bf16 | VRAM 감소 |
    | batch size 감소 | OOM 방지 |
    | LoRA / QLoRA | pretrained 구조 유지 + 메모리 절약 |

### (3) OCR 입력 형식 유지 필요

- 특히 유지해야 하는 값
    
    ```python
    pixel_values
    image_grid_thw
    mm_token_type_ids
    ```
    
- **이유**
    - multimodal position 계산, image token 위치 계산, RoPE 계산에 사용됨
- **위험 요소**
    - processor 출력 형식이 변경되면
        - image token mismatch  가능
        - multimodal alignment 오류 가능
        - image embedding 삽입 실패 가능

### (4) 긴 문서 처리 시 OOM 가능

- OCR/VLM 환경에서는 `image token + text token` 길이가 길어질 수 있음
- **영향**
    - attention memory 증가
    - VRAM 사용량 증가
    - batch 처리 불안정 가능
- **확인 필요 항목**
    
    
    | 항목 | 이유 |
    | --- | --- |
    | `max_length` | 긴 sequence 제한 |
    | image resolution | image token 수 증가 가능 |
    | batch size | 메모리 사용량 영향 |
    | PDFRenderer resolution | OCR 입력 크기 영향 |

### (5) Patch Size / Image Resolution Trade-off

```python
patch_size = 16
```

⇒ 이미지를 patch 단위로 분할하는 구조

- **특징**
    
    
    | patch size | 특징 |
    | --- | --- |
    | 작을수록 | 작은 글씨 인식 유리 |
    | 대신 | image token 증가 |
    | 결과 | attention memory 증가 가능 |
- **OCR 환경 영향**
    - OCR문서는 작은 글씨, 표, 수식 등이 많아 고해상도 입력이 필요한 경우가 많음
    - 하지만, image resolution 증가 시 image token 수가 증가하면서 VRAM 사용량이 급격히 증가할 수 있음

### (6) Attention 구조 직접 수정 위험

- Qwen3.5-9B는 `linear_attention + full_attention` 구조 사용
- **위험 요소**
    - attention 구조를 직접 변경할 경우
        - KV cache 구조 영향 가능
        - positional embedding 영향 가능
        - Flash Attention 호환성 문제 가능
    - 따라서 attention 구조 직접 수정보다는 기존 구조 유지 기반 fine-tuning이 더 안정적일 수 있음