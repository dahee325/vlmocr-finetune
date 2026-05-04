## olmOCR 파인튜닝 가이드 (소량 자체 데이터용)
### 1. 환경 설정
```bash
# olmOCR 기본 환경 설정
pip install .[train]
pip install transformers==4.52.4
pip install flash-attn>=2.8.0.post2 --no-build-isolation
```

### 2. 학습 데이터 준비

**데이터 형식**: PDF-Markdown 쌍으로 구성
```sh
my_training_data/
├── doc1.pdf          # 반드시 단일 페이지!
├── doc1.md
├── doc2.pdf
├── doc2.md
└── ...
```
#### Markdown 파일 형식:
```markdown
---
primary_language: ko  # 또는 en
is_rotation_valid: True
rotation_correction: 0
is_table: False
is_diagram: False
---
여기에 추출된 텍스트 내용...
```
### 3. 자체 데이터 준비 방법
#### 방법 1: 기존 PDF에서 자동 생성
```bash
# 1) olmOCR로 먼저 변환
python -m olmocr.pipeline ./workspace --pdfs /path/to/your/pdfs/*.pdf

# 2) 학습용 포맷으로 추출
python -m olmocr.data.prepare_workspace ./workspace ./my_training_data
```
이렇게 하면 PDF가 자동으로 단일 페이지로 분할되고, 올바른 형식으로 저장됩니다.
#### 방법 2: 수동으로 준비
* PDF를 단일 페이지로 분할
* 각 PDF에 대응하는 .md 파일을 직접 작성
* YAML front matter 포함 필수

### 4. 파인튜닝 설정 파일
#### 문서에서 언급된 LoRA 기반 파인튜닝 설정 사용:
```bash
# 설정 파일: qwen25_vl_olmocrv4_finetuning.yaml
# 이 파일에서 수정해야 할 부분:
```
#### 주요 수정 사항:
```yaml
## **주요 변경 사항:**

1. **`run_name`**: `custom-data`로 변경 (구분용)

2. **`dataset.train.root_dir`**: `./data/train` 
   - 학습용 PDF-MD 쌍이 여기 있어야 함

3. **`dataset.eval.root_dir`**: `./data/eval`
   - 평가용 PDF-MD 쌍이 여기 있어야 함

4. **`training.output_dir`**: `./checkpoints`
   - 체크포인트가 여기 저장됨

5. **`metric_for_best_model`**: `eval_nh-eval-data_loss`
   - 데이터셋 이름에 맞춰 변경

6. **`save_total_limit`**: `20`
   - 베스트 체크포인트 보존 상한

7. **체크포인트 정책(현재 코드 기준)**:
   - best 갱신 시점에 checkpoint 저장
   - history/랭킹 파일 자동 생성
```
### 5. 학습 실행
```bash
python -m olmocr.train.train \
  --config olmocr/train/configs/v0.4.0/qwen25_vl_olmocrv4_finetuning.yaml

위에 내용이 에러가 난다면, 아래 cmd 사용
CUDA_VISIBLE_DEVICES=0 python -m olmocr.train.train --config olmocr/train/configs/v0.4.0/qwen25_vl_olmocrv4_finetuning.yaml

그럼에도 OOM이 뜬다면, 아래 cmd 사용
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 python -m olmocr.train.train --config olmocr/train/configs/v0.4.0/qwen25_vl_olmocrv4_finetuning.yaml
```
#### 단일/멀티 GPU 실행 전환 (환경파일 1개)
```bash
# 1) .env.train에서 TRAIN_MODE를 single 또는 multi로 설정
# 2) 실행
./run_train.sh
```

`.env.train` 주요 항목:
- `TRAIN_MODE=single|multi`
- `TRAIN_CUDA_VISIBLE_DEVICES_SINGLE=0`
- `TRAIN_CUDA_VISIBLE_DEVICES_MULTI=0,1`
- `TRAIN_NPROC_PER_NODE=` (비우면 자동 계산)

멀티 모드에서는 `torch.distributed.run`(torchrun)로 실행되며,
train loop는 DDP 경로를 사용하도록 구성되어 있습니다.

#### 체크포인트 정책(최신)
- `load_best_model_at_end: true`이면 학습 종료 시 best checkpoint를 실제로 다시 로드
- best 갱신 기록 파일:
  - `<output_dir>/<run_name>/best_checkpoint_history.jsonl`
- 종료 후 순위 파일:
  - `<output_dir>/<run_name>/best_checkpoint_ranking.txt`
- pruning:
  - `save_total_limit` 초과 시 `eval + train` 기반 combined score로 성능이 낮은 checkpoint부터 삭제
  - 현재 global best checkpoint는 삭제 보호

#### 소요 시간/리소스:
* 소량 데이터(수백~수천 페이지): 수 시간 ~ 하루
* GPU: V100/A100/H100 등 (단일 GPU로 가능)
* Full training보다 훨씬 적은 리소스 필요

### 6. 체크포인트 준비 (학습 후)
```bash
# LoRA 어댑터를 전체 모델로 병합
python -m olmocr.train.prepare_checkpoint \
  ./checkpoints/my_olmocr_finetuned/checkpoint-XXXX \
  ./my_olmocr_final
```
### 7. FP8 양자화 (선택사항, 추론 속도 12% 향상)
```bash
python -m olmocr.train.compress_checkpoint \
  --config olmocr/train/quantization_configs/qwen2_5vl_w8a8_fp8.yaml \
  ./my_olmocr_final \
  ./my_olmocr_final_FP8
```
### 8. 파인튜닝된 모델 사용
```bash
python -m olmocr.pipeline ./workspace \
  --pdfs /new/documents/*.pdf \
  --model-path ./my_olmocr_final_FP8
```
--------------------------
--------------------------

### 핵심 차이점 정리
<table>
    <tr>
        <td>구분</td>
        <td>Full Training</td>
        <td>Fine-tuning</td>
    </tr>
    <tr>
        <td>시작점</td>
        <td>Qwen 베이스 모델</td>
        <td>학습된 olmOCR모델</td>
    </tr> 
    <tr>
        <td>데이터량</td>
        <td>~270,000</td>
        <td>페이지수백~수천 페이지</td></tr>
    <tr>
        <td>학습 방법</td>
        <td>Full model training</td>
        <td>LoRA adapter</td></tr>
    <tr>
        <td>소요 시간</td>
        <td>24-48시간 (B200)</td>
        <td>수 시간 ~ 하루</td>
    </tr>
    <tr>
        <td>비용</td>
        <td>~$300</td>
        <td>$10-50</td>
    </tr>
    <tr>
        <td>GPU</td>
        <td>B200/H100 x 1-8개</td>
        <td>V100/A100 x 1개</td>
    </tr>
    <tr>
        <td>설정 파일</td>
        <td>qwen25_vl_olmocrv4_rotation_1epoch_mix_1025_filtered.yaml</td>
        <td>qwen25_vl_olmocrv4_finetuning.yaml</td>
    </tr>
</table>

--------------------------
--------------------------
### 주의사항
1. **단일 페이지 필수**: PDF는 반드시 페이지당 하나씩 분할되어야 함
2. **데이터 품질**: 소량이므로 markdown 품질이 매우 중요 (직접 검토 권장)
3. **LoRA 사용**: 전체 모델 학습 대신 LoRA를 사용하면 메모리/시간 절약
4. **베이스 모델**: 이미 학습된 olmOCR-2-7B-1025-FP8를 시작점으로 사용