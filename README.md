# SMART Flow Head Overlay

이 디렉터리는 **`rainmaker22/SMART` main 브랜치에 덮어쓰는 patch overlay**다.
즉, 새 래포를 처음부터 다시 만든 것이 아니라,

- 기존 SMART의 **데이터 전처리 / 로딩 / map encoder / token state space / 학습 진입점**은 최대한 유지하고
- 기존 **agent next-token prediction(NTP) head**만 걷어내고
- 그 자리에 **2.0초 horizon sparse conditional flow matching head**를 넣은 버전이다.

핵심 방향은 아래 한 줄이다.

> **RoadNet 유지 + SMART map pipeline 유지 + SMART token state space 유지 + flow-based 2.0초 미래 생성 head로 교체**

---

## 1. 무엇이 바뀌었는가

### 유지한 것

- `smart/modules/map_decoder.py`
- map token matching 전처리
- dataset / datamodule / target builder 흐름
- `train.py`, `val.py`의 진입 구조
- SMART token library (`smart/tokens/*`)
- sparse factorized attention 기반의 backbone 사용 방식

### 바꾼 것

- 기존 `SMARTAgentDecoder`의 **next-token classification head 제거**
- 새 `SMARTAgentFlowDecoder` 추가
- `SMARTDecoder`가 새 flow decoder를 사용하도록 변경
- `SMART.training_step()` / `validation_step()`를 **flow loss + overlap loss** 기준으로 변경
- `eval.py` 추가 (`val.py` alias)
- 새 config 추가
  - `configs/train/train_flow.yaml`
  - `configs/validation/validation_flow.yaml`

---

## 2. 이번 구현에서 의도적으로 단순화한 부분

성능보다 먼저 **코드 변경량 최소화**와 **파이프라인 일관성**을 우선해서 아래처럼 정리했다.

### 반영한 것

- 2.0초 미래를 4개의 0.5초 segment로 생성
- scene 내부 multi-anchor 감독 사용
- SMART map encoder 재사용
- SMART token history state space 재사용
- flow matching 기반 open-loop 학습
- 0.5초씩 갱신하는 closed-loop rollout 추론

### 일부러 넣지 않은 것

1. **stage-2 short closed-loop fine-tuning**
   - 넣을 수는 있지만, discrete re-tokenization까지 학습 루프에 얹으면 코드가 급격히 복잡해진다.
   - 현재 버전은 **open-loop flow 학습 + closed-loop rollout 평가** 구조다.

2. **warm start inference**
   - 성능에는 도움될 수 있다.
   - 하지만 구현이 커지고 버그 지점이 늘어나므로, 현재 기본 추론은 **매 0.5초 step fresh noise**로 다시 시작한다.

3. **world/ADV 분리 학습**
   - 공개 SMART repo 흐름을 유지하기 위해 넣지 않았다.

즉, 이 구현은 **작고 일관된 첫 버전**이다.

---

## 3. 새로 들어간 핵심 로직

### 학습 입력

한 anchor 시각마다

- scene-wide context agent의 **과거 SMART token 6-slot + 현재 continuous anchor token 1개**를 memory로 만들고
- target agent의 **미래 2.0초(21점)**를 local 좌표계에서 뽑은 뒤
- 이를 **4개의 겹치는 0.5초 segment (`4 x 6 x 4`)**로 바꾼다.

### 학습 목표

정답 segment `z`에 대해

- `noise ~ N(0, I)`
- `x_t = (1 - t) * noise + t * z`
- `u_t = z - noise`

를 만들고,
모델이 `x_t, t, context, map`을 보고 `u_t`를 맞히게 한다.

loss는 두 개만 쓴다.

- `flow_loss`: 예측 velocity field와 정답 velocity field의 MSE
- `overlap_loss`: 이웃 segment 경계점 일치 오차

### 검증

- `val_open_loop_ade`: 2.0초 open-loop local future 평균 위치 오차
- `val_rollout_minADE`, `val_rollout_minFDE`: 8초 closed-loop rollout 기준 오차

---

## 4. 파일 변경 요약

### 새 파일

- `smart/modules/agent_flow_decoder.py`
- `smart/utils/flow_traj.py`
- `smart/utils/list.py`
- `configs/train/train_flow.yaml`
- `configs/validation/validation_flow.yaml`
- `eval.py`

### 수정 파일

- `smart/modules/smart_decoder.py`
- `smart/modules/__init__.py`
- `smart/model/smart.py`
- `train.py`
- `val.py`
- `smart/utils/__init__.py`

---

## 5. 적용 방법

이 디렉터리는 **overlay**다.
이미 가지고 있는 `rainmaker22/SMART` 포크 래포 루트에 아래처럼 덮어쓰면 된다.

```bash
cp -r flow_smart_patch/* /path/to/your/SMART_fork/
```

주의:

- upstream SMART의 `smart/tokens/` 폴더는 그대로 있어야 한다.
- upstream SMART의 data 구조도 그대로 유지하는 전제다.

---

## 6. 환경 설치

upstream SMART 방식 그대로 가면 된다.

```bash
conda env create -f environment.yml
conda activate smart
pip install -r requirements.txt
```

PyG 설치가 환경마다 자주 꼬인다.
그 경우에는 `requirements.txt`에 맞춰 아래 패키지 버전을 직접 맞추면 된다.

- `torch==1.12.1`
- `torch-geometric==2.6.1`
- `torch-cluster==1.6.0+pt112cu113`
- `torch-scatter==2.1.0+pt112cu113`
- `torch-sparse==0.6.16+pt112cu113`
- `torch-spline-conv==1.2.1+pt112cu113`

---

## 7. 데이터 준비

upstream SMART와 동일하다.

### raw dataset 구조

```text
SMART
├── data
│   ├── waymo
│   │   ├── scenario
│   │   │   ├── training
│   │   │   ├── validation
│   │   │   ├── testing
```

### preprocessing

```bash
python data_preprocess.py \
  --input_dir ./data/waymo/scenario/training \
  --output_dir ./data/waymo_processed/training
```

validation도 같은 식으로 돌리면 된다.

---

## 8. 학습 실행

가장 기본 실행은 아래다.

```bash
python train.py \
  --config configs/train/train_flow.yaml \
  --save_ckpt_path ./checkpoints/flow
```

기존 SMART checkpoint를 최대한 재사용해서 시작하고 싶으면:

```bash
python train.py \
  --config configs/train/train_flow.yaml \
  --pretrain_ckpt /path/to/upstream_smart.ckpt \
  --save_ckpt_path ./checkpoints/flow
```

shape가 맞는 weight만 로드한다.
그래서 map encoder와 공통 backbone 일부는 재사용되고,
새 flow head 부분은 새로 초기화된다.

---

## 9. 평가 실행

README 기준 실행 이름을 맞추기 위해 `eval.py`를 추가했다.
실제 내용은 `val.py`와 같다.

```bash
python eval.py \
  --config configs/validation/validation_flow.yaml \
  --pretrain_ckpt /path/to/flow_model.ckpt
```

기본 validation config에서는

- open-loop metric
- closed-loop rollout metric

둘 다 기록한다.

---

## 10. 최소 튜닝 포인트

새로 추가한 핵심 설정은 아주 적게만 두었다.

### `configs/train/train_flow.yaml`

- `future_window_steps: 20`
  - 한 번에 생성하는 미래 길이. 2.0초.
- `anchor_chunk_k: 4`
  - scene 내부에서 동시에 감독에 쓰는 anchor 수.
- `ode_steps: 4`
  - 추론 때 flow ODE 적분 step 수.
- `overlap_loss_weight: 0.1`
  - segment 경계 일치 loss 가중치.
- `closed_loop_eval: False`
  - training 중 validation에서 rollout까지 볼지 여부.

### `configs/validation/validation_flow.yaml`

- `closed_loop_eval: True`

나머지 hidden dim, layer 수, radius는 upstream SMART 값을 그대로 유지했다.

---

## 11. 로그 해석

### 학습 로그

- `train_loss`
  - 최종 loss
- `train_flow_loss`
  - conditional flow matching 기본 loss
- `train_overlap_loss`
  - segment 경계 일치 loss
- `train_open_loop_ade`
  - 2.0초 open-loop local future 평균 위치 오차

### 검증 로그

- `val_loss`
  - 최종 validation loss
- `val_flow_loss`
  - flow loss
- `val_overlap_loss`
  - overlap loss
- `val_open_loop_ade`
  - open-loop 검증 지표
- `val_rollout_minADE`
  - 8초 closed-loop rollout 평균 오차
- `val_rollout_minFDE`
  - 8초 closed-loop 마지막 시점 오차

---

## 12. 구현 메모

이 버전은 **SMART의 학습/평가 파이프라인을 거의 그대로 유지하면서**, agent 미래 생성 방식만 flow로 바꾼 첫 안정화 버전이다.

그래서 다음 순서로 확장하는 것이 가장 안전하다.

1. 먼저 이 버전으로 학습이 정상 수렴하는지 확인
2. 그 다음 warm start inference 추가
3. 그 다음 short closed-loop fine-tuning 추가

이 순서가 가장 덜 복잡하고, 디버깅하기 쉽다.