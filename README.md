# SMART new1 추가 패치

이 디렉터리는 **`seulbinHwang/SMART`의 `new1` 브랜치 위에 덮어쓰는 최소 변경 overlay**다.

이번 패치는 아래 3가지만 반영한다.

1. **4-step midpoint ODE** 적분 추가
2. **optional short closed-loop fine-tuning** 추가
3. **heading `(sin, cos)` 재정규화**를 조립/적분 뒤에 넣어 drift 완화

아래는 **일부러 넣지 않은 것**이다.

- warm-start inference
- WaymoTargetBuilder의 무작위 32개 제한 제거
- random anchor 1개 기본값
- batch size 1만 된다고 가정하는 학습 로직

즉, 이번 패치는 **`new1`의 기본 구조를 유지한 채**, 요청한 세 항목만 보강하는 버전이다.

---

## 1. 바뀌는 파일

### 수정 파일
- `smart/utils/flow_traj.py`
- `smart/utils/__init__.py`
- `smart/modules/agent_flow_decoder.py`
- `smart/modules/smart_decoder.py`
- `smart/model/smart.py`
- `train.py`
- `configs/train/train_flow.yaml`
- `configs/validation/validation_flow.yaml`

### 새 파일
- `configs/train/train_flow_finetune.yaml`

### 건드리지 않는 파일
- `smart/transforms/target_builder.py`
  - 기존 **무작위 32개 제한 그대로 유지**
- 기존 데이터 전처리 / datamodule / map encoder / token state space

---

## 2. 무엇이 실제로 추가되었는가

### 2-1. 4-step midpoint ODE
기존 `new1`은 추론에서 noisy future segment를 여러 번 업데이트하긴 했지만, 각 step마다
단순히 velocity를 더하는 방식이었다.

이번 패치는 `smart/utils/flow_traj.py`에 `midpoint_ode_solve()`를 추가하고,
`SMARTAgentFlowDecoder`가 추론 시 이 적분기를 사용하도록 바꾼다.

핵심은 아래 순서다.

- 현재 segment state에서 velocity 계산
- half step 위치 계산
- half step 위치에서 velocity를 한 번 더 계산
- 그 값을 사용해 full step 업데이트
- 각 step 뒤 `(sin, cos)` 정규화

즉, **same 4-step budget** 안에서 더 안정적으로 적분한다.

### 2-2. heading `(sin, cos)` 재정규화
아래 두 지점에서 정규화를 넣었다.

- `assemble_4x6_to_21()` 뒤
- `midpoint_ode_solve()`의 각 적분 step 뒤

그래서 추론 중 누적 오차로 `(sin, cos)` 길이가 틀어지는 문제를 줄인다.

### 2-3. optional short closed-loop fine-tuning
`smart/model/smart.py`에 `closed_loop_steps`를 읽는 경로를 추가했다.

- `closed_loop_steps == 0`
  - 기존 open-loop flow 학습만 수행
- `closed_loop_steps > 0`
  - open-loop loss에 더해,
  - 짧게 rollout한 결과와 GT 사이의 loss를 같이 더함

여기서 `closed_loop_steps`는 **0.5초 단위 rollout 개수**다.
예를 들어 `closed_loop_steps: 4`면

- `shift = 5` raw step
- `4 x 0.5초 = 2.0초`

길이만큼 self-feeding rollout loss를 추가하는 뜻이다.

중요한 점은,
**warm-start는 여전히 넣지 않았다.**
즉 fine-tuning 때도 각 rollout step은 새 noise에서 시작한다.

---

## 3. 적용 방법

이 디렉터리는 overlay다.
`new1` 체크아웃된 repo 루트에 그대로 덮어쓰면 된다.

```bash
cp -r flow_smart_patch_v3/* /path/to/SMART/
```

예시:

```bash
git clone https://github.com/seulbinHwang/SMART.git
cd SMART
git checkout new1
cp -r /path/to/flow_smart_patch_v3/* .
```

---

## 4. 환경 설치

`new1`과 동일하게 가면 된다.

```bash
conda env create -f environment.yml
conda activate smart
pip install -r requirements.txt
```

---

## 5. 데이터 준비

기존 `new1`과 동일하다.

예시:

```bash
python data_preprocess.py \
  --input_dir ./data/waymo/scenario/training \
  --output_dir ./data/waymo_processed/training
```

validation도 같은 방식으로 준비하면 된다.

---

## 6. 학습 실행

### 6-1. 1단계: open-loop pretraining

```bash
python train.py \
  --config configs/train/train_flow.yaml \
  --save_ckpt_path ./checkpoints/flow_pretrain
```

기존 SMART 또는 기존 flow 모델 가중치로 시작하고 싶으면:

```bash
python train.py \
  --config configs/train/train_flow.yaml \
  --pretrain_ckpt /path/to/model.ckpt \
  --save_ckpt_path ./checkpoints/flow_pretrain
```

### 6-2. 2단계: short closed-loop fine-tuning

```bash
python train.py \
  --config configs/train/train_flow_finetune.yaml \
  --pretrain_ckpt ./checkpoints/flow_pretrain/epoch=XX.ckpt \
  --save_ckpt_path ./checkpoints/flow_finetune
```

이 config는 기본적으로 아래처럼 동작한다.

- `ode_steps: 4`
- `closed_loop_steps: 4`
- `closed_loop_eval: True`

즉,
**2초 open-loop 생성기 위에 2초 길이의 짧은 self-feeding loss를 추가한 finetune**이다.

---

## 7. 평가 실행

`new1`의 실행 방식 그대로 쓰면 된다.
`eval.py`가 있다면 그대로 쓰고, 없으면 `val.py`를 써도 된다.

예시:

```bash
python eval.py \
  --config configs/validation/validation_flow.yaml \
  --pretrain_ckpt ./checkpoints/flow_finetune/epoch=YY.ckpt
```

또는

```bash
python val.py \
  --config configs/validation/validation_flow.yaml \
  --pretrain_ckpt ./checkpoints/flow_finetune/epoch=YY.ckpt
```

---

## 8. config 설명

### `configs/train/train_flow.yaml`
기본 open-loop pretrain용이다.

중요 항목:

- `future_window_steps: 20`
  - 2.0초 미래 생성
- `anchor_chunk_k: 4`
  - scene 내부 anchor-chunk 병렬 감독 수
- `ode_steps: 4`
  - midpoint ODE 적분 step 수
- `overlap_loss_weight: 0.1`
  - segment 경계 일치 loss 가중치
- `closed_loop_steps: 0`
  - short rollout loss 없음

### `configs/train/train_flow_finetune.yaml`
짧은 closed-loop fine-tune용이다.

중요 항목:

- `closed_loop_steps: 4`
  - 0.5초 x 4번 = 2.0초 rollout loss
- `closed_loop_eval: True`
  - validation에서 rollout metric도 함께 기록
- `lr: 1e-4`
  - finetune 시작값으로 낮춤

### `configs/validation/validation_flow.yaml`
평가용이다.

- `ode_steps: 4`
- `closed_loop_eval: True`
- `closed_loop_steps: 0`
  - 평가에서는 학습용 rollout loss를 쓰지 않음

---

## 9. 로그 해석

### 학습 로그
- `train_loss`
- `train_flow_loss`
- `train_overlap_loss`
- `train_open_loop_ade`
- `train_rollout_loss`  
  - `closed_loop_steps > 0`일 때만 기록

### 검증 로그
- `val_loss`
- `val_flow_loss`
- `val_overlap_loss`
- `val_open_loop_ade`
- `val_rollout_minADE`
- `val_rollout_minFDE`

---

## 10. 실행 순서 추천

가장 안전한 순서는 아래다.

1. `train_flow.yaml`로 open-loop pretrain
2. `validation_flow.yaml`로 open-loop + rollout metric 확인
3. `train_flow_finetune.yaml`로 short closed-loop fine-tune
4. 다시 `validation_flow.yaml`로 최종 비교

---

## 11. 구현 의도 요약

이번 버전은 아래 원칙을 지킨다.

- SMART의 기존 데이터 / map / history / sparse backbone은 유지
- 미래 head만 flow 방식으로 유지
- 추가 변경은 **midpoint ODE + heading renorm + optional short CL finetune**로 제한
- warm-start는 일부러 넣지 않음
- WaymoTargetBuilder의 32개 제한은 유지
- 코드가 batch size 1만 된다고 가정하지 않도록 logging 경로는 보정

즉, **new1의 구조를 거의 그대로 두고 필요한 세 항목만 얹은 작은 패치**다.