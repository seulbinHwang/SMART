# Flow-SMART patch for `rainmaker22/SMART`

이 폴더는 공식 SMART 레포에 **최소 변경**으로 flow-matching head를 붙이기 위한 교체 파일 모음이다.

핵심 아이디어는 아래 한 줄이다.

- **RoadNet, map tokenization, history token state space, sparse factorized interaction은 그대로 유지하고, `SMARTAgentDecoder`의 next-token 분류 head만 2.0초 / 4개 0.5초 segment / sparse conditional flow matching head로 바꾼다.**

이 구현은 아래 원칙을 따른다.

1. 기존 SMART의 데이터 전처리와 datamodule은 그대로 사용한다.
2. `map_decoder.py`는 그대로 둔다.
3. `smart/modules/agent_decoder.py`만 교체해서 `SMARTDecoder`의 호출 구조를 그대로 유지한다.
4. 학습은 **scene당 random anchor 1개**를 기본으로 한다.
5. history는 SMART 원래 설계와 같이 최근 최대 **6 token(3초)**을 유지한다.
6. 추론은 **4-step midpoint ODE**로 2.0초를 생성하고, 그중 앞 0.5초만 실제 다음 장면으로 사용한다.
7. 현재 구현은 **공식 공개 설정과 같은 `train_batch_size=1`**을 전제로 한다.

---

## 1. 어떤 파일을 어디에 복사해야 하는가

아래 파일을 공식 SMART 포크 레포의 같은 위치에 덮어쓰거나 새로 추가하면 된다.

### 교체 파일

- `smart/modules/agent_decoder.py`
- `smart/model/smart.py`
- `train.py`

### 새 파일

- `smart/utils/flow_traj.py`
- `configs/train/train_flow.yaml`

즉, 이 patch 폴더의 디렉터리 구조를 그대로 SMART 레포 루트에 복사하면 된다.

예시:

```bash
cd /path/to/your/SMART-fork
cp /path/to/flow_smart_patch/smart/modules/agent_decoder.py smart/modules/agent_decoder.py
cp /path/to/flow_smart_patch/smart/model/smart.py smart/model/smart.py
cp /path/to/flow_smart_patch/smart/utils/flow_traj.py smart/utils/flow_traj.py
cp /path/to/flow_smart_patch/configs/train/train_flow.yaml configs/train/train_flow.yaml
cp /path/to/flow_smart_patch/train.py train.py
```

`smart/modules/smart_decoder.py`는 **수정하지 않아도 된다.**
이유는 새 `agent_decoder.py`도 클래스 이름을 그대로 `SMARTAgentDecoder`로 유지했기 때문이다.

---

## 2. 환경 준비

공식 README와 동일하게 SMART 환경을 만든다.

```bash
conda env create -f environment.yml
conda activate SMART
pip install -r requirements.txt
```

PyG 설치 문제가 있으면 공식 README대로 설치한다.

---

## 3. 데이터 준비

공식 SMART와 동일하다.

### raw Waymo scenario 데이터를 전처리

```bash
python data_preprocess.py \
  --input_dir ./data/waymo/scenario/training \
  --output_dir ./data/waymo_processed/training
```

validation도 같은 방식으로 전처리한다.

구조 예시:

```text
SMART/
├── data/
│   ├── waymo/
│   │   └── scenario/
│   │       ├── training/
│   │       └── validation/
│   └── waymo_processed/
│       ├── training/
│       └── validation/
```

---

## 4. config 설명

이번 patch는 config를 거의 늘리지 않는다.

`configs/train/train_flow.yaml`에서 실제로 새로 생긴 핵심 항목은 아래뿐이다.

- `Model.train_stage`: `open_loop` 또는 `closed_loop`
- `Model.flow_ode_steps`: 기본 `4`
- `Model.closed_loop_unroll`: 기본 `4`

그 외 hidden dim, radius, time span 등은 공식 SMART 설정을 그대로 쓴다.

### 1단계 open-loop pretraining

```yaml
Model:
  train_stage: "open_loop"
  flow_ode_steps: 4
  closed_loop_unroll: 4
```

### 2단계 short closed-loop fine-tuning

같은 config를 복사해서 아래만 바꾼다.

```yaml
Model:
  train_stage: "closed_loop"
  flow_ode_steps: 4
  closed_loop_unroll: 4
```

---

## 5. 학습 순서

### 5-1. open-loop pretraining

공식 SMART checkpoint가 있으면 warm start를 권장한다.

```bash
python train.py \
  --config configs/train/train_flow.yaml \
  --pretrain_ckpt /path/to/original_smart.ckpt \
  --save_ckpt_path /path/to/save_dir
```

warm start 없이 처음부터도 가능하다.

```bash
python train.py \
  --config configs/train/train_flow.yaml \
  --save_ckpt_path /path/to/save_dir
```

이 단계에서 checkpoint monitor는 `val_flow_loss`다.

### 5-2. short closed-loop fine-tuning

`configs/train/train_flow.yaml`의 `Model.train_stage`를 `closed_loop`로 바꾼 뒤, open-loop ckpt를 이어서 학습한다.

```bash
python train.py \
  --config configs/train/train_flow.yaml \
  --pretrain_ckpt /path/to/open_loop.ckpt \
  --save_ckpt_path /path/to/save_dir_closed
```

이 단계에서 checkpoint monitor는 `val_minADE`다.

---

## 6. 검증 / 추론

현재 `smart/model/smart.py`의 `validation_step()`은 아래를 같이 수행한다.

1. open-loop flow loss 계산
2. 8초 rollout 추론
3. `val_minADE`, `val_minFDE` 계산

즉, `trainer.validate()`나 `trainer.fit()`의 validation 루프만으로도 rollout 성능을 같이 확인할 수 있다.

추론은 내부적으로 아래 순서로 돈다.

1. RoadNet을 scene당 한 번만 계산
2. 최근 최대 6 token history 유지
3. scene-level noise에서 시작해 2.0초 미래 생성
4. 앞의 0.5초만 실제 다음 장면으로 사용
5. 그 조각을 nearest SMART token으로 바꿔 history 갱신
6. 이를 16번 반복해 8초 rollout 완성

---

## 7. 중요한 구현 가정

이 patch는 일부러 compact하게 만들었기 때문에 아래 가정을 둔다.

### 배치 크기

- **batch_size=1 전제**다.
- 공식 공개 SMART 설정도 `train_batch_size=1`이므로, 이 가정을 유지하는 것이 가장 안전하다.
- multi-scene batch를 바로 지원하지는 않는다.

### supervision 대상

- context는 scene 안의 전체 agent를 쓴다.
- loss mask는 공개 SMART 관례를 따라 `category == 3` 기반 supervision을 유지한다.

### warm start

- 추론에서는 직전 step의 뒤 1.5초를 다음 step의 초기값으로 옮기는 **가벼운 warm start**가 들어 있다.
- 별도 proposal model은 사용하지 않는다.

---

## 8. 추천 실험 순서

가장 추천하는 순서는 아래다.

1. 공식 SMART ckpt로 warm start
2. `open_loop` stage 학습
3. `closed_loop` stage 짧게 fine-tune
4. validation rollout의 `val_minADE`, `val_minFDE` 확인
5. 필요한 경우에만 `flow_ode_steps`를 4에서 6으로 올려 비교

처음부터 튜닝 폭을 넓히지 말고, **기본 설정 그대로 먼저 돌리는 것**을 강하게 권장한다.

---

## 9. 이번 patch에서 의도적으로 하지 않은 것

아래 항목은 일부러 넣지 않았다.

- proposal + residual 2단 구조
- world/ADV alternating 학습
- Flow-Planner의 route branch / CFG / global fusion
- scene 안 13개 anchor full 병렬 학습
- history state space를 continuous encoder로 완전히 교체

이유는 전부 같다.
**너무 복잡하고 조잡해지기 쉽고, 공개 SMART의 강점을 버리게 되기 때문**이다.

---

## 10. 파일별 역할 요약

### `smart/modules/agent_decoder.py`

- 기존 NTP 분류 head 제거
- 2.0초 / 4 segment flow-matching decoder 추가
- 4-step ODE rollout 추가
- first-segment retokenization 추가

### `smart/utils/flow_traj.py`

- 21점 ↔ 4×6 segment 변환
- boundary consistency
- midpoint ODE solver

### `smart/model/smart.py`

- open-loop anchor sampling
- flow loss 계산
- short closed-loop training loop
- validation rollout metric

### `train.py`

- train stage에 따라 checkpoint monitor를 자동 전환

---

## 11. 마지막 확인 체크리스트

실행 전에 아래를 확인하면 된다.

- token 파일 경로가 공식 SMART와 동일한지
- `train_batch_size=1`인지
- `configs/train/train_flow.yaml`의 데이터 경로가 맞는지
- `--pretrain_ckpt`에 SMART 원본 ckpt를 넣었는지
- closed-loop stage에서는 `Model.train_stage: closed_loop`로 바꿨는지

이 다 맞으면, 이 patch만으로 공식 SMART 포크 레포 위에서 바로 학습을 시작할 수 있다.