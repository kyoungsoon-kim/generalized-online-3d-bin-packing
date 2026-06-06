# GOPT 구현 요약 가이드 (Implementation Summary)

본 문서는 **"Generalizable Online 3D Bin Packing via Transformer-based DRL"** 논문을 코드로 어떻게 구현했는지 간단히 설명하는 가이드입니다. 

---

## 1. 문제 정의 및 MDP 공식화 (State & Action)
논문은 온라인 3D 빈 패킹 문제를 **EMS(Empty Maximal Spaces, 빈 최대 공간)** 기반의 강화학습(RL) 문제로 정의했습니다.

* **State (상태):** 현재 상자(Bin)의 빈 공간을 나타내는 최대 80개의 EMS 벡터(`[N, 6]`)와 다음에 들어갈 아이템의 2가지 회전 상태(0도, 90도)를 나타내는 치수 행렬(`[2, 3]`)로 구성됩니다.
* **Action (행동):** `[들어갈 EMS 위치] x [아이템의 회전 방향]` 의 조합을 선택합니다. 불가능한 위치(아이템이 삐져나가거나 공중에 뜨는 등)는 Action Masking을 통해 제외됩니다.

이를 위해 `src/data.py` 에 강화학습 환경(`BinPackingEnv`) 뼈대를 구축하고 상태와 마스크를 생성하는 구조를 마련했습니다.

---

## 2. 핵심 아키텍처: Packing Transformer (`src/model.py`)
논문의 가장 큰 기여인 **Packing Transformer**는 크게 3단계로 구현되었습니다.

1. **임베딩 (Embedding):**
   * 입력된 EMS 행렬과 Item 행렬을 각각 2-Layer MLP를 통과시켜 128차원의 임베딩 벡터로 변환합니다.
2. **Transformer Encoder Block (총 3계층):**
   * **Self-Attention:** EMS들 간의 관계 파악, Item들 간의 관계를 각각 파악합니다.
   * **Bi-directional Cross-Attention:** EMS가 Item을, Item이 EMS를 양방향으로 쿼리(Query)하여 빈 공간과 물건 간의 상호작용 특징을 추출합니다.
   * 각 Attention 연산 이후에는 2-Layer MLP 및 LayerNorm과 Residual Connection이 적용되도록 세팅했습니다.
3. **Actor-Critic 헤드:**
   * **Actor:** Transformer를 통과한 EMS 특징과 Item 특징을 내적(Dot-product)하여 각 위치와 회전에 대한 행동 확률(Score map)을 계산합니다. 이때 물리적으로 불가능한 행동은 Mask를 씌워 `-inf`로 처리합니다.
   * **Critic:** 상태 특징들을 평균 풀링(Mean pooling)한 뒤 MLP를 거쳐 현재 상태의 가치(Value)를 스칼라 값으로 출력합니다.

---

## 3. PPO 훈련 루프 및 손실 함수 (`src/train.py`, `src/loss.py`)
* **손실 함수 (`ppo_loss`):** 논문에 명시된 PPO(Proximal Policy Optimization) 목적 함수를 구현했습니다. Policy Loss(클리핑 포함), Value Loss(MSE), Entropy Loss로 구성되며 논문에서 제시한 계수($c_1=0.5, c_2=0.001, \epsilon=0.3$)를 적용했습니다.
* **훈련 루프 (`train`):** 무작위로 생성된 더미(dummy) 에피소드 데이터를 통해 Actor와 Critic이 올바른 형태의 예측값과 로스를 반환하는지 확인할 수 있도록 최소 단위의 PPO 학습 사이클을 구현해두었습니다.

---

## 4. 파일 구조 요약
* `configs/base.yaml`: 모델 구조 및 학습에 필요한 논문 속 하이퍼파라미터 정의
* `src/model.py`: MLP, Packing Transformer, Actor-Critic 모델 아키텍처
* `src/loss.py`: PPO 학습을 위한 손실 함수
* `src/data.py`: 상자의 공간을 관리하고 EMS를 생성하는 RL 환경 클래스 스켈레톤
* `src/train.py`: 최소 단위의 PPO 훈련을 테스트해볼 수 있는 스크립트
* `REPRODUCTION_NOTES.md`: 논문에서 모호했던 부분(예: Attention Head 개수 등)의 판단 근거를 정리한 문서
