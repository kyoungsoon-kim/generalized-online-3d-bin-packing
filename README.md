# GOPT: Generalizable Online 3D Bin Packing

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![Method](https://img.shields.io/badge/Method-Transformer%20%2B%20PPO-success)
![Domain](https://img.shields.io/badge/Domain-3D%20Bin%20Packing-9cf)
![Status](https://img.shields.io/badge/Status-Educational%20Scaffold-orange)

> Xiong et al. (2024), *GOPT: Generalizable Online 3D Bin Packing via Transformer-based Deep
> Reinforcement Learning* 논문을 재구현합니다. **Placement Generator**(빈 공간 후보 EMS 생성)와
> **Packing Transformer**(아이템↔공간 cross-attention 융합)를 결합해, 단일 정책으로 **서로 다른
> 박스 크기에 일반화**하는 온라인 3D 빈 패킹 에이전트를 PPO로 학습합니다.
> 공식 코드: https://github.com/Xiong5Heng/GOPT

---

## 🎯 핵심 기여

- **Placement Generator (PG)** — heightmap에서 코너점을 찾아 **EMS(Empty Maximal Space)** 고정개수
  `N`개를 생성·정규화([0,1])합니다. 행동공간을 `2N`(방향×EMS)으로 묶어 박스 크기와 무관하게 만듭니다 (§III-B/C).
- **Packing Transformer** — self-attention(공간끼리)+양방향 cross-attention(아이템↔공간)으로
  "이 공간이 이 아이템에 맞는가"를 학습합니다. 일반화의 핵심 (§III-D).
- **PPO + GAE 학습** — actor-critic 정책을 clipped surrogate로 안정 학습합니다 (§III-E, §IV-A).

## 🧩 파이프라인 한눈에

```
heightmap → [Placement Generator] → EMS 시퀀스 (N,6) + 아이템 (2,3) + 마스크 (2,N)
          → MLP 임베딩 → [Packing Transformer × 3] → actor: score map (2,N) → 마스킹 → softmax
                                                    → critic: mean-pool → V(s)
          → PPO(clip) + GAE 로 정책 갱신
```

## 📂 파일 구조

```
src/
  data.py    — BinPackingEnv + Placement Generator(EMS/heightmap/안정성/reward) 동작 구현 (§III-A/B/C)
  model.py   — Packing Transformer + actor/critic (§III-D) ★핵심
  loss.py    — PPO clipped objective + GAE (§III-E, §IV-A)
  train.py   — 벡터화 PPO 학습 루프(병렬 env+GAE+미니배치) + greedy 평가 + 선형감쇠 LR
  utils.py   — ModelConfig + YAML 로더
configs/base.yaml — 전 하이퍼파라미터(논문 인용 또는 [UNSPECIFIED] 표기)
notebooks/walkthrough.ipynb — 논문↔코드 + 이론셀 + 런타임 점검(전 셀 통과)
PAPER_GUIDE.md     — 논문 섹션별 해설 (educational)
REPRODUCTION_NOTES.md — 미명세·추정 항목 전부 명시
```

## 🚀 빠른 시작

```bash
py -3.12 -m pip install -r requirements.txt

# 실제 PPO 학습 (GPU 권장). --steps 로 예산 조절
py -3.12 -m src.train --steps 400000 --device cuda

# 워크스루 노트북: notebooks/walkthrough.ipynb (이론 + 점검셀)
```

## 📊 결과 (이 구현)

RTX 3060 Ti, greedy 평가, bin 10³. 학습이 랜덤 베이스라인 위로 **단조 상승**합니다.

| 설정 | env 스텝 | 적재율 | 패킹수 |
|---|---|---|---|
| 랜덤 유효정책 | — | 24.6% | 10.3 |
| GOPT(이 구현) | 20k | 32.4% | 13.3 |
| GOPT(이 구현) | 200k | 38.5% | 15.5 |
| **논문 GOPT(목표)** | **40M** | **76.1%** | **29.6** |
| 최고 베이스라인(Xiong[3]) | 40M | 73.8% | 28.3 |

## ⚠️ 재현 한계

네트워크·PPO·GAE·환경 **전부 구현·검증**(학습 단조 상승 확인). 단 **논문 76.1% 미달**, 원인 2가지:
1. **연산예산** — 200k~1M 스텝 vs 논문 **40M**(0.5~2.5%만). ~440 steps/s라 40M = 약 **25시간**(GPU).
2. **환경 단순화** — 안정성=완전지지(논문은 convex-hull 오버행 허용), EMS=평면 최대박스(§III-B 근사).
   둘 다 달성 가능 상한을 76% 아래로 낮춤.

논문 수치엔 (a) env 벡터화로 40M 스텝 가능화 + (b) convex-hull 안정성·정확한 EMS 추출 필요.
미명세 항목(attention head 수, critic 풀링 등)은 `REPRODUCTION_NOTES.md`·`[UNSPECIFIED]` 주석에 표기.

## 📖 인용

```bibtex
@article{xiong2024gopt,
  title   = {GOPT: Generalizable Online 3D Bin Packing via Transformer-based Deep Reinforcement Learning},
  author  = {Xiong, Heng and others},
  year    = {2024}
}
```
