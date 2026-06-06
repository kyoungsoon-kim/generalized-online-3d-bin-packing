**==> picture [79 x 66] intentionally omitted <==**

**==> picture [78 x 66] intentionally omitted <==**

**==> picture [78 x 65] intentionally omitted <==**

**==> picture [78 x 66] intentionally omitted <==**

# Generalized online 3D bin packing via Transformer-based DRL 

**IEEE ROBOTICS AND AUTOMATION LETTERS, VOL. 9, NO. 11, NOVEMBER 2024** 

**Heng Xiong et al.** 

**==> picture [79 x 66] intentionally omitted <==**

**==> picture [79 x 66] intentionally omitted <==**

**==> picture [79 x 65] intentionally omitted <==**

**==> picture [79 x 66] intentionally omitted <==**

**==> picture [58 x 51] intentionally omitted <==**

## Introduction 

- 글로벌무역및전자상거래시장의성장과함께, 최근몇년간창고자동화가빠르게발전하고있다. 최적의 포장전략들을통한창고내에서의효율적인물품적재는인력절감과비용절감등다양한이점을가져올 수있다 [1]. 그림 1은로봇팔을이용한물품피킹및포장의예를보여준다. 본논문에서는로봇피킹이잘 구현되어있다고가정한다. 연구자들은일반적으로로봇포장에서의적재문제를온라인 3차원 Bin 패킹 문제(3D Bin Packing Problem, 3D-BPP)로공식화하여다루어왔다 [2], [3]. 고전적인조합최적화문제중 하나인 3D-BPP는알려진직육면체물품들을축에맞춰정렬된방식으로 Bin에적재하여공간활용도를 극대화하는것을목표로한다.그러나실제많은상황에서모든물품들을관찰하고완전한정보를얻는것은 어렵다. 온라인 3D-BPP는이러한 3D-BPP의보다실용적인변형으로, 들어오는물품만을관찰하면서 하나씩포장하는방식을의미한다. 

- 제한된정보때문에, 온라인 3D-BPP는정확한해를구하는알고리즘들로해결할수없다 [4]. 이전 연구에서는주로인간포장자의경험을추상화하여설계된탐욕적목표들의휴리스틱들을개발하는데 집중해왔다 [5]. 그러나이러한휴리스틱들은직관적이긴하지만일반적으로최적이하의해를제공한다. 최근몇년간, 온라인 3D-BPP를심층강화학습(Deep Reinforcement Learning, DRL)을통해해결하려는 연구관심이증가하고있으며 [2], [3], [6], [7], 실제로 DRL 기반방법들은인상적인성능을보여주고있다. 그럼에도불구하고, 학습과정에서수렴(convergence)에도달하는데어려움을겪는경우가많으며 [2], [8], 이러한방법들은특히서로다른크기들의 Bin들로구성된다양한포장시나리오에대해효과적으로 일반화하기어려운한계를가진다. 이러한제한들은 DRL의일반적인활용가능성을상당히제한한다. 보다 구체적으로, 현재최고수준의 DRL 기반방법은훈련된 Bin 크기와동일한크기의 Bin에대해서만추론이 가능하다 [3], [9]. 훈련된모델은다른크기의 Bin에전이할수없다. 또한, 이러한방법에서는포장행동 공간(action space)의크기가상자크기에본질적으로의존하기때문에, 특히큰상자를다룰때모델 수렴에큰어려움을초래한다 [10]. 

2/26 

**Online 3D Bin Packing** 

**IME AI Modeling** 

**==> picture [79 x 66] intentionally omitted <==**

## Introduction 

**==> picture [58 x 51] intentionally omitted <==**

**==> picture [79 x 66] intentionally omitted <==**

**==> picture [79 x 65] intentionally omitted <==**

**==> picture [79 x 66] intentionally omitted <==**

3/26 

**Online 3D Bin Packing** 

**IME AI Modeling** 

**==> picture [79 x 66] intentionally omitted <==**

**==> picture [79 x 66] intentionally omitted <==**

**==> picture [79 x 65] intentionally omitted <==**

**==> picture [79 x 66] intentionally omitted <==**

**==> picture [58 x 51] intentionally omitted <==**

## Introduction 

- 앞서언급한한계들에착안하여, 본논문에서는 Transformer 기반 DRL을활용한일반화가능한온라인 3D Bin 패킹문제의접근법인 GOPT를제안한다(그림 2 참조). GOPT에서 Placement Generator(PG) 모듈은 먼저휴리스틱을사용하여현재 Bin 내에서고정길이의자유하위공간(free sub-space) 집합을배치 후보들로생성하며, 이를통해포장행동공간(action space)의크기를제어할수있다. 배치후보들과 포장할물품은함께마르코프결정과정(Markov Decision Process, MDP)의상태(state)로정의된다. 그후, GOPT는 Packing Transformer 모듈을통합한새로운포장정책네트워크를적용한다. 이모듈은현재 물품과사용가능한하위공간간의공간적상관관계뿐만아니라 PG 모듈에서제안된하위공간들간의 - 

- 관계를본질적으로파악함으로써 GOPT의일반화능력을향상시킨다. Packing Transformer는자기 - 

- 주의(self-attention) 층들과양방향교차 주의(bidirectional cross-attention) 층들을활용하여강화학습 정책의입력으로사용할특징(feature)들을추출한다. 

- 실험결과, 제안한방법은공간활용도와포장된물품수측면에서최신포장방법들을능가함을보여준다. 우리가아는한, 본연구는훈련된모델을통해다양한 Bin들에대해추론할수있는일반화능력을 제공하면서도높은성능을유지한최초의연구이다. 또한, 본논문에서는제안한포장계획방법을로봇 매니퓰레이터에적용하여실제환경에서의실용성을입증하였다. 

- 요약하면, 본연구의주요기여는다음과같다: (1) 포장성능과일반화능력을향상시키는온라인 3DBPP를위한새로운방법인 GOPT; (2) 포장행동공간(action space)을조절하고 Bin의상태를표현하는 Placement Generator 모듈; (3) 현재물품과사용가능한하위공간간의관계뿐만아니라하위공간들간의 상호관계를포착하는 Packing Transformer 네트워크; (4) GOPT와기존기법들을비교한광범위한실험 평가. 

4/26 

**Online 3D Bin Packing** 

**IME AI Modeling** 

**==> picture [79 x 66] intentionally omitted <==**

**==> picture [79 x 66] intentionally omitted <==**

**==> picture [79 x 65] intentionally omitted <==**

**==> picture [58 x 51] intentionally omitted <==**

## Related Work – Heuristic Methods 

– 초기연구들은주로단순성을위해효율적인휴리스틱설계에집중하였다. 연구자들은인간작업자의 경험에서추출한포장규칙을정의하려시도하였으며, 예를들어 First Fit [12], Best Fit [13], DeepestBottom-Left-Fill [14] 등이있다. Corner points(CP) [15], extreme points(EP) [16], empty maximal spaces(EMS) [17], internal corners point(ICP) [18] 등은휴리스틱방법들을향상시키기위해물품들을 포장할수있는잠재적자유공간들을표현하려고시도한다. 예를들어, Ha et al. [5]은 OnlineBPH를 제안했으며, 이는포장할물품의면들과 EMS의면들사이의여백을최소화하도록하나의 EMS를 선택한다. Yarimcam et al. [19]은 Irace 파라미터튜닝알고리즘 [20]을활용하여정책행렬(policy matrices) 형태로표현된휴리스틱을제공한다. Wang et al. [21]은 Heightmap-Minimization(HM)을제안하여점유된 부피를최소화하는배치를선호한다. 실제환경에서발생하는불확실성을완화하기위해, Shuai et al. [22]은변형된박스들을서로가깝게쌓아안정성을높인다. Hu et al. [23]은향후큰물품들을포장하기 위해사용가능한빈공간을최적화하는 Maximize-Accessible-Convex-Space(MACS) 전략을개발하였다. 이러한방법들은직관적이고효과적이지만, 수작업으로설계된규칙들에의존하며다양한문제 환경들에서일관되게우수한성능을발휘할능력이부족하다. 본연구는휴리스틱에서의빈공간들의 표현을기반으로하지만, 도메인전문가지식에제한받지않고 DRL을통해포장패턴들을학습한다. 

**==> picture [79 x 66] intentionally omitted <==**

5/26 

**Online 3D Bin Packing** 

**IME AI Modeling** 

**==> picture [79 x 66] intentionally omitted <==**

**==> picture [79 x 66] intentionally omitted <==**

**==> picture [79 x 65] intentionally omitted <==**

**==> picture [79 x 66] intentionally omitted <==**

**==> picture [58 x 51] intentionally omitted <==**

## Related Work – DRL-based Methods 

– DRL은일부조합최적화문제를해결하는데유망한성과를보여주었다 [24], [25]. 따라서최근에는 DRL을 활용하여 3D-BPP를해결하려는경향이있다. Que et al. [26]은 Transformer 구조를가진 DRL을사용하여 위치, 물품선택, 방향과같은하위작업을순차적으로처리함으로써높이가가변적인오프라인 3D-BPP를 다루었다. 반면, 본연구는온라인 3D-BPP에초점을맞추어위치와방향을동시에결정한다. 우리의지식 범위에서, Deep-Pack [27]은 DRL 기반모델을사용하여 2D 온라인포장문제를해결한최초의연구이며, 온라인 3D-BPP로확장할가능성이있다. 이방법은현재 Bin의상태를보여주는이미지를입력으로받아 들어오는물품을포장할픽셀위치를출력한다. Verma et al. [6]은탐색휴리스틱과 DRL을결합하여 임의의수와크기의 Bin들에대한문제를해결하기위해 2단계전략을제안한다. Zhao et al. [2], [10]은 문제를제약이있는 MDP로공식화하고 CNN 기반 DRL 에이전트를훈련시키기위해 ACKTR 방법 [28]을 채택하였다. [2]에서 DRL 에이전트는각각행동확률, 가치, 실행가능성마스크를추정하기위한 actor, critic, predictor로구성된다. 이후포장행동을길이, 너비, 방향차원으로분해하여행동공간을줄임으로써 개선된다 [10]. 그들은이어서휴리스틱탐색규칙을기반으로한 Packing Configuration Tree(PCT)를 도입하고이를 DRL 에이전트에통합하였다 [8]. 에이전트는 Graph Attention Networks [29]를정책으로 사용하며 ACKTR로학습된다. 휴리스틱과 DRL의시너지효과를조사하기위해, Yang et al. [7]은 PackerBot을제안하여휴리스틱보상을활용해 DRL 에이전트의성능을향상시킨다. Xiong et al. [3]은 후보지도(candidate map) 메커니즘을도입하여탐험복잡도를줄이고 A2C [30]로학습된 CNN 기반 DRL 에이전트의성능을개선한다. 이러한방법들은일반적으로물품과 Bin의특징들을직접 결합(concatenate)하여정책을학습한다. 반면, GOPT는먼저 Bin 내자유하위공간(free sub-space)을 제안하고, 수정된 Transformer를활용하여이공간들간의관계와현재물품과의관계를파악한다. 본 방법은다양한포장환경에서일반화가능성을보장한다. 

6/26 

**Online 3D Bin Packing** 

**IME AI Modeling** 

**==> picture [79 x 66] intentionally omitted <==**

**==> picture [79 x 66] intentionally omitted <==**

**==> picture [79 x 65] intentionally omitted <==**

**==> picture [79 x 66] intentionally omitted <==**

**==> picture [58 x 51] intentionally omitted <==**

## Methodology – Problem Description 

- 그림 1에서보듯이, 로봇은다양한크기의상자형태물품들이무질서하게쌓여있는더미에서무작위로 하나의물품을선택한다. 모든물품에대한완전한정보는사전에알수없다. 한대의카메라가선택된 물품의치수를측정하며, 이후해당물품은포장용기에배치된다. 이특정시나리오는온라인 3D Bin 패킹 문제(online 3D-BPP)로특징지을수있다. 목표는가능한한많은물품을 Bin에넣고, Bin의공간활용도를 최대화하는것이다. 

- - - 

- 우리는길이(L), 너비(W), 높이(H)의치수를가진 Bin의전면 좌측 하단(FLB: Front-Left-Bottom) 꼭짓점을 · · 

- 원점 (0, 0, 0)으로정의하며, 길이 너비 높이방향을각각 X, Y, Z 축방향으로설정한다(그림 2a 참고). 물품의경우, t번째물품의 FLB 좌표를 (𝑥𝑡, 𝑦𝑡, 𝑧𝑡) 로, 치수를 (𝑙𝑡, 𝑤𝑡, ℎ𝑡) 로나타낸다. 로봇포장작업에서는 다음과같은물리적제약조건을고려해야한다. 

   - 직교배치(Orthogonal placement): 물품들은직교(orthogonal) 방식으로 Bin에배치되며, 물품의 측면은 Bin의측면과정렬되어야한다. 즉, 물품의모서리가 Bin의 X,Y,Z 축과평행하게배치된다. 

   - 선택적방향(Optional orientation): 물품들은똑바로세운(upright) 방식으로배치된다. 첫번째제약 조건과결합하여, 물품들은 Bin 바닥면에대해 0도또는 90도의두가지수직평면내방향(vertical inplane orientations)만가질수있다. 

   - 정적안정성(Static stability): 포장과정에서물품들은중력과물품간의힘하에서안정적(stable) 상태를유지해야한다. 계산효율성을위해, 물품의기하학적중심(geometric center)의투영이해당 물품의모든수평지지점들(horizontal support points)의볼록껍질(convex hull)로형성되는지지 다각형(support polygon) 내부에놓일때해당물품은안정적인것으로간주된다. [23]. 

7/26 

**Online 3D Bin Packing** 

**IME AI Modeling** 

**==> picture [79 x 66] intentionally omitted <==**

## Methodology – Problem Description 

**==> picture [58 x 51] intentionally omitted <==**

**==> picture [723 x 362] intentionally omitted <==**

8/26 

**Online 3D Bin Packing** 

**IME AI Modeling** 

**==> picture [79 x 66] intentionally omitted <==**

**==> picture [79 x 66] intentionally omitted <==**

**==> picture [79 x 65] intentionally omitted <==**

**==> picture [79 x 66] intentionally omitted <==**

**==> picture [58 x 51] intentionally omitted <==**

## Methodology – Placement Generator 

- 포장할선택된물품에대해, 우리는 Bin 내에서의수평위치 (𝑥𝑡, 𝑦𝑡) 와배치방향을예측한다. 수직위치 𝑧𝑡 는중력으로인해가능한가장낮은배치위치로분석적으로결정된다. 앞서언급했듯이, 한물품에는두 가지가능한배치방향이있다. 따라서치수가 (L,W,H)인 Bin에한물품을배치할경우, 총 𝐿× 𝑊× 2 개의 가능한배치가존재하게된다 [2]. 한편, 이러한가능한배치개수는순차적결정(sequential-decision) 성격을가진포장문제에서는감당하기어렵다. 왜냐하면 Bin 치수가커질수록이개수는 기하급수적오케이으로증가하기때문이다. 다른한편으로, 이배치집합내에는실제로물품을효율적으로 배치하는데도움이되지않는경우도불가피하게존재한다. 

– 잠재적으로매우큰배치탐색공간을제한하기위해, 우리는 Placement Generator (PG) 모듈을설계하여 들어오는물품과현재 Bin 상태를기반으로유한하고효율적인배치부분집합을생성한다. 먼저, 높이 맵(heightmap)을활용하여 Bin의실시간상태를명시적으로표현한다. 이전물품들의계획된배치들을 표현으로활용하는다른방법들[8]은피드백과폐쇄형제어(closed-loop control)가부족하다. 반면, 높이 맵은실제로봇포장작업에서 PG를적용할때, 카메라로관찰한영상을통해손쉽게얻을수있다. 빈 공간을관리하기위한빈최대공간(Empty Maximal Space; EMS) 방식을참고하여 [17], [31], 후보 배치(candidate placements)는현재상태를기반으로계산된다. 구체적으로, 우리는높이맵의 X 및 Y 방향을따라높이변화를감지하여오목구석점들(corner points; 두개의면에의해둘러싸여안으로들어간 오목한구석, 기존물품이나 Bin의벽으로부터두방향(X, Y)에서지지를받을수있는출발점)을식별한다. 이후각오목구석점에서단위사각형들을확장하면서, 더높은고도를만나면확장을멈추어 EMS를 생성한다(그림 3). 각 EMS는그림 3c에나타난것처럼 FLB 꼭짓점과반대쪽꼭짓점으로정의할수있다. 생성된 6차원벡터는 Bin 크기와관계없이 [0, 1] 범위로정규화된다. 우리는크기제어가가능한 EMS 하위 𝑁 집합을얻고, 높이값기준으로순위를매겨 {𝐸𝑖}𝑖=1 으로표시한다. 마지막으로, 포장할물품이주어지면, 각 EMS의타당성을섹션 III-A에따라확인하고, EMS와 (물품의배치) 방향간쌍별(pairwise) 마스크를 생성한다. Bin 내에서물품을포장할때, 적절한 EMS와 (물품의배치) 방향을선택하고, 물품의 FLB 꼭지점을선택된 EMS의 FLB 꼭짓점과정렬한다(EMS의 FLB는해당 EMS를생성한오목구석점의좌표). 

9/26 

**Online 3D Bin Packing** 

**IME AI Modeling** 

**==> picture [79 x 66] intentionally omitted <==**

## Methodology – Placement Generator 

**==> picture [154 x 142] intentionally omitted <==**

**----- Start of picture text -----**<br>
FLB Y<br>X<br>**----- End of picture text -----**<br>


**==> picture [79 x 66] intentionally omitted <==**

**==> picture [120 x 107] intentionally omitted <==**

**==> picture [87 x 88] intentionally omitted <==**

**----- Start of picture text -----**<br>
Z<br>Y<br>FLB<br>X<br>**----- End of picture text -----**<br>


**==> picture [58 x 51] intentionally omitted <==**

**==> picture [79 x 66] intentionally omitted <==**

10/26 

**Online 3D Bin Packing** 

**IME AI Modeling** 

**==> picture [79 x 66] intentionally omitted <==**

**==> picture [79 x 66] intentionally omitted <==**

**==> picture [79 x 65] intentionally omitted <==**

**==> picture [79 x 66] intentionally omitted <==**

**==> picture [58 x 51] intentionally omitted <==**

## Methodology – Reinforcement learning formulation 

- DRL 문제는일반적으로마르코프결정과정(Markov Decision Process, MDP)으로모델링된다. 본논문에서는파라미터 𝒮,𝒜,𝑃,𝑅,𝛾 를갖는 MDP를사용하여포장환경을특징화한다. 여기서, 𝒮 는상태공간(state space), 𝒜 는행동 

- 공간(action space), 𝑃:𝒮×𝒜×𝒮→(0,1] 는전이확률(transition probabilities), 𝑅:𝒮×𝒜→ℝ 은스칼라보상함수(scalar reward function), 𝛾∈(0,1] 은 DRL에서단기보상과장기보상사이의균형을조정하는할인율(discount factor)이다. 

- – 강화학습알고리즘은정책 𝜋:𝒮×𝒜→ℝ 를학습하는것을목표로하며, 이는주어진상태 𝑠 에서행동 𝑎 를선택할 확률을결정한다. 정책의목적은에피소드전체에서누적할인보상(cumulative discounted reward)을최대화하는 것이며, 수식으로는 σ𝑡 𝛾[𝑡] 𝑟𝑡 로표현된다. 여기서 𝑡 는시간단계(time step), 𝑟𝑡 ​, 𝑎𝑡 ​, 𝑠𝑡 는각각시간단계 𝑡 에서의보상, 행동, 상태를나타낸다. 이후, 우리는온라인 3D-BPP를 DRL 학습을위한 MDP로공식화한다. 

- 상태 **(State):** 각시간단계 𝑡 에서, 정책은상태 𝑠𝑡 를받으며, 여기에는포장할들어오는물품 𝑠𝑡,𝑖𝑡𝑒𝑚 과현재 Bin 구성 𝑠𝑡,𝑏𝑖𝑛 이 포함된다. 

   - 첫번째부분에서, 물품의치수 (𝑙𝑡,𝑤𝑡,ℎ𝑡) 가필수적이다. 일부연구 [3], [7]에서는이 3차원벡터를명시적으로물품 표현으로사용하며, 다른연구 [2], [9]에서는신경망설계편의성을위해 3채널맵을선호한다. 맵표현에서는각 채널이각각 𝑙𝑡,𝑤𝑡,ℎ𝑡 를할당받는다. 물품의기하학적정보와선택적 방향(orientation)을모두고려하기위해, 우리는 2×3 행렬형태의물품표현을제안한다: 

**==> picture [139 x 32] intentionally omitted <==**

   - ° 

   - – 여기서 (𝑙𝑡,𝑤𝑡,ℎ𝑡) 와 (𝑤𝑡,𝑙𝑡,ℎ𝑡) 는각각물품을 0°와 90 회전시킨후의치수를나타낸다. 

- 두번째부분에서, 기존방법으로는 heightmap [3], 포장된물품리스트 [8], 가중 3D 복셀그리드(weighted 3D voxel grid) [9] 등이있다. 우리는제안된 PG(Section III-B)를활용하여배치제약을만족하는 EMS(Empty Maximal Space) 순차열를생성하고, 이를 Bin 구성으로사용한다. 이순차열은더미 EMS를사용하여고정길이 𝑁 으로 패딩하거나잘라낸다. 즉, 

**==> picture [93 x 17] intentionally omitted <==**

11/26 

**Online 3D Bin Packing** 

**IME AI Modeling** 

**==> picture [79 x 66] intentionally omitted <==**

**==> picture [79 x 66] intentionally omitted <==**

**==> picture [79 x 65] intentionally omitted <==**

**==> picture [79 x 66] intentionally omitted <==**

**==> picture [58 x 51] intentionally omitted <==**

## Methodology – Reinforcement learning formulation 

– 행동 **(Action):** 주어진포장상태 𝑠𝑡 = (𝑠𝑡,𝑖𝑡𝑒𝑚,𝑠𝑡,𝑏𝑖𝑛) 에서, 행동 𝑎𝑡 는현재물품에대해사용가능한 EMS 순차열 중하나를 선택하고동시에 (물품의적재) 방향(orientation)을결정하는것을의미한다. 행동공간 𝒜 의크기는오직순차열길이와 선택가능한 (물품의적재) 방향의수에만의존하며, 즉 𝒜= 2𝑁 이고용기의치수와는무관하다. 훈련과정에서는 정책 π ∙𝑠𝑡 에따른행동확률분포에기반하여행동 𝑎𝑡 를선택한다. 여기서 ∙ 은상태 𝑠𝑡 에서가능한모든배치집합을 의미한다. 테스트과정에서는 π ∙𝑠𝑡 에서확률이최대인배치를선택하여결정론적방식(deterministic manner)으로 행동을수행한다. 또한, EMS와 (물품의적재) 방향사이의쌍별 (pairwise) 행동마스크를적용한확률분포는, 모든 EMS들이제약조건을충족하지못하는경우를제외하고는, 정책이항상유효한행동들(valid actions)을샘플링하도록 보장한다. 

- 상태전이 **(State-Transition):** 본연구의설정에서전이모델은결정론적(deterministic)이라고가정한다. 이는특정한쌍 𝑠𝑡,𝑎𝑡 가항상동일한다음상태 𝑠𝑡+1 로이어진다는것을의미한다. 

- – 보상 **(Reward):** 포장문제의목표는 Bin의공간활용률(space ratio)을최대화하는것이다. 따라서보상은공간활용도의 단계별증가분(step-wise enhancement)으로정의되며, 다음과같이표현된다. 

**==> picture [73 x 24] intentionally omitted <==**

- 이와같은밀집보상(dense reward)은DRL 에이전트가한에피소드에서더많은단계를수행하도록유도하여, 결과적으로더많은물품을포장하고더높은공간활용률을달성하게한다. 

12/26 

**Online 3D Bin Packing** 

**IME AI Modeling** 

**==> picture [79 x 66] intentionally omitted <==**

**==> picture [79 x 66] intentionally omitted <==**

**==> picture [79 x 65] intentionally omitted <==**

**==> picture [58 x 51] intentionally omitted <==**

## Methodology – Network architecture 

- DRL 에이전트를위한신경망아키텍처설계는매우중요하다. 선택된아키텍처는에이전트의학습능력과다양한 환경에서의일반화능력에직접적인영향을미치기때문이다. 가장단순한네트워크설계방식은 Bin과물품표현 [2] 또는임베딩 [7]을단순히연결(concatenate)하는것이다. 그러나이러한방식은합성곱(convolutional) 및선형계층(linear layer)의크기가 Bin의치수에의존하게되어, 학습된모델이서로다른 Bin에적용되기어렵다는한계가있다. 이러한 일반화문제를해결하기위해, 우리는물품과 Bin의부분공간간상관관계에집중하는어텐션기반네트워크 아키텍처를제안한다. 그림 2a에서와같이, 이아키텍처는크게세가지주요구성요소로이루어져있다: Packing Transformer, 액터네트워크(actor network), 크리틱네트워크(critic network)이다. 

- 우리의네트워크는PG로부터생성된 EMS 시퀀스를나타내는 Bin 표현 𝑠 ∈ℝ[𝑁×6] 과물품의치수를나타내는물품 𝑡,𝑏𝑖𝑛 

- 표현 𝑠𝑡,𝑏𝑖𝑛 ∈ℝ[2×3] 을입력으로받는다. 이입력들은각각 MLP(Multi-Layer Perceptron)를통해개별적으로처리되며, MLP는 LeakyReLU 활성화함수를가진 2계층선형네트워크이다. EMS와물품의임베딩차원은모두 128로설정된다. 이후, 우리는언어와시각간의교차모달학습(cross-modality learning) [32]에서영감을받아설계된 Packing Transformer를통해임베딩으로부터특징(feature)을추출한다. EMS와물품특징은액터네트워크로입력되어가능한 행동들의확률분포를생성하고, 동시에크리틱네트워크로입력되어현재상태에기반한기대누적보상(expected cumulative reward)을추정한다. 

**==> picture [79 x 66] intentionally omitted <==**

**==> picture [25 x 8] intentionally omitted <==**

**----- Start of picture text -----**<br>
13/26<br>**----- End of picture text -----**<br>


**Online 3D Bin Packing** 

**IME AI Modeling** 

**==> picture [79 x 66] intentionally omitted <==**

**==> picture [79 x 66] intentionally omitted <==**

**==> picture [79 x 65] intentionally omitted <==**

**==> picture [79 x 66] intentionally omitted <==**

## Methodology – Network architecture 

– Packing Transformer는그림 2b에자세히나타나있다. 이구조는 동일한인코더블록을여러개(실제로는3개) 쌓아올려 구성되며, 각블록은두개의셀프어텐션(self-attention) 레이어, 하나의양방향교차어텐션(bi-directional cross-attention) 레이어, 그리고 {128, 128} 뉴런으로이루어진 2계층 MLP 블록 4개를 포함한다. 양방향교차어텐션레이어는두개의단방향교차 어텐션(unidirectional cross-attention)으로이루어져있으며, 하나는 EMS에서아이템(item)으로, 다른하나는아이템에서 EMS로향한다. 각레이어이후에는잔차연결(residual connection)과레이어정규화(layer normalization, Norm)가 적용된다. 셀프어텐션레이어는 EMS들간혹은아이템의 치수들간에내재적연결관계를형성하는데중요한역할을 하며, 양방향교차어텐션레이어는서로다른두표현간의 내적관계들을발견하는데기여한다. 

- 액터네트워크(actor network)와크리틱네트워크(critic network)는모두그림 2a에제시된 MLP 계층으로구현된다. 액터네트워크에서는EMS 특징과아이템특징이각각 MLP를 통해처리되며, 그결과가곱해져행동의스코어맵(score map)을계산한다. 이후, 이스코어맵은액션마스크(action mask)와원소별곱(element-wise multiplication)을수행하여실행 불가능한행동(infeasible actions)을제거한다. 

**==> picture [58 x 51] intentionally omitted <==**

14/26 

**Online 3D Bin Packing** 

**IME AI Modeling** 

**==> picture [79 x 66] intentionally omitted <==**

**==> picture [79 x 66] intentionally omitted <==**

**==> picture [79 x 65] intentionally omitted <==**

**==> picture [58 x 51] intentionally omitted <==**

## Methodology – Training method 

- 우리는제안한 GOPT를학습하기위해 Proximal Policy Optimization (PPO) 알고리즘 [33]을사용한다. PPO는널리 사용되는정책 내 (on-policy) 강화학습알고리즘으로, 환경과의상호작용을통해데이터를수집하는단계와다음의 목적함수를최적화하는단계를번갈아수행한다. 각반복(iteration)에서다음의목적함수가근사적으로최대화된다: 𝐿 𝜃= 𝔼[෠] 𝑡 ℒ[𝐶𝐿𝐼𝑃] 𝜃−𝑐1ℒ[𝑉𝐹] 𝜃+ 𝑐2𝑆 𝜋𝜃 ∙𝑠𝑡 

   - 여기서, 𝜃 는네트워크파라미터를나타내고, 𝑐1,𝑐2 는계수이다. ℒ[𝐶𝐿𝐼𝑃] 𝜃 는클리핑대리목적함수(clipped surrogate objective), ℒ[𝑉𝐹] 𝜃 는가치함수(value function)에대한제곱오차손실, 𝑆 는정책의엔트로피를의미한다. 구체적으로, 대리목적함수(surrogate objective)는다음과같이정의된다: 

**==> picture [280 x 18] intentionally omitted <==**

– 𝜋𝜃 𝑎𝑡 𝑠𝑡 여기서, 𝑝𝑡 𝜃= 𝜋𝜃𝑜𝑙𝑑 𝑎𝑡 𝑠𝑡[는현재정책과이전정책간의행동확률비율][(action probability ratio)] 

- 𝐴[መ] 𝑡 는우위함수(advantage function)의추정치로, 우리는이를계산하기위해일반화된우위 추정(Generalized Advantage Estimator, GAE) [34] 방법을사용한다. 𝜖 은클리핑비율(clipped ratio)로, 업데이트의크기를제한하고학습절차를안정화하는데사용된다. 

**==> picture [79 x 66] intentionally omitted <==**

15/26 

**Online 3D Bin Packing** 

**IME AI Modeling** 

**==> picture [79 x 66] intentionally omitted <==**

**==> picture [79 x 66] intentionally omitted <==**

**==> picture [79 x 65] intentionally omitted <==**

**==> picture [58 x 51] intentionally omitted <==**

## Methodology – PPO (Proximal Policy Optimization) 

- 정책경사(policy gradient) 방법은정책경사의추정치를계산하고이를확률적경사상승(stochastic gradient ascent) 알고리즘에대입하여작동한다.  가장일반적으로사용되는경사추정량은다음과같은형태를가진다: – Ƹ𝑔= 𝔼[෠] 𝑡[∇𝜃 log𝜋𝜃 𝑎𝑡 𝑠𝑡 𝐴[መ] 𝑡] (1) 

- 여기서 𝜋𝜃 는확률적정책(stochastic policy)이고, 𝐴[መ] 𝑡 는시간단계 𝑡 에서의어드밴티지함수(advantage function)의 추정량이다. 기대값 𝔼[෠] 𝑡[⋯] 는샘플링과최적화를번갈아수행하는알고리즘에서, 샘플들의유한한배치(batch)에대한 경험적평균을의미한다. 자동미분(automatic differentiation) 소프트웨어를사용하는구현은, 그것의기울기가정책경사 추정량이되도록목적함수를구성하여동작한다. 추정량 Ƹ𝑔 은다음의목적함수를미분함으로써얻어진다: – 𝐿[𝑃𝐺] 𝜃=𝔼[෠] 𝑡[log𝜋𝜃 𝑎𝑡 𝑠𝑡 𝐴[መ] 𝑡] (2) 

- – 같은궤적(trajectory)을사용하여이손실 𝐿[𝑃𝐺] 에대해여러번최적화를수행하는것이매력적으로보일수있지만, 이는 정당성이충분하지않으며, 경험적으로볼때종종파괴적으로큰정책업데이트들을초래한다(섹션6.1 참조; 결과는 제시되지않았지만“클리핑또는페널티없음”설정과비슷하거나더나쁜경우가많았다). 

**==> picture [79 x 66] intentionally omitted <==**

16/26 

**Online 3D Bin Packing** 

**IME AI Modeling** 

**==> picture [79 x 66] intentionally omitted <==**

## Methodology – PPO (Proximal Policy Optimization) 

**==> picture [58 x 51] intentionally omitted <==**

- TRPO [Sch+15b]에서는정책업데이트의크기에대한제약조건을두고, 목적함수(대리 목적함수)를최대화한다. 구체적으로는다음과같다. 

**==> picture [79 x 66] intentionally omitted <==**

**==> picture [79 x 65] intentionally omitted <==**

**==> picture [79 x 66] intentionally omitted <==**

**==> picture [269 x 30] intentionally omitted <==**

subject to 𝔼[෠] 𝑡[𝐾𝐿[𝜋𝜃𝑜𝑙𝑑 ∙𝑠𝑡 ,𝜋𝜃 ∙𝑠𝑡 ]] ≤𝛿 (4) – 여기서 𝜃𝑜𝑙𝑑 는업데이트이전의정책파라미터벡터를의미한다. 이문제는목적함수에대해선형근사(linear approximation)를, 제약조건에대해 2차근사(quadratic approximation)를수행한후, 켤레구배법(conjugate gradient algorithm)을사용하여효율적으로근사적으로해결할수있다. 

- TRPO를정당화하는이론은실제로제약조건대신페널티(penalty)를사용하는것을제안한다. 즉, 어떤 계수 𝛽 에 대해 다음의제약없는최적화문제를해결하는것이다. 

**==> picture [413 x 30] intentionally omitted <==**

- 이는특정한대리목적함수(surrogate objective, 평균이아닌상태들에대한최대 KL을계산)가정책 𝜋 의성능에 대한하한(즉, 비관적경계)을형성한다는사실에서비롯된다. TRPO가페널티대신엄격한제약(hard — — 

- constraint)을사용하는이유는, 서로다른문제들에서 혹은학습과정에서특성이변하는단일문제내에서도 성능이잘나오는단일 𝛽 값을선택하기가어렵기때문이다. 따라서TRPO의단조(monotonic) 성능향상을 모방하는 1차알고리즘(first-order algorithm)을목표로할때, 단순히고정된페널티계수 𝛽 를선택하고SGD로 수식 (5)의페널티목적함수를최적화하는것만으로는충분하지않으며, 추가적인수정이필요함이실험을통해 나타났다. 

17/26 

**Online 3D Bin Packing** 

**IME AI Modeling** 

**==> picture [79 x 66] intentionally omitted <==**

## Methodology – PPO (Proximal Policy Optimization) 

**==> picture [58 x 51] intentionally omitted <==**

**==> picture [425 x 15] intentionally omitted <==**

**==> picture [79 x 66] intentionally omitted <==**

**==> picture [79 x 65] intentionally omitted <==**

**==> picture [108 x 36] intentionally omitted <==**

- 그러면, TRPO는다음과같은대리(surrogate) 목적함수를최대화한다. 

**==> picture [341 x 36] intentionally omitted <==**

- 위첨자 CPI는이목적함수가제안되었던보수적정책반복(conservative policy iteration) [KL02]을나타낸다.  제약 없이 𝐿𝐶𝑃𝐼 𝜃 를최대화하면지나치게큰정책업데이트가발생할수있으므로, 이제 𝑟𝑡 𝜃 를 1에서멀어지게하는 정책변화에페널티를부여하도록목적함수를수정하는방법을고려한다. 

- – 우리가제안하는주요목적함수는다음과같다. 

**==> picture [413 x 16] intentionally omitted <==**

- 여기서 𝜖 는하이퍼파라미터이며, 예를들어 𝜖= 0.2 이다. 이목적함수의동기는다음과같다. – min 안의첫번째항은 𝐿[𝐶𝑃𝐼] 이다. 

**==> picture [79 x 66] intentionally omitted <==**

- 두번째항, clip(𝑟𝑡 𝜃,1−𝜖,1+𝜖) 𝐴[መ] 𝑡 는확률비율을클리핑(clipping)함으로써대리 목적함수를수정하며, 𝑟𝑡 가구간 [1−𝜖,1+𝜖] 밖으로이동하도록하는유인을제거한다. 

- 마지막으로클리핑된항과클리핑되지않은항의최소값을취함으로써, 최종목적함수는클리핑되지 않은목적함수에대한하한(즉, 비관적경계)이된다. 

- 이방식에서는목적함수를향상시키게할때확률비율의변화를무시하고, 목적함수를악화시키는 경우에만그것을포함시킨다. 또한, 𝐿[𝐶𝐿𝐼𝑃] 𝜃 는 𝜃𝑜𝑙𝑑 주변(즉, 𝑟 =  1 인지점)에서는1차근사에서 𝐿[𝐶𝑃𝐼] 𝜃 와 같지만, 𝜃 가 𝜃𝑜𝑙𝑑 에서멀어지면두함수는달라진다. 그림 1은 𝐿[𝐶𝐿𝐼𝑃] 에서단일항(즉, 단일 t)을나타낸 것이다. 확률비율 𝑟 은어드밴티지(advantage)가양수인지음수인지에따라 1−𝜖 또는 1+𝜖 에서 클리핑된다. 

18/26 

**Online 3D Bin Packing** 

**IME AI Modeling** 

**==> picture [79 x 66] intentionally omitted <==**

## Methodology – PPO (Proximal Policy Optimization) 

**==> picture [58 x 51] intentionally omitted <==**

**==> picture [766 x 322] intentionally omitted <==**

19/26 

**Online 3D Bin Packing** 

**IME AI Modeling** 

**==> picture [79 x 66] intentionally omitted <==**

## Methodology – PPO (Proximal Policy Optimization) 

**==> picture [58 x 51] intentionally omitted <==**

- 그림 2는대리 목적함수 𝐿[𝐶𝐿𝐼𝑃] 에대한또다른직관적이해를제공한다. 이그림은연속제어문제에서근접정책 최적화(Proximal Policy Optimization, 곧소개할알고리즘)를통해얻은정책업데이트방향을따라여러목적함수가 어떻게변하는지를보여준다. 우리는 𝐿[𝐶𝐿𝐼𝑃] 가 𝐿[𝐶𝑃𝐼] 의하한(lower bound)이며, 너무큰정책업데이트에대한페널티를 포함하고있음을확인할수있다. 

**==> picture [731 x 327] intentionally omitted <==**

20/26 

**Online 3D Bin Packing** 

**IME AI Modeling** 

**==> picture [79 x 66] intentionally omitted <==**

## Methodology – PPO (Proximal Policy Optimization) 

**==> picture [58 x 51] intentionally omitted <==**

- 그림 2는대리 목적함수 𝐿[𝐶𝐿𝐼𝑃] 에대한또다른직관적이해를제공한다. 이그림은연속제어문제에서근접정책 최적화(Proximal Policy Optimization, 곧소개할알고리즘)를통해얻은정책업데이트방향을따라여러목적함수가 어떻게변하는지를보여준다. 우리는 𝐿[𝐶𝐿𝐼𝑃] 가 𝐿[𝐶𝑃𝐼] 의하한(lower bound)이며, 너무큰정책업데이트에대한페널티를 포함하고있음을확인할수있다. 

**==> picture [731 x 327] intentionally omitted <==**

21/26 

**Online 3D Bin Packing** 

**IME AI Modeling** 

**==> picture [79 x 66] intentionally omitted <==**

**==> picture [79 x 66] intentionally omitted <==**

**==> picture [79 x 65] intentionally omitted <==**

**==> picture [79 x 66] intentionally omitted <==**

**==> picture [58 x 51] intentionally omitted <==**

## Methodology – PPO (Proximal Policy Optimization) 

- 클리핑된대리 목적함수의대안으로, 혹은그와함께사용할수있는또다른접근법은 KL 발산(KL divergence)에 페널티를부여하고, 각정책업데이트마다KL 발산이목표값 𝑑targ 에도달하도록페널티계수를조정하는것이다. 우리의실험에서는KL 페널티가클리핑된대리 목적함수보다성능이낮았지만, 중요한기준선(baseline)이므로여기서 포함했다. 

- 이알고리즘의가장단순한형태에서는, 각정책업데이트에서다음단계를수행한다. 

   - 여러에포크(epoch)들의미니배치 SGD를사용하여 KL 페널티가적용된다음의목적함수를최적화한다. 

**==> picture [449 x 30] intentionally omitted <==**

**==> picture [356 x 30] intentionally omitted <==**

**==> picture [215 x 34] intentionally omitted <==**

- 업데이트된 𝛽 는다음정책업데이트에사용된다. 이방식에서는 KL 발산이 𝑑targ 와크게다른정책업데이트가 가끔발생할수있지만, 이는드물며 𝛽 가빠르게조정된다. 위에서사용된 1.5와2라는값은경험적으로선택된 것이지만, 알고리즘은이값들에크게민감하지않다. 𝛽 의초기값또한또다른하이퍼파라미터이지만, 알고리즘이빠르게조정하기때문에실제로는중요하지않다. 

22/26 

**Online 3D Bin Packing** 

**IME AI Modeling** 

**==> picture [79 x 66] intentionally omitted <==**

**==> picture [79 x 66] intentionally omitted <==**

**==> picture [79 x 65] intentionally omitted <==**

**==> picture [79 x 66] intentionally omitted <==**

**==> picture [58 x 51] intentionally omitted <==**

## Methodology – PPO (Proximal Policy Optimization) 

- 이전섹션에서다룬대리 손실은일반적인정책경사(policy gradient) 구현에서약간의수정만으로계산하고미분할수 있다. 자동미분(automatic differentiation)을사용하는구현에서는, 단순히 𝐿[𝑃𝐺] 대신 𝐿[𝐶𝐿𝐼𝑃] 또는 𝐿[𝐾𝐿𝑃𝐸𝑁] 손실을구성하고, 이 목적함수에대해여러번의확률적경사상승(stochastic gradient ascent) 단계를수행하면된다. 

– 분산감소(variance-reduced) 어드밴티지함수추정량들을계산하는대부분의기법은학습된상태가치함수(state-value function) 𝑉(𝑠) 를사용한다. 예를들어, 일반화된어드밴티지추정(generalized advantage estimation) [Sch+15a]이나 [Mni+16]의유한-시간(finite-horizon) 추정량들이있다. 정책(policy)과가치함수(value function) 사이에서파라미터를 공유하는신경망아키텍처를사용하는경우, 정책 surrogate와가치함수오차항을결합한손실함수를사용해야한다. 또한, 이전연구 [Wil92; Mni+16]에서제안한바와같이충분한탐색을보장하기위해엔트로피보너스(entropy bonus)를 추가하여이목적함수를확장할수있다. 이항들을결합하면, 각반복(iteration)에서(근사적으로) 최대화되는다음과 같은목적함수를얻는다. 

𝐿𝐶𝐿𝐼𝑃+𝑉𝐹+𝑆𝑡 𝜃= 𝔼[෠] 𝑡[𝐿𝐶𝐿𝐼𝑃𝑡 𝜃−𝑐1𝐿𝑉𝐹𝑡 𝜃+𝑐2𝑆[𝜋𝜃] 𝑠𝑡 ] (9) 

– 𝑉𝐹 여기서 𝑐1,𝑐2 는계수(coefficient)이고, 𝑆 는엔트로피보너스(entropy bonus)를나타내며, 𝐿𝑡 는제곱오차 targ[2] 손실(squared-error loss) 𝑉𝜃 𝑠𝑡 −𝑉𝑡 이다. 

23/26 

**Online 3D Bin Packing** 

**IME AI Modeling** 

**==> picture [79 x 66] intentionally omitted <==**

**==> picture [79 x 66] intentionally omitted <==**

**==> picture [58 x 51] intentionally omitted <==**

## Methodology – PPO (Proximal Policy Optimization) 

- [Mni+16]에서널리알려지고, 순환신경망(recurrent neural network)들과함께사용하기에적합한한가지정책경사(policy gradient) 구현방식은, 정책을T 타임스텝동안실행(T는에피소드길이보다훨씬짧음)하고, 수집된샘플들을사용하여 업데이트하는방식이다. 이방식에서는T 타임스텝이후를참조하지않는어드밴티지추정량이필요하다. [Mni+16]에서사용한추정량은다음과같다. 

**==> picture [419 x 17] intentionally omitted <==**

- 여기서 𝑡 는길이T의경로(segment) 내에서 [0, T] 범위의시간인덱스를나타낸다. 

**==> picture [79 x 65] intentionally omitted <==**

- - 

- 이를일반화하면, 𝜆 절단(truncated) 일반화어드밴티지추정(generalized advantage estimation) 버전을사용할수 있으며, 𝜆= 1 일때는식 (10)과동일해진다. 

**==> picture [383 x 35] intentionally omitted <==**

**==> picture [79 x 66] intentionally omitted <==**

24/26 

**Online 3D Bin Packing** 

**IME AI Modeling** 

**==> picture [79 x 66] intentionally omitted <==**

## Methodology – PPO (Proximal Policy Optimization) 

**==> picture [58 x 51] intentionally omitted <==**

- 고정길이의궤적 segment를사용하는근접정책최적화(Proximal Policy Optimization, PPO) 알고리즘은다음과같이 설명할수있다. 각반복(iteration)마다, N개의병렬(actor) 에이전트가T 타임스텝의데이터를수집한다. 그런다음, 이 NT 타임스텝데이터에대해 surrogate 손실을구성하고, 이를미니배치 SGD로 K 에포크동안최적화한다(보통성능 향상을위해Adam [KB14]을사용). 

**==> picture [79 x 66] intentionally omitted <==**

**==> picture [692 x 188] intentionally omitted <==**

**==> picture [61 x 16] intentionally omitted <==**

**==> picture [79 x 66] intentionally omitted <==**

25/26 

**Online 3D Bin Packing** 

**IME AI Modeling** 

**==> picture [58 x 51] intentionally omitted <==**

**==> picture [79 x 66] intentionally omitted <==**

## Methodology – PPO (Proximal Policy Optimization) 

**==> picture [647 x 409] intentionally omitted <==**

**==> picture [79 x 66] intentionally omitted <==**

**==> picture [79 x 65] intentionally omitted <==**

**==> picture [79 x 66] intentionally omitted <==**

26/26 

**Online 3D Bin Packing** 

**IME AI Modeling** 

