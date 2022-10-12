# RL_study

## 1. Markov Decision Process

**Markov property : *"The future is independent of the past given the present"***

$P[S_{t+1}| S_t] = P[S_{t+1} | S_1, ..., S_t]$

### 1) Markov reward process  
$(S,P,R,\gamma)$의 tuple로 표현할 수 있다.  
* $P[S_{t+1}| S_t] = P[S_{t+1} | S_1, ..., S_t]$  
* $R_s = E[R_{t+1}|S_t=s]$ => 특정 state s 에서받을 수 있는 reward의 기댓값 확률 $p(r|s)$를 따른다.  
* Return $G_t = R_{t+1}+ \gamma R_{t+2}+... = \displaystyle\sum_{k=0}^{\infty}{\gamma ^kR_{t+k+1}}$ : time step t로부터 받을 수 있는 총 discounted reward sum
* Value function $v(s)$ : MRP에서 state s로부터 받을 수 있는 expected return
  * $v(s) = E[G_t|S_t=s]$  
  $= E[\displaystyle\sum_{k=0}^{\infty}{\gamma ^kR_{t+k+1}}|S_t=s]$  
  $=E[R_{t+1}+\gamma G_{t+1}|S_t=s]$  
  $=E[R_{t+1}+\gamma E[G_{t+1}|S_{t+1}=s',\cancel{S_t=s}]]$ <= Markov property  
  $=E[R_{t+1}+\gamma v(S_{t+1})|S_t = s]$ <= <span style="color:red">Bellman equation</span> for MRP  
  $=E[R_{t+1}]+\gamma E[v(S_{t+1})|S_t=s]$  
  $=R_s+\gamma \displaystyle\sum_{s'\in S}{p(s'|s)v(s')}$
### 2) Markov decision process
$(S,A,P,R,\gamma )$의 tuple로 표현할 수 있다.
* State transition probability : $P_{ss'}^a=p(s'|s,a)$
* Action is following policy $\pi (a|s)$
* $R_s^a=E[R_{t+1}|S,A] = \displaystyle\sum_{a}{\pi(a|s)} \displaystyle\sum_{r}{p(r|s,a)r}=\displaystyle\sum_{a} \displaystyle\sum_{r}{p(r,a|s)r}=\displaystyle\sum_{r}{rp(r|s)}$
* Action value function $q_\pi (s,a)$: MDP에서 state s에 action a를 취한 뒤 이후 policy $\pi$를 따를 때 받을 수 있는 expected return  
  * $q(s,a)=E[G_t|S_t=s,A_t=a]$  
  $=E[R_{t+1}+ \gamma q_\pi (S_{t+1},A_{t+1})|S_t=s,A_t=a]$ <= <span style="color:red">Bellman equation </span> for MDP    
  $=R_s^\pi+\gamma \displaystyle\sum_{s' \in S}{p(s'|s,a)}\displaystyle\sum_{a'}{\pi(a'|s')}q_\pi (s',a')$  

### 3) $q(s,a)$와 $v(s)$의 관계
* $v_\pi (s) = E[G_t|S_t=s]=\displaystyle\sum_{a}{\pi (a|s)}E[G_t|S_t=s,A_t=a]=\displaystyle\sum_{a}{\pi (a|s)}q_\pi (s,a)$  
* $q_\pi (s,a)= E[R_{t+1}]+\gamma \displaystyle\sum_{s'}{p(s'|s,a)}v(s')$

## 2. Planning by Dynamic programming

Dynamic programming : 복잡한 문제를 푸는 최적화 기법으로 다음 두 가지 특성을 만족한다.
- Optimal substructure
  - 전체 문제의 optimal solution이 subproblem의 optimal solution들로 분해된다.
- Overlapping subproblems
  - 각 subproblem은 recursive한 형태로 나타낼 수 있다.
  - 각 iteration에서의 solution은 캐쉬로 저장할 수 있고 재사용 가능하다.
- MDP는 위의 두 특성을 모두 만족하여 dynamic programming으로 최적해를 구할 수 있다.
  - Bellman equation은 recursive decomposition 특성을 가짐
  - Value function은 저장되어 solution으로 재사용될 수 있음
  - For prediction(evaluation) : given policy $\pi$에 의해
    - input : MDP tuple ($S, A, P, R, \gamma$)
    - output : value function $v_{\pi}$ or action value function $q_{\pi}(s,a)$
  - For control(최적해):
    - input : MDP tuple ($S, A, P, R, \gamma$)
    - output : optimal value function $v_{\pi}$ or optimal policy $\pi$
### 1) Iterative policy evaluation
Problem : given policy $\pi$ 로 예측되는 $v_{\pi}$를 만드는 것  
- $v_{k+1}(s) = \sum_{a \in A}{\pi(a|s)}(R_s^a+\gamma \sum_{s' \in S}{p(s'|s,a)v_k(s')})$
- k번째 iteration의 state value $v_k(s')$로 $v_{k+1}(s)$를 구할 수 있다.
- Iteration을 통해 given policy $\pi$에 수렴하는 $v_{\pi}$를 구할 수 있다.
### 2) Policy iteration
Policy evaluation과 greedy action을 통해 policy를 improve하는 과정을 반복하면  
optimal policy $\pi^*$와 optimal value function $v^*$를 구할 수 있다.
- policy evaluation : 1)에서의 iterative policy evaluation을 통해 구함
- policy improvement : generate $\pi ' \ge \pi$ by greedy policy improvement
  - $\pi'(s) = \argmax_{a \in A}{q_\pi(s,a)}$
  - 모든 state s에 대해서 
    - $q_\pi(s, \pi'(s)) = \max_{a \in A}{q_\pi(s,a)} = v_{\pi'}(s)$
    - 즉 $v_{\pi'}(s) \ge v_\pi(s)$로 improve 할 수 있다.
### 3) Value iteration
$v_{k+1}(s) = \max_{a \in A}{R_s^a + \gamma \sum_{s' \in S}{p(s'|s,a)v_k(s')}}$

## 3. Model-free prediction
**Dynamic programming** 에서는 state transition probability p(s'|s,a)를 알아야 문제를 해결할 수 있지만 대부분의 MDP에서 state transition에 대한 정보를 완벽하게 알기가 어렵다.  
따라서 p(s'|s,a)를 모르더라도 문제를 해결할 수 있는 model-free의 방법이 유용하다.
