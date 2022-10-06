# RL_study

## 1. Markov Decision Process

**Markov property : *"The future is independent of the past given the present"***

$P[S_{t+1}| S_t] = P[S_{t+1} | S_1, ..., S_t]$

### 1) Markov reward process
$<S,P,R,\gamma>$ 로 표현할 수 있다.  
* $P[S_{t+1}| S_t] = P[S_{t+1} | S_1, ..., S_t]$  
* $R_s = E[R_{t+1}|S_t=s]$ => 특정 state s 에서받을 수 있는 reward의 기댓값 확률 $p(r|s)$를 따른다.  
* Return $G_t = R_{t+1}+ \gamma R_{t+2}+... = \displaystyle\sum_{k=0}^{\infty}{\gamma ^kR_{t+k+1}}$ : time step t로부터 받을 수 있는 총 discounted reward sum
* Value function $v(s)$ : MRP에서 state s로부터 받을 수 있는 expected return
  * $v(s) = E[G_t|S_t=s] 
  \\= E[\displaystyle\sum_{k=0}^{\infty}{\gamma ^kR_{t+k+1}}|S_t=s]
  \\=E[R_{t+1}+\gamma G_{t+1}|S_t=s]
  \\=E[R_{t+1}+\gamma E[G_{t+1}|S_{t+1}=s',\cancel{S_t=s}]]$ <= Markov property
  $\\=E[R_{t+1}+\gamma v(S_{t+1})|S_t = s]$ <= <span style="color:red">Bellman equation</span> for MRP
  $\\=E[R_{t+1}]+\gamma E[v(S_{t+1})|S_t=s]
  \\=R_s+\gamma \displaystyle\sum_{s'\in S}{p(s'|s)v(s')}$
### 2) Markov decision process
$<S,A,P,R,\gamma >$ 로 표현할 수 있다.
* State transition probability : $P_{ss'}^a=p(s'|s,a)$
* Action is following policy $\pi (a|s)$
* $R_s^a=E[R_{t+1}|S,A] = \displaystyle\sum_{a}{\pi(a|s)} \displaystyle\sum_{r}{p(r|s,a)r}=\displaystyle\sum_{a} \displaystyle\sum_{r}{p(r,a|s)r}=\displaystyle\sum_{r}{rp(r|s)}$
* Action value function $q_\pi (s,a)$: MDP에서 state s에 action a를 취한 뒤 이후 policy $\pi$를 따를 때 받을 수 있는 expected return  
  * $q(s,a)=E[G_t|S_t=s,A_t=a]
  \\=E[R_{t+1}+ \gamma q_\pi (S_{t+1},A_{t+1})|S_t=s,A_t=a]$ <= <span style="color:red">Bellman equation </span> for MDP  
    $=R_s^\pi+\gamma \displaystyle\sum_{s' \in S}{p(s'|s,a)}\displaystyle\sum_{a'}{\pi(a'|s')}q_\pi (s',a')$

### 3) $q(s,a)$와 $v(s)$의 관계
* $v_\pi (s) = E[G_t|S_t=s]=\displaystyle\sum_{a}{\pi (a|s)}E[G_t|S_t=s,A_t=a]
  \\=\displaystyle\sum_{a}{\pi (a|s)}q_\pi (s,a)$
* $q_\pi (s,a)= E[R_{t+1}]+\gamma \displaystyle\sum_{s'}{p(s'|s,a)}v(s')$

## 2. 