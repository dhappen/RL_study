# RL_study

## 1. Markov Decision Process

**Markov property : *"The one step future is only dependent on the present"***

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
* Action value function $q_\pi (s,a)$: MDP에서 state s에 a7uq2 ction a를 취한 뒤 이후 policy $\pi$를 따를 때 받을 수 있는 expected return  
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
    - input : MDP tuple ( $S, A, P, R, \gamma$ )  
    - output : value function $v_{\pi}$ or action value function $q_{\pi}(s,a)$
  - For control(최적해):
    - input : MDP tuple ( $S, A, P, R, \gamma$ )  
    - output : optimal value function $v_{\pi}$ or optimal policy $\pi$
### 1) Iterative policy evaluation
Problem : given policy $\pi$ 로 예측되는 $v_{\pi}$를 만드는 것  
- $v_{k+1}(s) = \sum_{a \in A}{\pi(a|s)}(R_s^a+\gamma \sum_{s' \in S}{p(s'|s,a)v_k(s')})$
- k번째 iteration의 state value $v_k(s')$로 $v_{k+1}(s)$를 구할 수 있다.
- Iteration을 통해 given policy $\pi$에 수렴하는 $v_{\pi}$를 구할 수 있다.
### 2) Policy iteration
Policy evaluation과 greedy action을 통해 policy를 improve하는 과정을 반복하면  
optimal policy $\pi^{ * }$와 optimal value function $v^{ * }$를 구할 수 있다.
- policy evaluation : 1)에서의 iterative policy evaluation을 통해 구함
- policy improvement : generate $\pi ' \ge \pi$ by greedy policy improvement
  - $\pi'(s) = arg\max_{a \in A}{q_\pi(s,a)}$
  - 모든 state s에 대해서 
    - $q_\pi(s, \pi '(s)) = \max_{a \in A}{q_\pi(s,a)} = v_{\pi '}(s)$
    - 즉 $v_{\pi '}(s) \ge v_\pi(s)$로 improve 할 수 있다.
### 3) Value iteration
$v_{k+1}(s) = \max_{a \in A}{R_s^a + \gamma \sum_{s' \in S}{p(s'|s,a)v_k(s')}}$

## 3. Model-free prediction
**Dynamic programming** 에서는 state transition probability $p(s'|s,a)$ 를 알아야 문제를 해결할 수 있지만 대부분의 MDP에서 state transition에 대한 정보를 완벽하게 알기가 어렵다.  
따라서 $p(s'|s,a)$ 를 모르더라도 문제를 해결할 수 있는 model-free의 방법이 유용하다.  
Bootstrapping : 추정값(value function)을 사용하여 update하는 것  
Sampling : 기댓값을 update하기 위해 sample을 사용하는 것
### 1) Monte-Carlo 
Episodic task에만 적용 가능하다. (전체 return을 사용해서 update하기 때문)
- Monte-Carlo Policy Evaluation
  - $v_{\pi}(s_t) = \lim_{N\to \infty}\frac{1}{N}\sum_{t}{G_t}$
  - 대수의 법칙을 이용하여 mean return으로 return의 기댓값인 value function을 표현할 수 있다.
  - Incremental mean (running mean)

    - $\mu_k = \frac{1}{k} \displaystyle\sum_{j=1}^{k}{x_j} = \frac{1}{k}(x_k + \displaystyle\sum_{j=1}^{k-1}{x_j})$     

    $\mu_k = \mu_{k-1}+\frac{1}{k}{x_k-\mu_{k-1}}$
    - $v(s_t) \gets v(s_t)+\frac{1}{N(s_t)}(G_t-v(s_t))$ 를 통해서 sampling을 이용한 policy evaluation이 가능하다.
    - $v(s_t) \gets v(s_t)+\alpha(G_t-v(s_t))$ 
### 2) Temporal-Difference Learning
TD methods를 통해서는 episode 전체를 수행하지 않아도 update를 진행할 수 있다.  
Episode가 완전히 끝나지 않더라도 online으로 $v_\pi$를 evaluation할 수 있다.
- TD(0)
  - $v(s_t) \gets v(s_{t})+\alpha(R_{t+1}+\gamma v(s_{t+1})-v(s_t))$
  - TD target = $R_{t+1} + \gamma v(s_{t+1})$
  - TD error : $\delta_t = R_{t+1}+\gamma v(s_{t+1}) - v(s_t)$ 
- Bias/Variance
  - Return $G_t$ 를 구성하는 값들은 $v_\pi (s_t)$와는 관련이 없기 때문에 이에 biased되지 않는다.  
  하지만 많은 random한 action, transition, reward가 포함되기 때문에 variance가 높다.
  - 반면 TD target $\delta_t = R_{t+1}+\gamma v(s_{t+1}) - v(s_t)$ 은 $v(s_{t+1})$ 에 bias되고 random한 action , transition, reward가 하나씩만 포함되기 때문에 variance는 작다.
- TD$( \lambda )$
  - n-step return
    - $n = 1$ 일 때 $G_t^{(1)}=R_{t+1}+\gamma v(s_{t+1})$
    - $n = 2$ 일 때 $G_t^{(2)}=R_{t+1}+\gamma R_{t+2}+ \gamma ^2v(s_{t+2})$
    - $G_t^{(n)} = R_{t+1}+\gamma R_{t+2}+ ... + \gamma ^n v(s_{t+n})$
  - n-step TD learning
    - $v(s_t) \gets v(s_t) + \alpha(G_t^{(n)} - v(s_t))$
  - Forward view of TD$( \lambda )$
    -  n-step return $G_t^{(n)}$ 을 weight $(1- \lambda )\lambda^{n-1}$로 combine하여 표현
    -  $G_t^{\lambda} = (1-\lambda) \displaystyle\sum_{n=1}^{\infty}{\lambda^{n-1}G_t^{(n)}}$
    -  TD$(1)$은 terminal state가 있는 경우 MC와 같다.

## 4. Model-free control
앞선 3. 에서는 model-free 에서 prediction을 하는 것 즉 policy $\pi$를 따르는 value function을 찾는 과정이었다. 기존의 policy iteration은 최적의 policy를 찾지만 model-based의 방법으로 transition probability가 필요한 과정이었다.  
Sampling을 통해 model-free 상황에서 최적의 policy를 찾아나가는 과정이 model-free control이다.

### 1) Introduction
- On and off policy learning
  - On policy
    - Learn about policy $\pi$ from experience sampled from  $\pi$
    - $\pi$를 통해 sample 된 trajectory로 $\pi$를 update하는 method
  - Off policy
    - Learn about policy $\pi$ from experience sampled from $\mu$
- $\epsilon$-Greedy exploration
  - Greedy 하게만 policy를 update하면 bias가 심하고 local minima에 빠질 수 있다.
  - $1 - \epsilon$ 의 확률로 greedy action을 택하고 $\epsilon$의 확률로 random action을 택하는 policy 이다. 
  - $\pi (a|s) = \begin{cases} \frac{\epsilon}{m}+1 - \epsilon, &
   \textrm{if}\;  a^*=\textrm{arg}\max_{a \in A}{Q(s,a)} \\ 
   \frac{\epsilon}{m}, & 
   \textrm{otherwise} \end{cases}$  
  - $\epsilon$-Greedy policy improvement theorem
    - For any $\epsilon$-greedy policy $\pi$, the $\epsilon$-greedy policy $\pi$' with respect to $q_\pi$ is improvement,  
    $v_{\pi'}(s) \geq v_{\pi}(s)$  
### 2) Monte-carlo control 
- Policy evaluation $\to$ MC policy evaluation, $q(s_t,a_t) \gets v(s_t,a_t)+\alpha(G_t-v(s_t,a_t))$ 
- Policy improvement $\to$ $\epsilon$-greedy policy improvement  
$\epsilon \gets \frac{1}{k}$, $\pi \gets \epsilon \textrm{-greedy}(Q)$
- $\epsilon$-Greedy policy improvement theorem을 통해 policy에 randomness가 추가되더라도 optimal로 수렴하더라

### 3) On policy Temporal-difference(TD) learning
- Policy evaluation $\to$ Sarsa, $Q \approx q_\pi$
  - Sarsa : $Q(s,a) \gets Q(s,a) + \alpha (R+\gamma Q(s',a') - Q(s,a))$
- Policy iteration $\to$ $\epsilon$-greedy policy improvement
### 4) Off-policy control with Q-learning
- Policy evaluation : $Q(s,a) \gets Q(s,a) + \alpha (R+\gamma \max_a{Q(s',a)} - Q(s,a))$
- Behavior policy 는 $\epsilon$-greedy policy 지만 target policy는 Q에 대해 greedy한 policy이다.
- $\pi(s_{t+1}) = \textrm{arg}\max_{a'}{Q(s_{t+1},a')}$ : target policy