# POLICY ITERATION ALGORITHM

## AIM
The aim of this experiment is to implement the Policy Iteration Algorithm in Reinforcement Learning to determine the optimal policy and corresponding value function for a given environment. Policy Iteration combines iterative policy evaluation and policy improvement steps to achieve convergence towards an optimal policy.

## PROBLEM STATEMENT
In Reinforcement Learning, the agent interacts with an environment modeled as a Markov Decision Process (MDP). The challenge is to find an optimal policy that maximizes the long-term cumulative reward. Policy Iteration addresses this by:

Evaluating the value of a given policy (Policy Evaluation). Improving the policy based on the evaluated value function (Policy Improvement). Repeating these steps until the policy converges to the optimal policy.

## POLICY ITERATION ALGORITHM
### 1.Initialize Policy:
Start with a random policy by assigning an action to each state.
### 2.Policy Evaluation:
Compute the value of each state by following the current policy.
Update the values repeatedly until the change between iterations becomes very small.
### 3.Policy Improvement:
For each state, choose the action that gives the highest expected reward based on the current value function.
### 4.Policy Stability Check:
Compare the updated policy with the previous one.<br>
=>If the policy does not change, stop the process.<br>
=>If it changes, repeat the evaluation and improvement steps.
### 5.Output
The final policy is the optimal policy, and the corresponding state values form the optimal value function.

## POLICY IMPROVEMENT FUNCTION
### Name : Nandakesore J
### Register Number : 212223240103
```python
def policy_improvement(V, P, gamma=1.0):
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)

    for s in range(len(P)):
        for a in range(len(P[s])):
            for prob, next_state, reward, done in P[s][a]:
                Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))

    new_pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]

    return new_pi
```
## POLICY ITERATION FUNCTION
### Name : Nandakesore J
### Register Number : 212223240103
```python
def policy_iteration(P,gamma=1.0,theta=1e-10):
  random_actions=np.random.choice(tuple(P[0].keys()),len(P))
  pi=lambda s: {s:a for s, a in enumerate(random_actions)}[s]
  while True:
    old_pi={s: pi(s) for s in range(len(P))}
    V=policy_evaluation(pi,P,gamma,theta)
    pi=policy_improvement(V,P,gamma)
    if old_pi=={s:pi(s) for s in range(len(P))}:
      break
  return V,pi
```

## OUTPUT:
### 1. Policy, Value function and success rate for the Adversarial Policy
<img width="851" height="287" alt="image" src="https://github.com/user-attachments/assets/359ea09d-769f-4482-b12c-ab7f6d907915" />
</br>

### 2. Policy, Value function and success rate for the Improved Policy
<img width="847" height="292" alt="image" src="https://github.com/user-attachments/assets/b66b5e85-8998-4276-b614-ac79e9ca993f" />
</br>
<img width="534" height="384" alt="image" src="https://github.com/user-attachments/assets/57b7a356-49da-4289-88a1-ea3cb82d6746" />
</br>

### 3. Policy, Value function and success rate after policy iteration
<img width="849" height="316" alt="image" src="https://github.com/user-attachments/assets/1c4fa092-caf3-40e0-ace3-4beed4b6db11" />
</br>
<img width="890" height="148" alt="image" src="https://github.com/user-attachments/assets/d4edf622-3b2a-4f93-89a2-98e7e88e3eb8" />
</br>

## RESULT:

The Policy Iteration algorithm successfully converged to an optimal policy for the Frozen Lake environment.
The optimal policy achieved a higher success rate and improved average return compared to the initial policy.
