# A Value Iteration Algorithm for Approximately Solving Markov Decision Processes (MDPs)
The Value_Iteration package provides an algorithm for approximately solving a range of Markov Decision Processes. 


## Installing from github with pip:

```bash
pip install git+https://github.com/holcrofm9664/value_iteration_package.git
```

## Example 1: Solving the Health Party Problem
```python
import numpy as np
from Value_Iteration.iteration import value_iteration

S = np.array(["H", "I"])  # states
A = np.array(["R", "P"])  # actions

transition_probabilities = np.array([
    [[0.95, 0.05], [0.5, 0.5]],  # action "R", transitioning to "H" or "I"
    [[0.7, 0.3], [0.1, 0.9]]  # action "P", transitioning to "H" or "I"
])

rewards = np.array([
    [7, 10],  # state "H", given we choose "R" or "P"
    [0, 2]    # state "I", given we choose "R" or "P"
])

policy, V = value_iteration(S, A, transition_probabilities, rewards, gamma=0.9, theta=0.001)

print("Optimal Policy:", policy)
print("Value Function:", V)
```
The package also includes a flexible way of easily constructing and rectangular gridworld problem for solving with the value iteration algorithm. The user needs only to input the grid height, the grid width, the number of possible actions, the penalty for hitting the boundary, and the reward sizes and coordinates. The package can transform this data into the correct form for inputting into the value iteration algorithm. An example for a 10x10 grid is shown below in Example 2.

## Example 2: The n x m gridworld problem
```python
import numpy as np
from Value_Iteration.iteration import value_iteration
from Value_Iteration.gridworld import compute_transition_probabilities, compute_rewards, state_index

# initialise values
grid_height = 10
grid_width = 10
num_states = grid_height * grid_width
num_actions = 4  # up, down, left, right
boundary_penalty = -1.0  # penalty for bumping into walls

# define the states and actions
S = np.arange(num_states)  # state space
A = np.arange(num_actions)  # actions: 0 = up, 1 = down, 2 = left, 3 = right

# create transition probability matrix (choosing the probability of moving in the intended direction = 0.7)
transition_probabilities = compute_transition_probabilities(grid_height, grid_width, p=0.7)

# define non-zero state rewards and their coordinated (going from 0)
state_rewards = {
    state_index(2, 7, grid_width): 3, # a positive, goal reward
    state_index(4, 3, grid_width): -5, # a negative penalty/obstacle
    state_index(7, 3, grid_width): -10,
    state_index(7, 8, grid_width): 10
}

# compute the rewards using the function
rewards = compute_rewards(state_rewards, boundary_penalty, transition_probabilities, A, S)

# run value iteration
optimal_policy = value_iteration(S, A, transition_probabilities, rewards, threshold=0.0001, max_iterations=1000)

# display results
optimal_policy
