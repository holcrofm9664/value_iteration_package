import numpy as np
from typing import Union
from .bellman import Bellmans_Eq

def value_iteration(S: np.ndarray, A: np.ndarray, transition_probabilities: np.ndarray, rewards: np.ndarray, threshold: float, max_iterations: int) -> Union[np.ndarray, np.ndarray]:
    """
    Description:
        A function that runs value iteration to find an approximately optimal policy for a Markov Decision Process (MDP)
    Inputs:
        S - The set of all states
        A - The set of all actions
        P - The transition probabilities P(s'|s,a)
        R - The rewards associated with each state action pair R(s,a)
    Output:
        Policy - The approximately optimal policy
        V - The value function
    """   
    V = np.zeros(len(S)) # initialise the values as zeros for each state
    a_opt = np.full(len(S), -np.inf) # initialise the actions (as minus infinity as this is not contained in the action set)
    k = 0 # initialise the counter
    changes = np.full(len(S), np.inf) # initialise the changes (to monitor convergence)
    while max(changes) > threshold and k < max_iterations: # run repeatedly until the maximum value change is below our threshold or the counter exceeds the maximum number of iterations
        Values_Stored = V.copy() # store the values
        k = k + 1 # move the counter on
        for s in range(len(S)):
            a_max, V_max = Bellmans_Eq(s = s, S = S, A = A, V = V, P = transition_probabilities, R = rewards, D = 0.8) # compute the value update using Bellman's equation
            V[s] = V_max
            a_opt[s] = a_max
        Values_new = V
        changes = Values_new - Values_Stored # find the amount by which the values have changed
    Policy = a_opt # retrieve the approximately optimal policy
    return Policy, V