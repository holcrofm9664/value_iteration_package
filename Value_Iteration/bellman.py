import numpy as np
from typing import Union

def Bellmans_Eq(s: int, S: np.ndarray, A: np.ndarray, V: np.ndarray, P: np.ndarray, R: np.ndarray, D: float) -> Union[float, float]:
    """
    Description:
        A function that runs a value update for a single state using Bellman's equation
    Inputs:
        s - The state we are applying the update to
        S - The set of all states
        A - The set of all actions we can take
        V - The set of all current state values
        P - The transition probabilities between states
        R - The rewards associated with each state
        D - The dicount factor applied to future rewards
    Output:
        best_action - The action that leads to the greatest value
        best_value - The best value computed for this state, when taking the best action
    """
    V_a = np.zeros(len(A)) # initialise the values for each action as zeros
    for a in range(len(A)):
        V_a[a] = R[s][a] + D*sum(P[a][s][s1]*V[s1] for s1 in range(len(S))) # calculate the value associated with each value
        best_value = max(V_a) # find the best value so far
        best_action = np.where(V_a == best_value)[0][0] # find the action associated with that value
    return best_action, best_value # return the best action and its associated value