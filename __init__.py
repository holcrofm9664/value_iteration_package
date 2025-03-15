"""
Value_Iteration_Package

This package provides an implementation of the value iteration algorithm, including for the rectangular gridworld problem.
"""

__version__ = "0.0.0"

from .iteration import value_iteration  # imports the value iteration function
from .bellman import Bellmans_Eq # imports the Bellman's equation for use in the value iteration function
from .gridworld import compute_transition_probabilities, state_index, compute_rewards # imports the functions relevant fr the gridworld problem
