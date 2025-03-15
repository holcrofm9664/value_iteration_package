import numpy as np

def compute_transition_probabilities(grid_height: int, grid_width: int, p: float) -> np.ndarray:
    """
    Description:
        Creates a stochastic transition probability matrix for a rectangular Gridworld.
    Inputs:
        grid_height (int): The number of rows.
        grid_width (int): The number of columns.
        p (float): The probability of moving in the intended direction.
    Output:
        transition_probabilities (np.ndarray): A (4, num_states, num_states) transition probability matrix.
    """
    num_states = grid_height * grid_width
    num_actions = 4  # the four possible actions - up, down, left and right
    P = np.zeros((num_actions, num_states, num_states))  # initialising the transition probability matrix, with shape (A, S, S')

    unintended_prob = (1 - p) / 3  # the probability of moving in each of the unintended directions

    for s in range(num_states):
        for a in range(num_actions):
            intended_next = s  # set the default movement to be to stay at the same point, which occurs if we are at a boundary
            stay_probability = 0  # initialise the probability of staying in one location

            row, col = divmod(s, grid_width)  # find the row and column number

            # define the intended movement (if not blocked by a boundary)
            if a == 0 and row > 0:  # up (if we're not in the top row)
                intended_next = s - grid_width
            if a == 1 and row < grid_height - 1:  # down (if we're not in the bottom row)
                intended_next = s + grid_width
            if a == 2 and col > 0:  # left (ife we're not in the left column)
                intended_next = s - 1
            if a == 3 and col < grid_width - 1:  # right (if we're not in the right column)
                intended_next = s + 1

            # assign the probability for the intended move
            P[a, s, intended_next] = p

            # assign unintended movements
            for a_unintended in range(num_actions):
                if a_unintended != a:  # don't include the intended move
                    next_state = s  # default: no movement
                    if a_unintended == 0 and row > 0:  
                        next_state = s - grid_width  # up
                    elif a_unintended == 1 and row < grid_height - 1:  
                        next_state = s + grid_width  # down
                    elif a_unintended == 2 and col > 0:  
                        next_state = s - 1  # left
                    elif a_unintended == 3 and col < grid_width - 1:  
                        next_state = s + 1  # right
                    else:
                        # add up each probability of moving when movement is blocked (to add to staying probability later)
                        stay_probability += unintended_prob
                        continue  # Skip assigning to P if movement is invalid

                    # assign unintended probability
                    P[a, s, next_state] = unintended_prob

            # add this to the probability of staying put
            P[a, s, s] += stay_probability

    return P

def state_index(row: int, col: int, grid_width: int) -> int:
    """
    Description:
        Converts cartesian coordinates into standard numbers
    Inputs: 
        row (int): The row number of the point
        col (int): The column number of the point
        grid_width (int): The width of the grid
    Output:
        square_index (int): The index assigned to that grid square
    """
    square_index = row * grid_width + col
    return square_index

def compute_rewards(state_rewards: dict, boundary_penalty: float, transition_probabilities: np.ndarray, A: np.ndarray, S: np.ndarray) -> np.ndarray:
    """
    Description:
        Computes the expected reward for each state-action pair in the gridworld setting
    Inputs:
        state_rewards (dict): The sqauares associated with rewards, and the reward values
        boundary_penalty (float): The penalty associated with colliding with the grid boundary
        transition_probabilities (np.ndarray): The probabilities associated with transitioning between each pair of states
        A (np.ndarray): The set of possible actions
        S (np.ndarray): The set of possible states
    Output:
        rewards (np.ndarray): The expected rewards for each state-action pair
    """
    num_states = len(S)
    num_actions = len(A)

    rewards = np.full((num_states, num_actions), 0.0, dtype=np.float64) # initialise the rewards array, using floats to avoid rounding to integers

    for s in range(num_states):
        for a in range(num_actions):
            expected_reward = 0.0
            for s_next in range(num_states):
                prob = transition_probabilities[a, s, s_next]  # probability of moving to s_next
                reward = state_rewards.get(s_next, 0)  # enter a reward if there is one, else the reward is set to zero

                # apply penalty if bumping into a boundary
                if s_next == s:  # this is equivalent to not moving
                    reward += boundary_penalty  # add the penalty

                expected_reward += prob * reward  # sum over all possible next states

            rewards[s, a] = expected_reward  # assign expected reward each state-action pair

    return rewards  # return the computed rewards array