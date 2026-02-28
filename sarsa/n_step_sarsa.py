import numpy as np
import random

# Globals:
ACTIONS = ("up", "down", "left", "right") 

# Rewards, terminals and obstacles are characters:
REWARDS = {" ": -1, ".": 0.1, "+": 100, "-": -100}
TERMINALS = ("+", "-") # Note a terminal should also have a reward assigned
OBSTACLES = ("#")

# Discount factor
gamma = 1

# The probability of a random move:
rand_move_probability = 0

class World:  
  def __init__(self, width, height):
    self.width = width
    self.height = height
    # Create an empty world where the agent can move to all cells
    self.grid = np.full((width, height), ' ', dtype='U1')
  
  def add_obstacle(self, start_x, start_y, end_x=None, end_y=None):
    """
    Create an obstacle in either a single cell or rectangle.
    """
    if end_x == None: end_x = start_x
    if end_y == None: end_y = start_y
    
    self.grid[start_x:end_x + 1, start_y:end_y + 1] = OBSTACLES[0]

  def add_reward(self, x, y, reward):
    assert reward in REWARDS, f"{reward} not in {REWARDS}"
    self.grid[x, y] = reward

  def add_terminal(self, x, y, terminal):
    assert terminal in TERMINALS, f"{terminal} not in {TERMINALS}"
    self.grid[x, y] = terminal

  def is_obstacle(self, x, y):
    if x < 0 or x >= self.width or y < 0 or y >= self.height:
      return True
    else:
      return self.grid[x ,y] in OBSTACLES 

  def is_terminal(self, x, y):
    return self.grid[x ,y] in TERMINALS

  def get_reward(self, x, y):
    """ 
    Return the reward associated with a given location
    """ 
    return REWARDS[self.grid[x, y]]

  def get_next_state(self, current_state, action):
    """
    Get the next state given a current state and an action. The outcome can be
    stochastic  where rand_move_probability determines the probability of 
    ignoring the action and performing a random move.
    """    
    assert action in ACTIONS, f"Unknown acion {action} must be one of {ACTIONS}"
    
    x, y = current_state 
    
    # If our current state is a terminal, there is no next state
    if self.grid[x, y] in TERMINALS:
      return None

    # Check of a random action should be performed:
    if np.random.rand() < rand_move_probability:
      action = np.random.choice(ACTIONS)

    if action == "up":      y -= 1
    elif action == "down":  y += 1
    elif action == "left":  x -= 1
    elif action == "right": x += 1

    # If the next state is an obstacle, stay in the current state
    return (x, y) if not self.is_obstacle(x, y) else current_state


def n_step_sarsa(world, start_state, n, Q_table=None, alpha=0.5, gamma=1.0, epsilon=0.1, episodes=1000, max_steps=1000):
    if Q_table is None:
        Q_table = np.full((world.width, world.height, len(ACTIONS)), 0.0) # Q_table[x, y, a_idx] #
    # initialize_q_table(world, Q_table) #
    all_steps = []    
    policy = lambda state, q_func=Q_table : {k : 1-epsilon+epsilon/len(ACTIONS) #
                                             if k == ACTIONS[np.argmax(q_func[*state, :])] 
                                             else epsilon/len(ACTIONS) for k in ACTIONS}

    for episode in range(episodes):

        # Ring buffers (size n+1)
        S = [None] * (n + 1)
        A = [None] * (n + 1)
        R = [0.0] * (n + 1)

        S[0] = start_state
        action_prob = policy(start_state)
        action = random.choices(population=list(action_prob.keys()),
                                    weights=action_prob.values(), k=1)[0]
        A[0] = ACTIONS.index(action)

        T = float("inf")
        t = 0
        steps = 0

        while True:

            if t < T and steps < max_steps:
                action_idx = A[t % (n + 1)]
                action = ACTIONS[action_idx]

                next_state = world.get_next_state(S[t % (n + 1)], action)
                reward = world.get_reward(*next_state)
                R[(t + 1) % (n + 1)] = reward
                S[(t + 1) % (n + 1)] = next_state

                if world.is_terminal(*next_state):
                    T = t + 1
                else:
                    action_prob = policy(next_state)
                    action = random.choices(population=list(action_prob.keys()),
                                                weights=action_prob.values(), k=1)[0]
                    A[(t + 1) % (n + 1)] = ACTIONS.index(action)

            tau = t - n + 1 

            if tau >= 0: # enters at t = 3 if n = 4

                G = 0.0
                for i in range(tau + 1, min(tau + n, T) + 1): # +1 to include T
                    G += (gamma ** (i - tau - 1)) * R[i % (n + 1)] # roughly: sum for gamma^i * R_1

                if tau + n < T: # if not end reached
                    s = S[(tau + n) % (n + 1)]
                    a = A[(tau + n) % (n + 1)]
                    G += (gamma ** n) * Q_table[*s, a] # G + q-value for tau + n'th step

                s_tau = S[tau % (n + 1)] # s to est.
                a_tau = A[tau % (n + 1)] # a to est.

                Q_table[*s_tau, a_tau] += alpha * (G - Q_table[*s_tau, a_tau]) # est.

            if tau == T - 1:
                break

            t += 1
            steps += 1
        all_steps.append(steps)

    return Q_table, all_steps

def q_visualizer(world : World, q : np.ndarray) -> None:
    new_q = np.full((world.width, world.height), "", dtype=object)
    action_symbols = {
        'up': '↑',
        'down': '↓', 
        'left': '←',
        'right': '→'
    }
    for x in range(world.width):
        for y in range(world.height):
            if world.is_terminal(x, y):
                new_q[x, y] = '+'
            elif world.is_obstacle(x, y):
                new_q[x, y] = '#'
            else:
                new_q[x, y] = action_symbols[ACTIONS[np.argmax(q[x, y, :])]]

class CliffWorld(World):
    def __init__(self):
        super().__init__(width=12, height=4)

        # ---- Start and Goal ----
        self.start_state = (0, 0)
        self.goal_state  = (11, 0)

        # Goal is a terminal with positive reward
        self.add_terminal(*self.goal_state, "+")   # +10 reward from REWARDS

        # ---- Create the cliff ----
        # bottom row y = 0, cells 1..10
        self.cliff_cells = {(x, 0) for x in range(1, 11)}

    def is_cliff(self, x, y):
        return (x, y) in self.cliff_cells

    def get_next_state(self, current_state, action):
        """
        Same movement logic as World,
        but stepping into the cliff teleports to start.
        """
        next_state = super().get_next_state(current_state, action)

        # If terminal, base class returns None
        if next_state is None:
            return None

        x, y = next_state

        # --- Cliff behavior ---
        if self.is_cliff(x, y):
            return self.start_state   # teleport

        return next_state

    def get_reward(self, x, y):
        """
        Override reward to handle cliff penalty.
        """
        if self.is_cliff(x, y):
            return -100
        return super().get_reward(x, y)

def main(): 
   cliff_world = World(4, 12)


if __name__ == '__main__':
   main()