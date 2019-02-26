
import numpy as np
from random import shuffle
from time import time, sleep
from collections import deque

from settings import s
from settings import e

from keras.models import Sequential
from keras.layers import Dense, Activation, InputLayer


def look_for_targets(free_space, start, targets, logger=None):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    """
    if len(targets) == 0: return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x,y) for (x,y) in [(x+1,y), (x-1,y), (x,y+1), (x,y-1)] if free_space[x,y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger: logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]


def setup(self):
    """Called once before a set of games to initialize data structures etc.

    The 'self' object passed to this method will be the same in all other
    callback methods. You can assign new properties (like bomb_history below)
    here or later on and they will be persistent even across multiple games.
    You can also use the self.logger object at any time to write to the log
    file for debugging (see https://docs.python.org/3.7/library/logging.html).
    """
    self.logger.debug('Successfully entered setup code')
    np.random.seed()
    # Fixed length FIFO queues to avoid repeating the same actions
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0

    self.experience  = []# Lukas: will contain arrays with [[s,a,r],[s,a,r],...]
    self.episodes = [0] #ranges for the different games
    #self.q_matrix = np.diag([1,1,1,1])
    #self.q_matrix = np.zeros((4,4))#
    self.q_matrix = np.random.rand(4,4) #random Initializing

    #neural network with Keras
    self.model = Sequential()
    self.model.add(InputLayer(batch_input_shape=(1, 4))) ##############
    self.model.add(Dense(10, activation='sigmoid'))
    self.model.add(Dense(4, activation='linear'))
    self.model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    self.hyperpar = {"y":0.4, "eps": 0.5, "lr":0.2, "training_decay":0.99, "mini_batch_size":100}    #y, eps, learning rate, decay factor alpha
    #not yet sure where the decay factor goes

    trace = np.trace(self.q_matrix)
    su = np.sum(self.q_matrix)
    off_diag = su -trace
    difference = trace - off_diag
    self.visualize_convergence = [difference]


def act(self):
    """Called each game step to determine the agent's next action.

    You can find out about the state of the game environment via self.game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    Set the action you wish to perform by assigning the relevant string to
    self.next_action. You can assign to this variable multiple times during
    your computations. If this method takes longer than the time limit specified
    in settings.py, execution is interrupted by the game and the current value
    of self.next_action will be used. The default value is 'WAIT'.
    """
    self.logger.info('Picking action according to rule set')


    # Gather information about the game state
    arena = self.game_state['arena']
    x, y, _, bombs_left,_ = self.game_state['self']
    bombs = self.game_state['bombs']
    bomb_xys = [(x,y) for (x,y,t) in bombs]
    others = [(x,y) for (x,y,n,b, s) in self.game_state['others']]
    coins = self.game_state['coins']
    bomb_map = np.ones(arena.shape) * 5
    for xb,yb,t in bombs:
        for (i,j) in [(xb+h, yb) for h in range(-3,4)] + [(xb, yb+h) for h in range(-3,4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i,j] = min(bomb_map[i,j], t)

    # If agent has been in the same location three times recently, it's a loop
    if self.coordinate_history.count((x,y)) > 2:
        self.ignore_others_timer = 5
    else:
        self.ignore_others_timer -= 1
    self.coordinate_history.append((x,y))

    # Check which moves make sense at all
    directions = [(x,y), (x+1,y), (x-1,y), (x,y+1), (x,y-1)]
    valid_tiles, valid_actions = [], []
    for d in directions:
        if ((arena[d] == 0) and
            (self.game_state['explosions'][d] <= 1) and
            (bomb_map[d] > 0) and
            (not d in others) and
            (not d in bomb_xys)):
            valid_tiles.append(d)
    if (x-1,y) in valid_tiles: valid_actions.append('LEFT')
    if (x+1,y) in valid_tiles: valid_actions.append('RIGHT')
    if (x,y-1) in valid_tiles: valid_actions.append('UP')
    if (x,y+1) in valid_tiles: valid_actions.append('DOWN')
    if (x,y)   in valid_tiles: valid_actions.append('WAIT')
    # Disallow the BOMB action if agent dropped a bomb in the same spot recently
    if (bombs_left > 0) and (x,y) not in self.bomb_history: valid_actions.append('BOMB')
    self.logger.debug(f'Valid actions: {valid_actions}')

    # Collect basic action proposals in a queue
    # Later on, the last added action that is also valid will be chosen
    # action_ideas = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    # shuffle(action_ideas)

    # Compile a list of 'targets' the agent should head towards
    # dead_ends = [(x,y) for x in range(1,16) for y in range(1,16) if (arena[x,y] == 0)
    #                 and ([arena[x+1,y], arena[x-1,y], arena[x,y+1], arena[x,y-1]].count(0) == 1)]
    # crates = [(x,y) for x in range(1,16) for y in range(1,16) if (arena[x,y] == 1)]
    targets = coins #+ dead_ends + crates
    # Add other agents as targets if in hunting mode or no crates/coins left
    # if self.ignore_others_timer <= 0 or (len(crates) + len(coins) == 0):
    #     targets.extend(others)

    # Exclude targets that are currently occupied by a bomb
    # targets = [targets[i] for i in range(len(targets)) if targets[i] not in bomb_xys]

    # Take a step towards the most immediately interesting target
    free_space = arena == 0
    if self.ignore_others_timer > 0:
        for o in others:
            free_space[o] = False

    state = 0
    d = look_for_targets(free_space, (x,y), targets, self.logger)
    #encoding see report step one
    if d == (x,y-1): state =0 #action_ideas.append('UP')
    if d == (x,y+1): state =2#action_ideas.append('DOWN')
    if d == (x-1,y): state =3#action_ideas.append('LEFT')
    if d == (x+1,y): state =1#action_ideas.append('RIGHT')
    # if d is None:
    #     self.logger.debug('All targets gone, nothing to do anymore')
    #     action_ideas.append('WAIT')

    # Add proposal to drop a bomb if at dead end
    # if (x,y) in dead_ends:
    #     action_ideas.append('BOMB')
    # Add proposal to drop a bomb if touching an opponent
    # if len(others) > 0:
    #     if (min(abs(xy[0] - x) + abs(xy[1] - y) for xy in others)) <= 1:
    #         action_ideas.append('BOMB')
    # Add proposal to drop a bomb if arrived at target and touching crate
    # if d == (x,y) and ([arena[x+1,y], arena[x-1,y], arena[x,y+1], arena[x,y-1]].count(1) > 0):
    #     action_ideas.append('BOMB')

    # Add proposal to run away from any nearby bomb about to blow
    # for xb,yb,t in bombs:
    #     if (xb == x) and (abs(yb-y) < 4):
    #         # Run away
    #         if (yb > y): action_ideas.append('UP')
    #         if (yb < y): action_ideas.append('DOWN')
    #         # If possible, turn a corner
    #         action_ideas.append('LEFT')
    #         action_ideas.append('RIGHT')
    #     if (yb == y) and (abs(xb-x) < 4):
    #         # Run away
    #         if (xb > x): action_ideas.append('LEFT')
    #         if (xb < x): action_ideas.append('RIGHT')
    #         # If possible, turn a corner
    #         action_ideas.append('UP')
    #         action_ideas.append('DOWN')
    # # Try random direction if directly on top of a bomb
    # for xb,yb,t in bombs:
    #     if xb == x and yb == y:
    #         action_ideas.extend(action_ideas[:4])
    #
    # # Pick last action added to the proposals list that is also valid
    # while len(action_ideas) > 0:
    #     a = action_ideas.pop()
    #     if a in valid_actions:
    #         self.next_action = a
    #         break

    # # Keep track of chosen action for cycle detection
    # if self.next_action == 'BOMB':
    #     self.bomb_history.append((x,y))

    self.next_action = np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB'], p=[.25, .25, .25, .25, .0])

    number_to_actions = {0:"UP", 1:"RIGHT", 2:"DOWN", 3:"LEFT"}

    a = -1
    # select the action with highest cummulative reward
    self.hyperpar["eps"] =self.hyperpar["eps"]*self.hyperpar["training_decay"]
    if np.random.random() < self.hyperpar["eps"]:
        self.next_action = np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB'], p=[.25, .25, .25, .25, .0])
    else:
        a = np.argmax(self.model.predict(np.identity(4)[state:state + 1]))
        self.next_action = number_to_actions[a]

    while (self.next_action not in valid_actions): #not future proof
        self.next_action = np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB'], p=[.25, .25, .25, .25, .0])


    #if self.train_flag.is_set(): would have to chain trianing flag into this but then I would have to change
    #the original code. so far it will safe a lot of data. Unclear how to best avoid
    actions_to_number = {"UP":0, "RIGHT":1, "DOWN":2, "LEFT":3}
    reward = 0
    if(state != actions_to_number[self.next_action]):
        reward = -0.01
    self.experience.append([state, actions_to_number[self.next_action], reward])

    #store state

def reward_update(self):
    """Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occured during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state. In
    contrast to act, this method has no time limit.
    """
    self.logger.debug(f'Encountered {len(self.events)} game event(s)')

    #translate events to text
    #print(self.events)

    if(e.COIN_COLLECTED in self.events):
        current_step = self.experience[-1]
        current_step[2] = 1
        self.experience[-1] = current_step #reward 1 for finding a coin
        #print(self.experience)


def end_of_episode(self):
    """Called at the end of each game to hand out final rewards and do training.

    This is similar to reward_update, except it is only called at the end of a
    game. self.events will contain all events that occured during your agent's
    final step. You should place your actual learning code in this method.
    """
    self.logger.debug(f'Encountered {len(self.events)} game event(s) in final step')

    #imrpove Q matrix
    #might break if training size is to small at the beginning
    for i in np.random.randint(len(self.experience)-1, size=self.hyperpar["mini_batch_size"]):
        s, a, r = self.experience[i]
        new_s, new_a, new_r = self.experience[i+1]
        #self.q_matrix[s, a] += r + self.hyperpar["lr"] * (self.hyperpar["y"] * np.max(self.q_matrix[new_s, :]) - self.q_matrix[s, a])

        target = r + self.hyperpar["y"] * np.max(self.model.predict(np.identity(4)[new_s:new_s + 1]))
        target_vec = self.model.predict(np.identity(4)[s:s + 1])[0]
        target_vec[a] = target
        self.model.fit(np.identity(4)[s:s + 1], target_vec.reshape(-1, 4), epochs=1, verbose=0)

    #print(self.q_matrix)
    #print(len(self.visualize_convergence))

    #measure for convergence; now i have to reconstruct the q_matrix, not usefull
    # trace = 0
    # off_diag = 0
    # for i in range(4):
    #     row_of_q = self.model.predict(np.identity(4)[i:i + 1]).reshape(-1)
    #     trace += row_of_q[i]
    #     off = [j for j in range(4) if j!=i]
    #     for j in off:
    #         off_diag += row_of_q[j]

    self.episodes.append(len(self.experience))

    #total reward this game
    start = self.episodes[-2]
    end = self.episodes[-1]
    reward = 0
    for i in range(start, end):
        reward += self.experience[i][2]
    print(reward)



    # trace = np.trace(self.q_matrix)
    # su = np.sum(self.q_matrix)
    # off_diag = su -trace
    difference = trace - off_diag
    self.visualize_convergence.append(difference)
    print(difference)

    # if(len(self.visualize_convergence)==51):
    #     print(np.array(self.visualize_convergence))
    #     np.savetxt("data_for_visualisations/step1sdfdggtdbr_qmatrix.out", np.array(self.visualize_convergence))
