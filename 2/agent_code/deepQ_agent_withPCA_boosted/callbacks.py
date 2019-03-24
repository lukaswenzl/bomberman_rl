
import numpy as np
from random import shuffle
from time import time, sleep
from collections import deque

from settings import s, e

import pickle

from keras.models import Sequential
from keras.layers import Dense, Activation, InputLayer, Flatten
from keras.utils import plot_model, to_categorical
from keras.models import load_model

from sklearn.decomposition import PCA

AGENT_NAME = "deepQ_agent_withPCA_boosted"


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


    self.training_input = 60

    #neural network with Keras
    self.logger.debug(self)
    self.model = Sequential()
    self.model.add(Dense(10,input_shape=(self.training_input,))) ##############
    #self.model.add(Flatten())##############!
    self.model.add(Dense(10, activation='sigmoid')) #, input_shape=(31,31)
    self.model.add(Dense(6, activation='linear'))
    self.model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    self.hyperpar = {"y":0.4, "eps": 0.9999, "lr":0.2, "training_decay":0.9999, "mini_batch_size":1000}    #y, eps, learning rate, decay factor alpha
    #not yet sure where the decay factor goes

    self.train = True #####################needs to be changed

    self.visualize_convergence = []

    filename = 'PCA.save'
    # load the model from disk
    self.pca = pickle.load(open(filename, 'rb'))
    filename = 'PCA_full_field.save'
    self.pca_full_field = pickle.load(open(filename, 'rb'))

    self.boost_count = 0






def check_actions(self):
    self.logger.info('Picking action according to rule set')

    # Gather information about the game state
    arena = self.game_state['arena']
    x, y, _, bombs_left, score = self.game_state['self']
    bombs = self.game_state['bombs']
    bomb_xys = [(x,y) for (x,y,t) in bombs]
    others = [(x,y) for (x,y,n,b,s) in self.game_state['others']]
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
            (not d in bomb_xys)): #and (self.coordinate_history.count(d) > 2)
            valid_tiles.append(d)
    if (x-1,y) in valid_tiles: valid_actions.append('LEFT')
    if (x+1,y) in valid_tiles: valid_actions.append('RIGHT')
    if (x,y-1) in valid_tiles: valid_actions.append('UP')
    if (x,y+1) in valid_tiles: valid_actions.append('DOWN')
    if (x,y)   in valid_tiles: valid_actions.append('WAIT')
    # Disallow the BOMB action if agent dropped a bomb in the same spot recently
    if (bombs_left > 0) and (x,y) not in self.bomb_history: valid_actions.append('BOMB')
    self.logger.debug(f'Valid actions: {valid_actions}')
    return valid_actions

def create_input_arena(self, in_size=9):
    arena = self.game_state['arena']
    x, y, _, bombs_left, score = self.game_state['self']
    bombs = self.game_state['bombs']
    bomb_xys = [(x,y) for (x,y,t) in bombs]
    others = [(x,y) for (x,y,n,b,s) in self.game_state['others']]
    coins = self.game_state['coins']
    bomb_map = np.ones(arena.shape) * 5
    for xb,yb,t in bombs:
        for (i,j) in [(xb+h, yb) for h in range(-3,4)] + [(xb, yb+h) for h in range(-3,4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i,j] = min(bomb_map[i,j], t)

    arena_copy = self.game_state['arena'].copy()
    for i in range(len(coins)):
        arena_copy[coins[i]] = 2
    for i in range(len(bomb_xys)):
        arena_copy[bomb_xys[i]] = 3
    for i in range(len(others)):
        arena_copy[others[i]] = 4
    arena_copy[x,y] = 5
    #print(arena_copy.T)

    input_size, arena_size = in_size, 17
    center = int((input_size-1)/2)
    center_off = [x-center,y-center]
    grid_off = int((input_size-1)/2)

    input = -1*np.ones((input_size,input_size))
    arena_x, arena_y = np.linspace(0,arena_size-1,arena_size,dtype=int), np.linspace(0,arena_size-1,arena_size,dtype=int)
    proj = [(xi,yi) for xi in arena_x for yi in arena_y if np.any(abs(xi-x)<=grid_off) &  np.any(abs(yi-y)<=grid_off)]
    for (xi,yi) in proj:
        input[xi-center_off[0],yi-center_off[1]] = arena_copy[xi,yi]

    input = input+1 #now numbers going from 0 to 6
    input = input.reshape(-1) #reshape to 1d array
    input = input.astype(int) #We want integers
    b = np.zeros((input.size, 7))
    b[np.arange(input.size), input] = 1

    b = b.reshape(-1)

    return b


    '''arena_x, arena_y = np.linspace(0,16,17), np.linspace(0,16,17)
    arena_xv, arena_yv = np.meshgrid(arena_x,arena_y)
    arena_xv, arena_yv = arena_xv.astype(int), arena_yv.astype(int)
    input = -1*np.ones((31,31))
    new_arena_xv, new_arena_yv = arena_xv+7+center_off[0], arena_yv+7+center_off[1]
    input[new_arena_xv,new_arena_yv] = arena_copy[arena_xv, arena_yv]'''


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

    #Load weights at this point?

    # Gather information about the game state
    arena = self.game_state['arena']
    x, y, _, bombs_left, score = self.game_state['self']
    bombs = self.game_state['bombs']
    bomb_xys = [(x,y) for (x,y,t) in bombs]
    others = [(x,y) for (x,y,n,b,s) in self.game_state['others']]
    coins = self.game_state['coins']
    bomb_map = np.ones(arena.shape) * 5
    for xb,yb,t in bombs:
        for (i,j) in [(xb+h, yb) for h in range(-3,4)] + [(xb, yb+h) for h in range(-3,4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i,j] = min(bomb_map[i,j], t)

    '''
    #Build modified arena as input for NN
    #put all the information into arena shaped array
    arena_copy = self.game_state['arena']
    for i in range(len(coins)):
        arena_copy[coins[i]] = 2
    for i in range(len(bomb_xys)):
        arena_copy[bomb_xys[i]] = 3
    for i in range(len(others)):
        arena_copy[others[i]] = 4
    arena_copy[x,y] = 5
    #print(arena_copy.T)

    center = [8,8]
    center_off = np.subtract(center,[x,y])
    grid_off = [7,7]
    arena_x, arena_y = np.linspace(0,16,17), np.linspace(0,16,17)
    arena_xv, arena_yv = np.meshgrid(arena_x,arena_y)
    arena_xv, arena_yv = arena_xv.astype(int), arena_yv.astype(int)
    input = -1*np.ones((31,31))
    new_arena_xv, new_arena_yv = arena_xv+7+center_off[0], arena_yv+7+center_off[1]
    input[new_arena_xv,new_arena_yv] = arena_copy[arena_xv, arena_yv]
    #print('new_arena')
    '''
    b = create_input_arena(self,  in_size=9)

    #applying PCA
    input1 = self.pca.transform([b])[0]
    b = create_input_arena(self,  in_size=33)
    input2 = self.pca_full_field.transform([b])[0]
    input = np.concatenate([input1,input2])
    #print(input.T)

    # If agent has been in the same location three times recently, it's a loop
    if self.coordinate_history.count((x,y)) > 2:
        self.ignore_others_timer = 5
    else:
        self.ignore_others_timer -= 1
    #self.coordinate_history.append((x,y)) #can not run this twice

    # jan: Determine which actions are valid
    valid_actions = check_actions(self)

    # Compile a list of 'targets' the agent should head towards
    crates = [(x,y) for x in range(1,16) for y in range(1,16) if (arena[x,y] == 1)]
    targets = coins + crates
    # Add other agents as targets if in hunting mode or no crates/coins left
    if self.ignore_others_timer <= 0 or (len(crates) + len(coins) == 0):
        targets.extend(others)
    '''
    # Exclude targets that are currently occupied by a bomb
    targets = [targets[i] for i in range(len(targets)) if targets[i] not in bomb_xys]
    '''

    '''
    if(coins == []):
        next_goal = [0,0]
    else:
        next_goal = coins[0]

    input = np.array([[next_goal[0]-x, next_goal[1]-y]])
    '''

    self.next_action = np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB', 'WAIT'], p=[.23, .23, .23, .23, .08, 0.0])

    number_to_actions = {0:"UP", 1:"RIGHT", 2:"DOWN", 3:"LEFT", 4:"BOMB", 5:"WAIT"}

    a = -1
    # select the action with highest cummulative reward
    if(self.game_state["train"]):
            self.hyperpar["eps"] =self.hyperpar["eps"]*self.hyperpar["training_decay"]
            if np.random.random() < self.hyperpar["eps"]:
                #self.next_action = np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB', 'WAIT'], p=[.23, .23, .23, .23, .08, 0.0])
                #follow idea of simple agent
                self.next_action = follow_simple_agent(self)
            else:
                a = np.argmax(self.model.predict(input.reshape(1, self.training_input)))
                #print(a)
                self.next_action = number_to_actions[a]
    else:
        a = np.argmax(self.model.predict(input.reshape(1, self.training_input)))
        self.next_action = number_to_actions[a]

    while ((self.next_action not in valid_actions) and (valid_actions != [])): #not future proof
        if(valid_actions==['WAIT']):
            self.next_action = 'WAIT'
        else:
            self.next_action = np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB', 'WAIT'], p=[.23, .23, .23, .23, .08, 0.0])

    # Keep track of chosen action for cycle detection
    if self.next_action == 'BOMB':
        self.bomb_history.append((x,y))

    actions_to_number = {"UP":0, "RIGHT":1, "DOWN":2, "LEFT":3, "BOMB": 4, "WAIT": 5}
    reward = 0
    self.experience.append([input, actions_to_number[self.next_action], reward])

def follow_simple_agent(self):
    """For training use ideas of simple agent some of the time"""
    self.boost_count = self.boost_count+1
    self.logger.info('Picking action according to rule set')

    # Gather information about the game state
    arena = self.game_state['arena']
    x, y, _, bombs_left, score = self.game_state['self']
    bombs = self.game_state['bombs']
    bomb_xys = [(x,y) for (x,y,t) in bombs]
    others = [(x,y) for (x,y,n,b,s) in self.game_state['others']]
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
    action_ideas = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    shuffle(action_ideas)


    # Compile a list of 'targets' the agent should head towards
    dead_ends = [(x,y) for x in range(1,16) for y in range(1,16) if (arena[x,y] == 0)
                    and ([arena[x+1,y], arena[x-1,y], arena[x,y+1], arena[x,y-1]].count(0) == 1)]
    #print(arena)
    crates = [(x,y) for x in range(1,16) for y in range(1,16) if (arena[x,y] == 1)]
    targets = coins + dead_ends + crates
    # Add other agents as targets if in hunting mode or no crates/coins left
    if self.ignore_others_timer <= 0 or (len(crates) + len(coins) == 0):
        targets.extend(others)

    # Exclude targets that are currently occupied by a bomb
    targets = [targets[i] for i in range(len(targets)) if targets[i] not in bomb_xys]

    # Take a step towards the most immediately interesting target
    free_space = arena == 0
    if self.ignore_others_timer > 0:
        for o in others:
            free_space[o] = False
    d = look_for_targets(free_space, (x,y), targets, self.logger)
    if d == (x,y-1): action_ideas.append('UP')
    if d == (x,y+1): action_ideas.append('DOWN')
    if d == (x-1,y): action_ideas.append('LEFT')
    if d == (x+1,y): action_ideas.append('RIGHT')
    if d is None:
        self.logger.debug('All targets gone, nothing to do anymore')
        action_ideas.append('WAIT')

    # Add proposal to drop a bomb if at dead end
    #print(x)

    if (x,y) in dead_ends:
        #print("reached")
        action_ideas.append('BOMB')
    # Add proposal to drop a bomb if touching an opponent
    if len(others) > 0:
        if (min(abs(xy[0] - x) + abs(xy[1] - y) for xy in others)) <= 1:
            action_ideas.append('BOMB')
    # Add proposal to drop a bomb if arrived at target and touching crate
    if d == (x,y) and ([arena[x+1,y], arena[x-1,y], arena[x,y+1], arena[x,y-1]].count(1) > 0):
        action_ideas.append('BOMB')

    # Add proposal to run away from any nearby bomb about to blow
    for xb,yb,t in bombs:
        if (xb == x) and (abs(yb-y) < 4):
            # Run away
            if (yb > y): action_ideas.append('UP')
            if (yb < y): action_ideas.append('DOWN')
            # If possible, turn a corner
            action_ideas.append('LEFT')
            action_ideas.append('RIGHT')
        if (yb == y) and (abs(xb-x) < 4):
            # Run away
            if (xb > x): action_ideas.append('LEFT')
            if (xb < x): action_ideas.append('RIGHT')
            # If possible, turn a corner
            action_ideas.append('UP')
            action_ideas.append('DOWN')
    # Try random direction if directly on top of a bomb
    for xb,yb,t in bombs:
        if xb == x and yb == y:
            action_ideas.extend(action_ideas[:4])

    # Pick last action added to the proposals list that is also valid
    #action = np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB', 'WAIT'], p=[.23, .23, .23, .23, .08, 0.0])
    while len(action_ideas) > 0:
        a = action_ideas.pop()
        if a in valid_actions:
            action = a
            break

    # Keep track of chosen action for cycle detection
    # if self.next_action == 'BOMB':
    #     self.bomb_history.append((x,y))

    return action




def reward_update(self):
    """Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occured during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state. In
    contrast to act, this method has no time limit.
    """
    self.logger.debug(f'Encountered {len(self.events)} game event(s)')
    #self.logger.debug((self.events == 9).count())

    if(e.CRATE_DESTROYED in self.events):
        ev, count = np.unique(self.events,return_counts=True)
        N_destroyed = count[ev==9][0]
        current_step = self.experience[-1]
        current_step[2] += 1*N_destroyed
        self.experience[-1] = current_step #reward 0.1 per destroyed crate.
        #CAUTION: would also give reward, if other agent destroyed a crate!

    if(e.COIN_FOUND in self.events):
        current_step = self.experience[-1]
        current_step[2] += 5
        self.experience[-1] = current_step #reward 1 for finding a coin

    if(e.COIN_COLLECTED in self.events):
        current_step = self.experience[-1]
        current_step[2] += 10
        self.experience[-1] = current_step #reward 2 for collecting a coin

    if(e.INVALID_ACTION in self.events):
        current_step = self.experience[-1]
        current_step[2] += -10
        self.experience[-1] = current_step #This should never be able to happen

    if(e.KILLED_SELF in self.events):
        current_step = self.experience[-1]
        current_step[2] += -5
        self.experience[-1] = current_step #This should never be able to happen


def end_of_episode(self):
    """Called at the end of each game to hand out final rewards and do training.

    This is similar to reward_update, except it is only called at the end of a
    game. self.events will contain all events that occured during your agent's
    final step. You should place your actual learning code in this method.
    """
    self.logger.debug(f'Encountered {len(self.events)} game event(s) in final step')



    if(e.COIN_COLLECTED in self.events):
        current_step = self.experience[-1]
        current_step[2] += 10
        self.experience[-1] = current_step #reward 2 for collecting a coin

    if(e.SURVIVED_ROUND in self.events):
        current_step = self.experience[-1]
        current_step[2] += 0
        self.experience[-1] = current_step #reward 2 for collecting a coin

    if(e.KILLED_SELF in self.events):
        current_step = self.experience[-1]
        current_step[2] += -5
        self.experience[-1] = current_step #This should never be able to happen

    current_step = self.experience[-1]
    current_step[2] += -0.05*self.game_state["step"]
    self.experience[-1] = current_step #This should never be able to happen

    if(self.game_state["train"]):
        #might break if training size is to small at the beginning
        mini_batch = self.hyperpar["mini_batch_size"]
        if(mini_batch >= len(self.experience)):
            mini_batch = len(self.experience)-1
        for i in np.random.randint(len(self.experience)-1, size=mini_batch):
            s, a, r = self.experience[i]
            new_s, new_a, new_r = self.experience[i+1]
            target = r + self.hyperpar["y"] * np.max(self.model.predict(new_s.reshape(1, self.training_input)))
            target_vec = self.model.predict(s.reshape(1, self.training_input))[0]
            target_vec[a] = target
            self.model.fit(s.reshape(1, self.training_input), target_vec.reshape(-1,6), epochs=1, verbose=0)

    #save the point where the experience for this game ends
    self.episodes.append(len(self.experience))

    #total reward this game
    start = self.episodes[-2]
    end = self.episodes[-1]
    reward = 0
    for i in range(start, end):
        reward += self.experience[i][2]
    _, _, _,_,game_reward = self.game_state['self']
    print(str(reward)+" and game rewards: "+str(game_reward)+" after step "+str(self.game_state["step"])+" boosted: "+str(self.boost_count))
    self.boost_count = 0
    '''
    if(game_reward == 9):
        self.model.save_weights("agent_code/"+AGENT_NAME+"/"+AGENT_NAME+'.h5')
    '''

    _, _, _,_,game_reward = self.game_state['self']
    self.visualize_convergence.append(game_reward)

    a = np.asarray(self.visualize_convergence)
    np.savetxt(AGENT_NAME+"_rewards.csv", a, delimiter=",")
