
import numpy as np
from random import shuffle
from time import time, sleep
from collections import deque

from settings import s, e

from keras.models import Sequential
from keras.layers import Dense, Activation, InputLayer, Flatten
from keras.utils import plot_model, to_categorical
from keras.models import load_model

AGENT_NAME = "deepQ_agent"

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

    self.inputarena = 9

    #neural network with Keras
    self.logger.debug(self)
    self.model = Sequential()
    self.model.add(Dense(10,input_shape=(self.inputarena,self.inputarena))) ##############
    self.model.add(Flatten())
    self.model.add(Dense(10, activation='sigmoid')) #, input_shape=(31,31)
    self.model.add(Dense(6, activation='linear'))
    self.model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    self.hyperpar = {"y":0.4, "eps": 0.8, "lr":0.2, "training_decay":0.99, "mini_batch_size":1000}    #y, eps, learning rate, decay factor alpha
    #not yet sure where the decay factor goes

    self.train = True #####################needs to be changed

    self.visualize_convergence = []

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

def create_input_arena(self):
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

    arena_copy = self.game_state['arena']
    for i in range(len(coins)):
        arena_copy[coins[i]] = 2
    for i in range(len(bomb_xys)):
        arena_copy[bomb_xys[i]] = 3
    for i in range(len(others)):
        arena_copy[others[i]] = 4
    arena_copy[x,y] = 5
    #print(arena_copy.T)

    input_size, arena_size = self.inputarena, 17
    center = int((input_size-1)/2)
    center_off = [x-center,y-center]
    grid_off = int((input_size-1)/2)

    input = -1*np.ones((input_size,input_size))
    arena_x, arena_y = np.linspace(0,arena_size-1,arena_size,dtype=int), np.linspace(0,arena_size-1,arena_size,dtype=int)
    proj = [(xi,yi) for xi in arena_x for yi in arena_y if np.any(abs(xi-x)<=grid_off) &  np.any(abs(yi-y)<=grid_off)]
    for (xi,yi) in proj:
        input[xi-center_off[0],yi-center_off[1]] = arena_copy[xi,yi]
    return input

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
    input = create_input_arena(self)
    #print(input.T)

    # If agent has been in the same location three times recently, it's a loop
    if self.coordinate_history.count((x,y)) > 2:
        self.ignore_others_timer = 5
    else:
        self.ignore_others_timer -= 1
    self.coordinate_history.append((x,y))

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
                self.next_action = np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB', 'WAIT'], p=[.23, .23, .23, .23, .08, 0.0])
            else:
                a = np.argmax(self.model.predict(input.reshape(-1,self.inputarena,self.inputarena)))
                #print(a)
                self.next_action = number_to_actions[a]
    else:
        a = np.argmax(self.model.predict(input.reshape(-1,self.inputarena,self.inputarena)))
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
            target = r + self.hyperpar["y"] * np.max(self.model.predict(new_s.reshape(-1,self.inputarena,self.inputarena)))
            target_vec = self.model.predict(s.reshape(-1,self.inputarena,self.inputarena))[0]
            target_vec[a] = target
            self.model.fit(s.reshape(-1,self.inputarena,self.inputarena), target_vec.reshape(-1,6), epochs=1, verbose=0)

    #save the point where the experience for this game ends
    self.episodes.append(len(self.experience))

    #total reward this game
    start = self.episodes[-2]
    end = self.episodes[-1]
    reward = 0
    for i in range(start, end):
        reward += self.experience[i][2]
    _, _, _,_,game_reward = self.game_state['self']
    print(str(reward)+" and game rewards: "+str(game_reward)+" after step "+str(self.game_state["step"]))
    '''
    if(game_reward == 9):
        self.model.save_weights("agent_code/"+AGENT_NAME+"/"+AGENT_NAME+'.h5')
    '''

    _, _, _,_,game_reward = self.game_state['self']
    self.visualize_convergence.append(game_reward)

    a = np.asarray(self.visualize_convergence)
    np.savetxt(AGENT_NAME+"_rewards.csv", a, delimiter=",")
