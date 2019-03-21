''' This seems to be obselete, since it is effectively implemented already...
def encode_actions(action):
    '''Converts an action in the form of a string to a sparse array (see one 
    hot encoding). Reverses the decode_actions function.'''
    actions_to_number = {"UP":0, "RIGHT":1, "DOWN":2, "LEFT":3, "BOMB": 4, "WAIT": 5}
    int_actions = np.arange(np.len(actions_to_number))
    enc_actions = to_categorical(int_actions)
    return encoded = enc_actions[actions_to_number(action)]

def decode_actions(encoded):
    '''Converts an encoded action in the form of a sparse array to a string
    that is accepted by the game. Reverses the encode_actions function.'''
    number_to_actions = {0:"UP", 1:"RIGHT", 2:"DOWN", 3:"LEFT", 4:"BOMB", 5:"WAIT"}
    return action = number_to_actions(np.argmax(encoded))
'''
