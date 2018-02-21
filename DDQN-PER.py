# DDQN - PER
# -------------------
#
# This code demonstrates a Double DQN network with Priority Experience Replay
# in an OpenGym environment (any game).
# it is based on AI-Blog code (forked), with additional implementation of:
#  - command-line arguments
#  - any game can be chosen
#  - model load/save
#  - memory load/save
# 
# Usage: DDQN-PER -g <game_name> (optional, defaults='KungFuMaster') --learn --render
#
# author: Jaromir Janisch, 2016
# author: David Renaudie, 2018

import random, numpy, math, gym, scipy
from SumTree import SumTree
import scipy.misc
import os
import pickle
import argparse
from keras.models import load_model
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

#-------------------- PARAMETERS -----------------------
DEFAULT_GAME = 'KungFuMaster'

MAX_EPSILON = 1
MIN_EPSILON = 0.1
FORCED_EPSILON = 0.1

IMAGE_WIDTH = 84
IMAGE_HEIGHT = 84
IMAGE_STACK = 2

HUBER_LOSS_DELTA = 2.0
LEARN_RATE = 0.00025

MEMORY_CAPACITY = 200000

BATCH_SIZE = 32

GAMMA = 0.99

EXPLORATION_STOP = 500000   # at this step epsilon will be 0.01
LAMBDA = - math.log(0.01) / EXPLORATION_STOP  # speed of decay

UPDATE_TARGET_FREQUENCY = 10000

#-------------------- UTILITIES -----------------------
def huber_loss(y_true, y_pred):
    err = y_true - y_pred

    cond = K.abs(err) < HUBER_LOSS_DELTA
    L2 = 0.5 * K.square(err)
    L1 = HUBER_LOSS_DELTA * (K.abs(err) - 0.5 * HUBER_LOSS_DELTA)

    loss = tf.where(cond, L2, L1)   # Keras does not cover where function in tensorflow :-(

    return K.mean(loss)

def processImage( img ):
    rgb = scipy.misc.imresize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), interp='bilinear')

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b     # extract luminance

    o = gray.astype('float32') / 128 - 1    # normalize
    return o

#-------------------- BRAIN ---------------------------
class Brain:
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.model = self._createModel()
        self.model_ = self._createModel()  # target network

    def _createModel(self):
        model = Sequential()

        model.add(Conv2D(32, (8, 8), strides=(4,4), activation='relu', input_shape=(self.stateCnt), data_format='channels_first'))
        model.add(Conv2D(64, (4, 4), strides=(2,2), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(units=512, activation='relu'))

        model.add(Dense(units=actionCnt, activation='linear'))

        opt = RMSprop(lr=LEARN_RATE)
        model.compile(loss=huber_loss, optimizer=opt)

        return model

    def train(self, x, y, epochs=1, verbose=0):
        self.model.fit(x, y, batch_size=32, epochs=epochs, verbose=verbose)

    def predict(self, s, target=False):
        if target:
            return self.model_.predict(s)
        else:
            return self.model.predict(s)

    def predictOne(self, s, target=False):
        return self.predict(s.reshape(1, IMAGE_STACK, IMAGE_WIDTH, IMAGE_HEIGHT), target).flatten()

    def updateTargetModel(self):
        self.model_.set_weights(self.model.get_weights())

#-------------------- MEMORY --------------------------
class Memory:   # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample) 

    def sample(self, n):
        batch = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append( (idx, data) )

        return batch

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)

#-------------------- AGENT ---------------------------
class Agent:
    #steps = 0
    #epsilon = MAX_EPSILON

    def __init__(self, stateCnt, actionCnt, epsilon=MAX_EPSILON, steps=0):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt
        self.steps = steps
        self.epsilon = epsilon
        self.brain = Brain(stateCnt, actionCnt)
        self.memory = Memory(MEMORY_CAPACITY)
        
    def act(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, self.actionCnt-1)
        else:
            return numpy.argmax(self.brain.predictOne(s))

    def observe(self, sample):  # in (s, a, r, s_) format
        x, y, errors = self._getTargets([(0, sample)])
        self.memory.add(errors[0], sample)

        if self.steps % UPDATE_TARGET_FREQUENCY == 0:
            self.brain.updateTargetModel()

        # slowly decrease Epsilon based on our experience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def _getTargets(self, batch):
        no_state = numpy.zeros(self.stateCnt)

        states = numpy.array([ o[1][0] for o in batch ])
        states_ = numpy.array([ (no_state if o[1][3] is None else o[1][3]) for o in batch ])

        p = agent.brain.predict(states)

        p_ = agent.brain.predict(states_, target=False)
        pTarget_ = agent.brain.predict(states_, target=True)

        x = numpy.zeros((len(batch), IMAGE_STACK, IMAGE_WIDTH, IMAGE_HEIGHT))
        y = numpy.zeros((len(batch), self.actionCnt))
        errors = numpy.zeros(len(batch))
        
        for i in range(len(batch)):
            o = batch[i][1]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]
            
            t = p[i]
            oldVal = t[a]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * pTarget_[i][ numpy.argmax(p_[i]) ]  # double DQN

            x[i] = s
            y[i] = t
            errors[i] = abs(oldVal - t[a])

        return (x, y, errors)

    def replay(self):    
        batch = self.memory.sample(BATCH_SIZE)
        x, y, errors = self._getTargets(batch)

        #update errors
        for i in range(len(batch)):
            idx = batch[i][0]
            self.memory.update(idx, errors[i])

        self.brain.train(x, y)

class RandomAgent:
    memory = Memory(MEMORY_CAPACITY)
    exp = 0

    def __init__(self, actionCnt):
        self.actionCnt = actionCnt

    def act(self, s):
        return random.randint(0, self.actionCnt-1)

    def observe(self, sample):  # in (s, a, r, s_) format
        error = abs(sample[2])  # reward
        self.memory.add(error, sample)
        self.exp += 1

    def replay(self):
        pass

#-------------------- ENVIRONMENT ---------------------
class Environment:
    def __init__(self, problem):
        self.problem = problem
        self.env = gym.make(problem)

    def run(self, agent, learning=True, rendering=False):                
        img = self.env.reset()
        w = processImage(img)
        s = numpy.array([w, w])

        R = 0
        while True:         
            a = agent.act(s)

            r = 0
            img, r, done, info = self.env.step(a)
            s_ = numpy.array([s[1], processImage(img)]) #last two screens

            r = np.clip(r, -1, 1)   # clip reward to [-1, 1]

            if done: # terminal state
                s_ = None

            if learning:
                agent.observe( (s, a, r, s_) )
                agent.replay()            

            if rendering:
                self.env.render()      

            s = s_
            R += r

            if done:
                break

        print("Total reward:", R)

#-------------------- MAIN ----------------------------
if __name__ == "__main__":

    # Parser
    parser = argparse.ArgumentParser(description='DQN')
    parser.add_argument('-g', '--game', dest='game',
                        help='Chosen game', default=DEFAULT_GAME)
    parser.add_argument('--learn', dest='learning',
                        help='activate agent learning', action="store_true")
    parser.add_argument('--render', dest='rendering',
                        help='activate rendering', action="store_true")

    arguments = parser.parse_args()

    # Inits
    model_file = 'DDQN-PER-'+ arguments.game + '.h5'
    memory_file = 'DDQN-PER-'+ arguments.game + '.p'
    env = Environment(arguments.game + '-v0')
    stateCnt  = (IMAGE_STACK, IMAGE_WIDTH, IMAGE_HEIGHT)
    actionCnt = env.env.action_space.n

    # Launch
    print('-'*80)
    print("Launching DQN with game: {}, learning: {}; rendering: {}".format(arguments.game, arguments.learning, arguments.rendering))
    print('-'*80)

    # Model Loading
    if os.path.exists(model_file):
        agent = Agent(stateCnt, actionCnt, FORCED_EPSILON, EXPLORATION_STOP)
        print('-'*80)
        print("Loading model: {}...".format(model_file))
        agent.brain.model = load_model(model_file, custom_objects={'huber_loss': huber_loss})
        if os.path.exists(memory_file):
            print("Loading memory: {}...".format(memory_file))
            agent.brain.memory = pickle.load(open(memory_file, 'rb'))
        print('-'*80)
    else:
        print("Initialization with random agent...")
        randomAgent = RandomAgent(actionCnt)
        agent = Agent(stateCnt, actionCnt, MAX_EPSILON, 0)
        while randomAgent.exp < MEMORY_CAPACITY:
            env.run(randomAgent)
            print(randomAgent.exp, "/", MEMORY_CAPACITY)
        agent.memory = randomAgent.memory
        randomAgent = None

    # Main loop
    try:
        print("Starting agent with epsilon = {}.".format(agent.epsilon))
        while True:
            env.run(agent, arguments.learning, arguments.rendering)

    finally:
        print('-'*80)
        print("Saving model: {}".format(model_file))
        agent.brain.model.save(model_file)
        print("Saving memory: {}".format(memory_file))
        pickle.dump(agent.memory, open(memory_file, 'wb'))
        print('-'*80)
        print("Parameters: steps={}; epsilon={}".format(agent.steps, agent.epsilon))
        print('-'*80)
        print('Exiting.')

