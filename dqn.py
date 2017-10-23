import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

import gym
import itertools
import numpy as np
import os
import random
import sys
import copy

if "../" not in sys.path:
  sys.path.append("../")

import plots
from collections import deque, namedtuple

import matplotlib.pyplot as plt
from IPython import display
from matplotlib import pylab as pl



# if gpu is to be used
use_cuda = True
#use_cuda= torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor
regTensor= torch.Tensor


# Atari Actions: 0 (noop), 1 (fire), 2 (right) and 3 (left) are valid actions
#VALID_ACTIONS = [0, 1, 2, 3]
VALID_ACTIONS = [1, 2, 3]
nA=len(VALID_ACTIONS)


def process(screen):
    """
    Args:
        state: A [210, 160, 3] Atari RGB State

    Returns:
        A processed [1, 84, 84] state representing grayscale values.
        
    Note: remember this is only for processing a single frame. THe actual network needs 4 CNN frames
    """
    screen= screen[34:-16,:]
    img_transform=T.Compose([T.ToPILImage(),T.Scale((84,84))])
    screen= T.ImageOps.grayscale(img_transform(screen))
    screen= T.ToTensor()(screen)
    return screen.squeeze(1).type(regTensor)

def disp_state(state):
    a,b,c,d=state.chunk(4,dim=1)
    plt.figure()
    plt.imshow(a.squeeze().numpy(),cmap='gray')
    plt.figure()
    plt.imshow(b.squeeze().numpy(),cmap='gray')
    plt.figure()
    plt.imshow(c.squeeze().numpy(),cmap='gray')
    plt.figure()
    plt.imshow(d.squeeze().numpy(),cmap='gray')
    
def weights_init(m): 
    if isinstance(m, nn.Linear):
        size = m.weight.size()
        fan_out = size[0] # number of rows
        fan_in = size[1] # number of columns
        variance = np.sqrt(2.0/(fan_in + fan_out))
        m.weight.data.normal_(0.0, variance)
        
class QNetworkLinear(nn.Module):
    """Q-Value Estimator Linear layer.

    This network is used for both the Q-Network and the Target Network.
    """
    def __init__(self, in_frames=4,n_actions=nA):
        # Our input are 4 grayscale frames of shape 84,84 each
        # N*4*84*84
        super(QNetwork,self).__init__()        
        
        # Three convolutional layers
        self.fc1= nn.Linear(in_frames*84*84,n_actions)
        weights_init(self)
        
    def forward(self, x):
        """
        Predicts action values.

        Args:
          s: State input of shape [batch_size, 4, 84,84, 3]

        Returns:
          Tensor of shape [batch_size, NUM_VALID_ACTIONS] containing the estimated 
          action values.
        """
        #h= x/255. #normalizing data
        h=x.view(x.size(0), -1)
        return self.fc1(h)
    
    
class QNetwork(nn.Module):
    """Q-Value Estimator neural network.

    This network is used for both the Q-Network and the Target Network.
    """
    def __init__(self, in_frames=4,n_actions=nA):
        # Our input are 4 grayscale frames of shape 84,84 each
        # N*4*84*84
        super(QNetwork,self).__init__()        
        
        # Three convolutional layers
        self.conv1= nn.Conv2d(in_frames,32,8,4) 
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2= nn.Conv2d(32,64,4,2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3= nn.Conv2d(64,64,3,1)
        self.bn3 = nn.BatchNorm2d(64)
        # Fully connected layer
        self.fc1= nn.Linear(64*7*7,1500)
        self.fc2= nn.Linear(1500,n_actions)
        
        
    def forward(self, x):
        """
        Predicts action values.

        Args:
          s: State input of shape [batch_size, 4, 84,84, 3]

        Returns:
          Tensor of shape [batch_size, NUM_VALID_ACTIONS] containing the estimated 
          action values.
        """
        #h= x/255. #normalizing data
        h= F.relu(self.bn1(self.conv1(x)))
        h= F.relu(self.bn2(self.conv2(h)))
        h= F.relu(self.bn3(self.conv3(h)))
        #h= F.relu(self.conv1(x))
        #h= F.relu(self.conv2(h))
        #h= F.relu(self.conv3(h))
        h=h.view(x.size(0), -1)
        h= F.relu(self.fc1(h))
        h= self.fc2(h)
        return h
     
def make_epsilon_greedy_policy(estimator, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

    Args:
        estimator: An estimator that returns q values for a given state
        nA: Number of actions in the environment.

    Returns:
        A function that takes the (sess, observation, epsilon) as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """
    def policy_fn(observation, epsilon):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator(observation)
        _,best_action= q_values.max(1)
        best_action= best_action.cpu().data.numpy()
        #best_action= best_action.data.numpy()
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def copy_model_parameters(estimator1, estimator2):
    """
    Copies the model parameters of one estimator to another.

    Args:
      estimator1: Estimator to copy the paramters from
      estimator2: Estimator to copy the parameters to
    """
    estimator2.load_state_dict(estimator1.state_dict())
    
Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

class ReplayMemory(object):
    def __init__(self, size=10000):
        super(ReplayMemory,self).__init__()
        self.memory= []
        self.cur_size=0
        self.capacity= size
        
    def push(self,*args):
        self.memory.append(Transition(*args))
        self.cur_size+=1
        if self.cur_size>self.capacity:
            self.memory.pop(0)
            self.cur_size-=1

    def sample(self, batch_size=32):
        return random.sample(self.memory,batch_size)
    
    def __len__(self):
        return self.cur_size
    
def save_checkpoint(state, filename='dqn_chkpt.pth'):
    torch.save(state, filename)
    print(f"=> saving checkpoint '{filename}' at episode {state['epoch']}")

        
def load_checkpoint(state,filename='dqn_chkpt.pth'):
    if os.path.isfile(filename):
        print(f"=> loading checkpoint '{filename}'")
        checkpoint = torch.load(filename)
        i_episode = checkpoint['epoch']
        state['q_estimator'].load_state_dict(checkpoint['state_dict'])
        state['optimizer'].load_state_dict(checkpoint['optimizer'])
        print(f"=> loaded checkpoint at {filename}, epoch {i_episode}")
        return i_episode
    else:
        print(f"=> no checkpoint found at {filename}")
        return 0


def optimize(batch_size, replay_memory,q_estimator, target_estimator, optimizer, discount_factor, grad_clip=True):

    batch = Transition(*zip(*replay_memory.sample(batch_size)))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = ByteTensor(tuple(not last_state for last_state in batch.done))

    # We don't want to backprop through the expected action values and volatile
    # will save us on temporarily changing the model parameters'
    # requires_grad to False!
    non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                if s is not None]).cuda(),
                                     volatile=True)
    state_batch = Variable(torch.cat(batch.state).cuda())
    action_batch = Variable(LongTensor(batch.action))
    reward_batch = Variable(Tensor(batch.reward))

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = q_estimator(state_batch).gather(1, action_batch.unsqueeze(1))

    # Compute V(s_{t+1}) for all next states.
    next_state_values = Variable(torch.zeros(batch_size).type(Tensor))
    next_state_values[non_final_mask] = target_estimator(non_final_next_states).max(1)[0]
    # Now, we don't want to mess up the loss with a volatile flag, so let's
    # clear it. After this, we'll just end up with a Variable that has
    # requires_grad=False
    next_state_values.volatile = False
    # Compute the expected Q values
    targets = (next_state_values * discount_factor) + reward_batch


    # Perform gradient descent update
    
    #Mean squared error loss
    #loss= torch.sum((targets-state_action_values)**2)
    
    # Huber loss
    loss = F.smooth_l1_loss(state_action_values,targets)
    
    optimizer.zero_grad()
    loss.backward()
    if grad_clip:
        for param in q_estimator.parameters():
            param.grad.data.clamp_(-1, 1)
    optimizer.step()
    
    return loss.cpu().data.numpy()[0]


q_values=[]
# Keeps track of useful statistics
episode_lengths=np.zeros(10000)
episode_rewards=np.zeros(10000)


def main():
    global q_values
    #Initalizing and filling up replay memory

    num_episodes=10000
    replay_memory_size=500000
    replay_memory_init_size=10000
    update_target_estimator_every=3500
    save_qnetwork_every=50
    epsilon_start=1.0
    epsilon_end=0.1
    epsilon_decay_steps=500000
    discount_factor= .99
    batch_size=32

    q_estimator= QNetwork()
    target_estimator= QNetwork()
    q_estimator.cuda()
    target_estimator.cuda()
    optimizer= torch.optim.RMSprop(q_estimator.parameters(),lr=0.00025, momentum=0, weight_decay=0.99, eps=1e-6)

    state= {'q_estimator':q_estimator, 'optimizer':optimizer}
    start_episode=load_checkpoint(state)

    copy_model_parameters(q_estimator,target_estimator) #Initializing them to same values

    replay_memory= ReplayMemory(replay_memory_size)

    # The epsilon decay schedule
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

    # The policy we're following
    policy = make_epsilon_greedy_policy(q_estimator,nA)


    # Populate the replay memory with initial experience
    print("Populating replay memory with intial experience")
    env = gym.envs.make("Breakout-v0")
    state = env.reset() 
    state= process(state) 
    state=torch.cat([state.clone() for i in range(4)]) 
    #It is possible that more efficient way of storing states is as numpy arrays
    #Watch out for GPU memory blowups since replay memory is all GPU memory
    state=state.unsqueeze(0) #Dim is 1*4*84*84

    for i in range(replay_memory_init_size):
        # make random actions and observe action sequences
        a_probs= policy(Variable(state,volatile=True).cuda(), 1)
        #print a_probs. I want to see if it's working correctly
        action = random.choices(VALID_ACTIONS, weights=a_probs)[0]
        new_state,reward,done,_= env.step(action)
        #if done:
        #    reward=-1
        reward = max(-1.0, min(reward, 1.0))
        new_state= process(new_state).unsqueeze(0)
        new_state=torch.cat([state[:,1:,:,:], new_state],dim=1)
        replay_memory.push(state,action-1,reward,new_state,done)

        state= new_state

        if done:
            state= env.reset()
            state= process(state) 
            state=torch.cat([state.clone() for i in range(4)])
            state=state.unsqueeze(0)
    print(f"Replay memory is size {len(replay_memory)}")


    print("Training")
    total_t=0 #indexes episode,step
    for i_episode in range(start_episode,num_episodes):

        # Reset the environment
        state = env.reset() 
        state= process(state) 
        state=torch.cat([state.clone() for i in range(4)]) 
        state=state.unsqueeze(0)

        loss = None
        done=False
        #k=4
        action=None

        if i_episode>start_episode and i_episode % save_qnetwork_every == 0:
            save_checkpoint({'epoch': i_episode + 1,
                             'state_dict': q_estimator.state_dict(),
                             'optimizer' : optimizer.state_dict()
                            })
        while not done:
            #if action is not None and total_t%k==0:
            #    _,_,done,_= env.step(action)
            #    total_t+=1
            #    continue

            # Epsilon for this time step
            epsilon = epsilons[min(total_t, epsilon_decay_steps-1)]

            # Update the target estimator
            if total_t % update_target_estimator_every == 0:
                copy_model_parameters(q_estimator,target_estimator)

            # Print out which step we're on, useful for debugging.
            if total_t%100==0:
                print(f"""\rStep {episode_lengths[i_episode]} ({total_t}) @ Episode {i_episode+1}/{num_episodes},loss: {loss}, episode_reward: {episode_rewards[i_episode]}, epsilon:{epsilon}""",end="")
            sys.stdout.flush()

            #Make epsilon-greedy policy
            policy= make_epsilon_greedy_policy(q_estimator,nA)

            # Take a step in the environment
            q_values.append(q_estimator(Variable(state.cuda(),volatile=True)).mean().cpu().data.numpy())
            a_probs= policy(Variable(state,volatile=True).cuda(),epsilon)
            action = random.choices(VALID_ACTIONS, weights=a_probs)[0]
            new_state,reward,done,_= env.step(action)
            new_state= process(new_state).unsqueeze(0)
            new_state=torch.cat([state[:,1:,:,:], new_state],dim=1)

            # Save transition to replay memory
            #if done:
            #    reward= -1
            reward = max(-1.0, min(reward, 1.0))
            replay_memory.push(state,action-1,reward,new_state,done)
        
            # Update statistics
            episode_rewards[i_episode] += reward
            episode_lengths[i_episode] += 1

            loss= optimize(batch_size, replay_memory, q_estimator, q_estimator, optimizer, discount_factor)

            state = new_state
            total_t += 1
            
if __name__=='__main__':
    main()
    
    


