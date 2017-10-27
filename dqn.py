import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

import gym
import numpy as np
import os
import random
import sys
import argparse

import plots
from collections import namedtuple

parser = argparse.ArgumentParser(description='DQN Configuration')
parser.add_argument('--model', default='dqn', type=str, help='Model name, prefix of saved file')
parser.add_argument('--use_cpu',  action='store_true', help='Use CPU instead of default GPU for training')
parser.add_argument('--load_latest',  action='store_true', help='load latest checkpoint')
parser.add_argument('--debug',  action='store_true', help='Debug mode- save qvalues')
parser.add_argument('--clip', action='store_true', help='clipping the delta between -1 and 1')
parser.add_argument('--skip_frames', action='store_true', help='Skip 4 frames')
parser.add_argument('--num_episodes', default=10000, type= int, help="Number of training episodes")
parser.add_argument('--replay_memory_size', default=500000, type= int, help="Max capacity of replay memory")
parser.add_argument('--replay_memory_init_size', default=100000, type= int, help="Number of states to populate replay memory with during initialization")
parser.add_argument('--update_target_estimator_every', default=10000, type= int, help="Update target network every _ steps")
parser.add_argument('--save_qnetwork_every', default=50, type= int, help="Save QNetwork weights every _ episodes")
parser.add_argument('--epsilon_start', default=1.0, type= float, help="Initial epsilon") 
parser.add_argument('--epsilon_end', default=0.1, type= float, help="Epsilon at end of annealing")
parser.add_argument('--epsilon_decay_steps', default=500000, type= int, help="Epsilon decay schedule length")
parser.add_argument('--discount_factor', default= .99, type= float, help="Discount factor for MDP") 
parser.add_argument('--batch_size', default=128, type= int, help="Batch size for CNN training")
args= parser.parse_args()
print(args)

if "../" not in sys.path:
  sys.path.append("../")

# if gpu is to be used
use_cuda= False
if torch.cuda.is_available() and args.use_cpu==False:
    use_cuda=True
    print("Using GPU")
else:
    print("Using CPU")

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
        self.conv2= nn.Conv2d(32,64,4,2)
        self.conv3= nn.Conv2d(64,64,3,1)
        # Batch norm is supposedly not helpful 
        # self.bn1 = nn.BatchNorm2d(32)
        # self.bn2 = nn.BatchNorm2d(64)
        # self.bn3 = nn.BatchNorm2d(64)
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
        h= F.relu(self.conv1(x))
        h= F.relu(self.conv2(h))
        h= F.relu(self.conv3(h))
        # h= F.relu(self.bn1(self.conv1(x)))
        # h= F.relu(self.bn2(self.conv2(h)))
        # h= F.relu(self.bn3(self.conv3(h)))
        h=h.view(x.size(0), -1)
        h= F.relu(self.fc1(h))
        h= self.fc2(h)
        return h
     
def make_epsilon_greedy_policy(estimator, nA, use_cuda=True):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

    Args:
        estimator: An estimator that returns q values for a given state
        nA: Number of actions in the environment.
        use_cuda: Use GPU or not. Currently unused

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
    def __init__(self, size):
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
    
def save_checkpoint(state, filename):
    torch.save(state, filename)
    print(f"=> saving checkpoint '{filename}' at episode {state['epoch']}")

        
def load_checkpoint(filename):
    """
    state: is a dictionary with initialized 'q_estimator' and 'optimizer' keys
    filename: location of file

    """
    if os.path.isfile(filename):
        state={}
        print(f"=> loading checkpoint '{filename}'")
        state = torch.load(filename)
        if state:
            print(f"=> loaded checkpoint at {filename}")
            return state
        else:
            print("Loading failed!!")
            return None
    else:
        print(f"=> no checkpoint found at {filename}")
        return None


def optimize(batch_size, replay_memory,q_estimator, target_estimator, optimizer, discount_factor, grad_clip=False):

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



def main():
    model_name= args.model
    num_episodes= args.num_episodes
    replay_memory_size=args.replay_memory_size
    replay_memory_init_size=args.replay_memory_init_size
    update_target_estimator_every=args.update_target_estimator_every
    save_qnetwork_every=args.save_qnetwork_every
    epsilon_start=args.epsilon_start
    epsilon_end=args.epsilon_end
    epsilon_decay_steps=args.epsilon_decay_steps
    discount_factor=args.discount_factor
    batch_size=args.batch_size
    grad_clip= args.clip
    debug= args.debug
    skip_frames= args.skip_frames


    q_estimator= QNetwork()
    target_estimator= QNetwork()
    if use_cuda:
        q_estimator.cuda()
        target_estimator.cuda()
    optimizer= torch.optim.RMSprop(q_estimator.parameters(),lr=0.00025, momentum=0, weight_decay=0.99, eps=1e-6)

    start_episode=0
    state={}
    if args.load_latest:
        checkpoint=load_checkpoint(model_name+'_chkpt.pth')

    total_t=0 #indexes episode,step
    start_episode=0

    # Keeps track of useful statistics
    q_values=[]
    episode_lengths=np.zeros(10000)
    episode_rewards=np.zeros(10000)

    if checkpoint:
        start_episode= checkpoint['epoch']
        episode_lengths= checkpoint['episode_lengths']
        episode_rewards = checkpoint['episode_rewards']
        total_t= checkpoint['total_t']
        q_estimator.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Starting at episode {start_episode}, global step:{total_t}")

    copy_model_parameters(q_estimator,target_estimator) #Initializing them to same values

    replay_memory= ReplayMemory(replay_memory_size)

    # The epsilon decay schedule
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

    # The policy we're following
    policy = make_epsilon_greedy_policy(q_estimator,nA)

    # Populate the replay memory with initial experience
    env = gym.envs.make("Breakout-v0")
    state = env.reset() 
    state= process(state) 
    state=torch.cat([state.clone() for i in range(4)]) 
    state=state.unsqueeze(0) #Dim is 1*4*84*84

    print("Populating replay memory with intial experience")
    for i in range(replay_memory_init_size):
        # make random actions and observe action sequences
        a_probs= policy(Variable(state,volatile=True).cuda(), 1)
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
    for i_episode in range(start_episode,num_episodes):

        # Reset the environment
        state = env.reset() 
        state= process(state) 
        state=torch.cat([state.clone() for i in range(4)]) 
        state=state.unsqueeze(0)

        k=4 # Used to decide number of frames skipped
        action=None
        loss = None
        done=False

        if i_episode>start_episode and i_episode % save_qnetwork_every == 0:
            save_checkpoint({'epoch': i_episode + 1,
                             'total_t':total_t,
                             'state_dict': q_estimator.state_dict(),
                             'optimizer' : optimizer.state_dict(),
                             'episode_lengths':episode_lengths,
                             'episode_rewards':episode_rewards,
                            }, model_name+'_chkpt.pth')
        while not done:
            if skip_frames and action is not None:
               _,_,done,_= env.step(action)
               _,_,done,_= env.step(action)
               _,_,done,_= env.step(action)
               _,_,done,_= env.step(action)

            # Update the target estimator
            if total_t % update_target_estimator_every == 0:
                copy_model_parameters(q_estimator,target_estimator)

            # Epsilon for this time step
            epsilon = epsilons[min(total_t, epsilon_decay_steps-1)]

             #Make epsilon-greedy policy
            policy= make_epsilon_greedy_policy(q_estimator,nA)

             # Print out which step we're on, useful for debugging.
            if total_t%100==0:
                log_str= f"""Episode length: {episode_lengths[i_episode]}%d @ Episode {i_episode+1}/{num_episodes} Step: {total_t}"""
                log_str+= f""",loss: {loss}, episode_reward: {episode_rewards[i_episode]}, epsilon:{epsilon}"""
                log_str+=f", Replay memory size {len(replay_memory)}/{replay_memory_size}"
                print(log_str)
                sys.stdout.flush() 
                #Add logger statements here

            # Take a step in the environment
            if debug:
                q_values.append(q_estimator(Variable(state.cuda(),volatile=True)).mean().cpu().data.numpy())

            a_probs= policy(Variable(state,volatile=True).cuda(),epsilon)
            action = random.choices(VALID_ACTIONS, weights=a_probs)[0]
            new_state,reward,done,_= env.step(action)
            new_state= process(new_state).unsqueeze(0)
            new_state=torch.cat([state[:,1:,:,:], new_state],dim=1)

            #Reward shaping. Dangerous
            #if done:
            #    reward= -1
            # Save transition to replay memory
            reward = max(-1.0, min(reward, 1.0)) #Reward clipping
            replay_memory.push(state,action-1,reward,new_state,done)
        
            # Update statistics
            episode_rewards[i_episode] += reward
            episode_lengths[i_episode] += 1

            loss= optimize(batch_size, replay_memory, q_estimator,
                    target_estimator, optimizer, discount_factor, grad_clip)

            state = new_state
            total_t += 1
            
if __name__=='__main__':
    main()
    
    


