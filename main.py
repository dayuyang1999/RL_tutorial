from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 900))
display.start()

import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display as ipythondisplay # to avoid conflict with the ``display" above
plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)



# define DQN model
class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        
        self.modelname = "vanilla"

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))




class DQN1(nn.Module):

    # class initialization
    def __init__(self, h, w, out_dim):
        super().__init__()
        # convolution layers -- may add pooling layers or add additional conv layer
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=10,kernel_size=5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(in_channels=10,out_channels=20,kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=20,out_channels=20,kernel_size=5)
        self.modelname = "max_pooling"

        # compute size after one convolution layer
        def conv2d_size_out(size, kernel_size=5, stride=1):
            outsize = (size-kernel_size)//stride+1
            return outsize
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)//2)//2)//2
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)//2)//2)//2
        self.fc1 = nn.Linear(in_features=20*convh*convw, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.fc3 = nn.Linear(in_features=60, out_features=out_dim)

    # forward propagation
    def forward(self, x):
        x = x.to(device)
        # Convolution + MaxPool layers
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = F.relu(self.pool(self.conv3(x)))
        # Fully connected layers
        x = F.relu(self.fc1(x.view(x.size(0),-1)))
        x = F.relu(self.fc2(x))
        # Output layer
        x = self.fc3(x)
        return x
    


class DQN2(nn.Module):
    '''
    the conv layer is increased
    '''

    def __init__(self, h, w, outputs):
        super(DQN2, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.bn5 = nn.BatchNorm2d(64)
        
        self.modelname = "deeper_conv"

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 3, stride = 1):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))))
        linear_input_size = convw * convh * 64
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        return self.head(x.view(x.size(0), -1))

    
class DQN3(nn.Module):
    '''
    the linear layer is increased
    '''

    def __init__(self, h, w, outputs):
        super(DQN3, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        
        self.modelname = "deeper_linear"

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        #print(linear_input_size, convw, convh)
        self.lin1 = nn.Linear(linear_input_size, 256) #, outputs)
        self.lin2 = nn.Linear(256, 64)
        self.lin3 = nn.Linear(64, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        return x

    
class DQN4(nn.Module):
    '''
    the linear layer and conv layer are both increased
    '''

    def __init__(self, h, w, outputs):
        super(DQN4, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.bn5 = nn.BatchNorm2d(64)
        
        self.modelname = "deeper_both"

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 3, stride = 1):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))))
        linear_input_size = convw * convh * 64
        self.lin1 = nn.Linear(linear_input_size, 256) #, outputs)
        self.lin2 = nn.Linear(256, 64)
        self.lin3 = nn.Linear(64, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))  
        return x

Transition = namedtuple('Transition',('state','action','next_state','reward'))

class ReplayMemory():
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity) # doubly ended queue

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def get_cart_location(env, screen_width):
    world_width = env.x_threshold * 2 # env location values are centered at 0, so multiplied by 2 to get the whole length
    scale = screen_width / world_width # find scale
    return int(env.state[0] * scale + screen_width / 2.0) # MIDDLE OF CART

def get_screen(env, screen_width, world_width):
    def get_cart_location(screen_width):
        world_width = env.x_threshold * 2 # env location values are centered at 0, so multiplied by 2 to get the whole length
    scale = screen_width / world_width # find scale
    return int(env.state[0] * scale + screen_width / 2.0) # MIDDLE OF CART
    
    screen = env.render(mode='rgb_array') # 400x600x3 tensor
    # Cart is in the lower half, so strip off the top and bottom of the screen
    screen_height, screen_width, _ = screen.shape
    screen = screen[int(screen_height*0.4):int(screen_height * 0.8),:,:]

    # get the cart location -- MIDDLE POINT OF CART   
    cart_location = get_cart_location(screen_width)

    # Strip off the edges, so that we have a rectangle image (160x180) centered on a cart
    view_width = int(screen_width * 0.3)
    if cart_location-view_width//2 < 0:
        slice_range = slice(view_width)
    elif cart_location+view_width//2 > screen_width:
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location-view_width//2, cart_location+view_width//2)

    screen = screen[:,slice_range,:]

    # Convert to float, rescale, convert to torch tensor
    screen = screen.astype(np.float32)/255
    screen = screen.transpose((2,0,1))
    screen = torch.from_numpy(screen)

    # furthur down sizing the screen
    resize = T.Compose([T.ToPILImage(),
                    T.Resize(size=40, interpolation=Image.BICUBIC),
                    T.ToTensor()])
    screen = resize(screen).unsqueeze(0)
    return screen

########## training function

def training(GAMMA=0.999, EPS_DECAY=100, model = DQN ):
    BATCH_SIZE = 128
    #GAMMA = 0.999
    EPS_START = 0.9
    EPS_END = 0.05
    #EPS_DECAY = 200
    TARGET_UPDATE = 10
    
    # create env
    env = gym.make('CartPole-v0').unwrapped
    env.reset()    

    
    ## helper function

    def get_cart_location(screen_width):
        world_width = env.x_threshold * 2 # env location values are centered at 0, so multiplied by 2 to get the whole length
        scale = screen_width / world_width # find scale
        return int(env.state[0] * scale + screen_width / 2.0) # MIDDLE OF CART

    def get_screen():
        
        screen = env.render(mode='rgb_array') # 400x600x3 tensor
        # Cart is in the lower half, so strip off the top and bottom of the screen
        screen_height, screen_width, _ = screen.shape
        screen = screen[int(screen_height*0.4):int(screen_height * 0.8),:,:]

        # get the cart location -- MIDDLE POINT OF CART   
        cart_location = get_cart_location(screen_width)

        # Strip off the edges, so that we have a rectangle image (160x180) centered on a cart
        view_width = int(screen_width * 0.3)
        if cart_location-view_width//2 < 0:
            slice_range = slice(view_width)
        elif cart_location+view_width//2 > screen_width:
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(cart_location-view_width//2, cart_location+view_width//2)

        screen = screen[:,slice_range,:]

        # Convert to float, rescale, convert to torch tensor
        screen = screen.astype(np.float32)/255
        screen = screen.transpose((2,0,1))
        screen = torch.from_numpy(screen)

        # furthur down sizing the screen
        resize = T.Compose([T.ToPILImage(),
                        T.Resize(size=40, interpolation=Image.BICUBIC),
                        T.ToTensor()])
        screen = resize(screen).unsqueeze(0)
        return screen
    Transition = namedtuple('Transition',('state','action','next_state','reward'))


    def select_action(state, steps):
        sample = random.random()
        # eps_threshold is decaying
        eps_threshold = EPS_END+(EPS_START-EPS_END) * math.exp(-1. * steps / EPS_DECAY)
        if sample > eps_threshold:
            with torch.no_grad():
                return policy_net(state).max(dim=1)[1].view(1,1) # return state
        else:
            return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long) 

    
    
    init_screen = get_screen()
    _, _, screen_height, screen_width = init_screen.shape

    # Get number of actions from gym action space
    n_actions = env.action_space.n

    policy_net = model(screen_height,screen_width,n_actions).to(device)
    target_net = model(screen_height,screen_width,n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval() # set to evaluation mode -- dropout and batchnorm are disabled in the evaluation mode

    learning_rate = 1e-3
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    memory = ReplayMemory(10000)

    # start loop
    num_episodes = 500
    episode_durations = []
    steps = 0 # global counter for setting epsilon of action selection
    for i_episode in range(num_episodes):
        # initialize environment and state
        env = gym.make('CartPole-v0').unwrapped
        env.reset()
        last_screen = get_screen()
        current_screen = get_screen()
        state = current_screen-last_screen
        steps_count = 0 # step counter for current episode
        # start the game of this episode
        while True:
            action = select_action(state,steps)
            steps += 1 # for epsilon-greedy
            steps_count += 1 # for episode duration
            _, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)

            # Observe new state
            last_screen = current_screen
            current_screen = get_screen()
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # if memory contains enough batch size,
            # perform one step optimization  
            if len(memory) >= BATCH_SIZE:
                transitions = memory.sample(BATCH_SIZE)
                # convert a list of Transitions to a Transition of lists (tuples)
                # so e.g., batch.state is a tuple with BATCH_SIZE elements
                batch = Transition(*zip(*transitions))
                # concatenate all batches to create tensors
                state_batch = torch.cat(batch.state)
                action_batch = torch.cat(batch.action)
                reward_batch = torch.cat(batch.reward)

                # check whether the batches are non-final
                is_non_final = tuple(map(lambda s: s is not None, batch.next_state)) # apply lambda function to all elements in batch.next_state
                non_final_mask = torch.tensor(is_non_final, device=device, dtype=torch.bool)

                # concatenate all non-final next_states batches
                temp_next_states = []
                for s in batch.next_state:
                    if s is not None:
                        temp_next_states.append(s)
                non_final_next_states = torch.cat(temp_next_states)

                # alternatively, may use this simple statement:
                # non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

                ### start Q-learning
                # Compute Q(St,a)
                state_action_values = policy_net(state_batch).gather(1, action_batch) # format should be matching for state_action_values and expected_state_action_values

                # Compute the TD Q values: reward+gamma*maxQ(s_{t+1},a)
                next_state_values = torch.zeros(BATCH_SIZE, device=device)
                next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach() # Returns a new Tensor, detached from the current graph. The result will not require gradient.
                expected_state_action_values = reward_batch + next_state_values * GAMMA

                # Compute Huber loss
                criterion = nn.SmoothL1Loss()
                loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                for param in policy_net.parameters():
                    param.grad.data.clamp_(-1, 1)  # gradient clipping
                optimizer.step()

            # Move to the next state
            state = next_state

            if done:
                # record this episode's duration
                episode_durations.append(steps_count+1)

                # save target net and plot durations
                if (i_episode+1) % 100 == 0:
                    torch.save(target_net.state_dict(), "DQN_target_" + policy_net.modelname + str((i_episode+1) // 100) + ".pth")
                    eps = EPS_END+(EPS_START-EPS_END) * math.exp(-1. * i_episode / EPS_DECAY)
                    print("eps = {}".format(eps))
                    print("Episode {} finished after {} timesteps".format(i_episode+1, steps_count+1))

                    if is_ipython:
                        ipythondisplay.clear_output() # remove previous plot
                    fig = plt.figure(2)
                    plt.clf()
                    durations_t = torch.tensor(episode_durations, dtype=torch.float)
                    plt.title('Training...')
                    plt.xlabel('Episode')
                    plt.ylabel('Duration')
                    plt.plot(durations_t.numpy())
                    plt.title(f"training duration vs episode: {policy_net.modelname}")
                    # Take 100 episode averages and plot them too
                    if len(durations_t) >= 100:
                        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
                        means = torch.cat((torch.zeros(99), means))
                        plt.plot(means.numpy())
                    plt.pause(0.001) 
                    if (i_episode+1) % 100 == 5:
                        fig.savefig(f"gamma_{GAMMA}_eps_decay_{EPS_DECAY}_model_{policy_net.modelname}.png")




                # break out of while loop
                env.close()
                break

        # Update the target network
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
    
    print('Training Completed for episods') 
