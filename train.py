import gym
import random
import numpy as np
from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize
import torch
import torch.nn as nn
import torch.optim as optim
from model import Network

def to_one_hot(target, action_dim):
    batch_size = target.shape[0]
    onehot = torch.zeros(batch_size, action_dim)
    onehot[np.arange(batch_size), target] = 1
    return onehot

class DQNAgent:
    def __init__(self, action_size):
        self.render = False
        self.load_model = False
        # environment settings
        self.state_size = (4, 84, 84)
        self.action_size = action_size
        # parameters about epsilon
        self.epsilon = 1.
        self.epsilon_start, self.epsilon_end = 1.0, 0.1
        self.exploration_steps = 1000000.
        self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) \
                                  / self.exploration_steps
        # parameters about training
        self.batch_size = 32
        self.train_start = 50000
        self.update_target_rate = 10000
        self.discount_factor = 0.99
        self.memory = deque(maxlen=400000)
        self.no_op_steps = 30
        # build model
        self.model = Network().cuda()
        self.target_model = Network().cuda()
        self.update_target_model()

        self.optimizer = optim.RMSprop(self.model.parameters(), lr = 0.00025, alpha = 0.95, eps = 0.01)

        self.avg_q_max, self.avg_loss = 0, 0
        self.unclipped_score, self.clipped_score = 0, 0
        
        self.q_log, self.loss_log = [], []
        self.unclipped_log, self.clipped_log = [], []
        

        if self.load_model:
            self.model.load_weights("./models/model.ckpt")
        

    # if the error is in [-1, 1], then the cost is quadratic to the error
    # But outside the interval, the cost is linear to the error
    def compute_loss(self, history, action, target):

        py_x = self.model(history)

        a_one_hot = to_one_hot(action, self.action_size).cuda()
        q_value = torch.sum(py_x * a_one_hot, dim = 1)
        error = torch.abs(target - q_value)

        quadratic_part = torch.clamp(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = torch.mean(0.5 * (quadratic_part ** 2) + linear_part)

        return loss

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # get action from model using epsilon-greedy policy
    def get_action(self, history):
        history = torch.tensor(history / 255.0, dtype = torch.float).cuda()
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model(history).detach().cpu().numpy()
            return np.argmax(q_value[0])

    # save sample <s,a,r,s'> to the replay memory
    def replay_memory(self, history, action, reward, next_history, dead):
        self.memory.append((history, action, reward, next_history, dead))

    # pick samples randomly from replay memory (with batch_size)
    def train_replay(self):
        if len(self.memory) < self.train_start:
            return
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step

        mini_batch = random.sample(self.memory, self.batch_size)

        history = []
        next_history = []
        target = np.zeros(self.batch_size)
        action, reward, dead = [], [], []
        
        # <0, 1, 2, 3, 4  >
        # <s, a, r, s' dead>
        for i in range(self.batch_size):
            history.append(torch.tensor(mini_batch[i][0] / 255, dtype = torch.float))
            next_history.append(torch.tensor(mini_batch[i][3] / 255, dtype = torch.float))
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            dead.append(mini_batch[i][4])
        
        history = torch.cat(history, dim = 0).cuda()
        next_history = torch.cat(next_history, dim = 0).cuda()
        
        target_value = self.target_model(next_history).detach().cpu().numpy()

        # like Q Learning, get maximum Q value at s'
        # But from target model
        for i in range(self.batch_size):
            if dead[i]:
                target[i] = reward[i]
            else:
                target[i] = reward[i] + self.discount_factor * \
                                        np.max(target_value[i])

        target = torch.tensor(target, dtype = torch.float).cuda()
        action = np.array(action)
        loss = self.compute_loss(history, action, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.avg_loss += loss.item()

    def save_model(self):
        torch.save(self.model.state_dict(), './models/model.ckpt')
    
    def log_and_reset(self):
        self.q_log.append(self.avg_q_max)
        self.loss_log.append(self.avg_loss)
        self.unclipped_log.append(self.unclipped_score)
        self.clipped_log.append(self.clipped_score)
        
        np.save('./logs/q_log.npy', np.array(self.q_log, dtype = np.float))
        np.save('./logs/loss_log.npy', np.array(self.loss_log, dtype = np.float))
        np.save('./logs/unclipped_log.npy', np.array(self.unclipped_log, dtype = np.float))
        np.save('./logs/clipped_log.npy', np.array(self.clipped_log, dtype = np.float))
        
        self.avg_q_max, self.avg_loss = 0, 0
        self.unclipped_score, self.clipped_score = 0, 0
        

# 210*160*3(color) --> 84*84(mono)
# float --> integer (to reduce the size of replay memory)
def pre_processing(observe):
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe


if __name__ == "__main__":

    env = gym.make('BreakoutDeterministic-v4')
    agent = DQNAgent(action_size=3)
    
    print("< Playing Atari Breakout >")
    global_step = 0

    e = 1
    while True:
        done = False
        dead = False
        # 1 episode = 5 lives
        step, score, start_life = 0, 0, 5
        observe = env.reset()

        # this is one of DeepMind's idea.
        # just do nothing at the start of episode to avoid sub-optimal
        for _ in range(random.randint(1, agent.no_op_steps)):
            observe, _, _, _ = env.step(1)

        # At start of episode, there is no preceding frame
        # So just copy initial states to make history
        state = pre_processing(observe)
        history = np.stack((state, state, state, state))
        history = np.expand_dims(history, axis = 0)

        while not done:
            if agent.render:
                env.render()
            global_step += 1
            step += 1

            # get action for the current history and go one step in environment
            action = agent.get_action(history)
            # change action to real_action
            if action == 0:
                real_action = 1
            elif action == 1:
                real_action = 2
            else:
                real_action = 3

            observe, reward, done, info = env.step(real_action)
            
            # pre-process the observation --> history
            next_state = pre_processing(observe)
            next_state = np.reshape([next_state], (1, 1, 84, 84))
            next_history = np.append(next_state, history[:, :3, :, :], axis = 1)
            
            q_val = agent.model(torch.tensor(history / 255, dtype = torch.float).cuda())
            q_val = q_val.detach().cpu().numpy()[0]
            agent.avg_q_max += np.max(q_val)

            # if the agent missed ball, agent is dead --> episode is not over
            if start_life > info['ale.lives']:
                dead = True
                start_life = info['ale.lives']
            
            agent.unclipped_score += reward
            reward = np.clip(reward, -1., 1.)
            agent.clipped_score += reward
            
            # save the sample <s, a, r, s'> to the replay memory
            agent.replay_memory(history, action, reward, next_history, dead)
            
            # every some time interval, train model
            agent.train_replay()
            
            # update the target model with model
            if global_step % agent.update_target_rate == 0:
                agent.update_target_model()

            # if agent is dead, then reset the history
            if dead:
                dead = False
            else:
                history = next_history

            # if done, plot the score over episodes
            if done:
                print("episode:", e, 
                      "  score:", agent.unclipped_score, 
                      "  clipped score:", agent.clipped_score,
                      "  memory length:", len(agent.memory), 
                      "\nepsilon:", agent.epsilon,
                      "  global_step:", global_step, 
                      "  average_q:", agent.avg_q_max / float(step), 
                      "  average loss:", agent.avg_loss / float(step))
                e += 1
                agent.log_and_reset()

        if e % 1000 == 0:
            agent.save_model()