import random
from collections import deque, defaultdict
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from algos.dqn.utils.action import ActionType
from algos.dqn.utils.observation_dqn import make_batch_observation

class QNetwork(nn.Module):
    def __init__(self, action_spec, observation_spec):
        super().__init__()
        # image_spec, vector_spec, olfactory_spec = observation_spec
        torch.set_default_dtype(torch.float32)
        self.observation_spec = observation_spec
        self.action_spec = action_spec
        cat_layer_size = 0
        
        # [set base layers]
        if self.observation_spec['USE_VIS']:
            self.conv1 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=2, padding=1)
            self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
            self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
            self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
            self.fc_img = nn.Linear(in_features=2048, out_features=1000) #64

            cat_layer_size += 1000

            self.bn1 = nn.BatchNorm2d(num_features=16)
            self.bn2 = nn.BatchNorm2d(num_features=32)
            self.bn3 = nn.BatchNorm2d(num_features=64)
            self.bn4 = nn.BatchNorm2d(num_features=128)

        if self.observation_spec['USE_OLF']:
            self.fc_olfactory = nn.Linear(in_features=self.observation_spec['OLF_SIZE']*2, out_features=100)
            cat_layer_size += 100

        if self.observation_spec['USE_TEMP']:
            self.fc_temperature = nn.Linear(in_features=self.observation_spec['THERMO_SIZE']*2, out_features=100)
            cat_layer_size += 100

        if self.observation_spec['USE_TOUCH']:
            self.fc_touch = nn.Linear(in_features=self.observation_spec['TOUCH_SIZE']*2, out_features=100)
            cat_layer_size += 100
        
        if self.observation_spec['USE_COLL']:
            self.fc_collision = nn.Linear(in_features=self.observation_spec['COLL_SIZE']*2, out_features=100)
            cat_layer_size += 100

        self.fc_ev = nn.Linear(in_features=self.observation_spec['EV_SIZE']*2, out_features=50)
        cat_layer_size += 50

        self.fc_cat = nn.Linear(in_features=cat_layer_size, out_features=400)  #shkim: in_features = 1350 
        self.fc_out = nn.Linear(in_features=400, out_features=self.action_spec)

    def call(self, observations, return_activation=False):
        if self.observation_spec['USE_VIS']:
            observations['image'] = np.float32(observations['image'] / 255.0)

        return self.forward(observations, return_activation)

    def forward(self, observations, return_activation=False, device=torch.device("cpu")):
        concat_layers = []

        ev = self.fc_ev(torch.from_numpy(observations['ev']).to(device))  # (batch_size, 1, 1, 50)
        # ev = nn.Flatten(ev)  # (batch_size, 1x1x50)
        concat_layers.append(ev)

        if self.observation_spec['USE_VIS']:
            image = self.conv1(torch.from_numpy(observations['image']).to(device))
            image = self.bn1(image)
            image = nn.ReLU()(image)
            image = self.conv2(image)
            image = self.bn2(image) 
            image = nn.ReLU()(image)
            image = self.conv3(image)
            image = self.bn3(image)
            image = nn.ReLU()(image)
            image = self.conv4(image)
            image = self.bn4(image) 
            image = nn.ReLU()(image)
            image = nn.Flatten()(image)
            image = self.fc_img(image)
            image = nn.ReLU()(image)

            concat_layers.append(image)

        if self.observation_spec['USE_OLF']:
            olfactory = self.fc_olfactory(torch.from_numpy(observations['olfactory']).to(device))  # (batch_size, 1, 1, 100)
            olfactory = nn.ReLU()(olfactory)
        # olfactory = nn.Flatten(olfactory)  # (batch_size, 1x1x100)
            concat_layers.append(olfactory)

        if self.observation_spec['USE_TEMP']:
            temperature = self.fc_temperature(torch.from_numpy(observations['temperature']).to(device)) 
            temperature = nn.ReLU()(temperature)
            concat_layers.append(temperature)

        if self.observation_spec['USE_TOUCH']:
            touch = self.fc_touch(torch.from_numpy(observations['touch']).to(device)) 
            touch = nn.ReLU()(touch)
            concat_layers.append(touch)
        
        if self.observation_spec['USE_COLL']:
            collision = self.fc_collision(torch.from_numpy(observations['collision']).to(device)) 
            collision = nn.ReLU()(collision)
            concat_layers.append(collision)

        out = torch.cat(concat_layers, dim=1)
        out = self.fc_cat(out)
        out = nn.ReLU()(out)
        out = self.fc_out(out)

        return out


class DqnAgent:
    def __init__(
        self,
        observation_spec,
        action_spec: int,
        epsilon: float = 1.0,
        target_update_period: int = 1000,
        gamma: float = 0.99,
        lr=0.0001,
        reward_shaping: bool = False,
        batch_size: int = 32,
        max_time_step: int = 1000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        exploration_steps: int = 1000000,
        device=torch.device("cpu"),
    ):
        self._observation_spec = observation_spec
        self._action_spec = action_spec
        self._epsilon = epsilon
        self._target_update_period = target_update_period
        self._gamma = gamma
        self._reward_shaping = reward_shaping
        self._step_counter = 0

        self.batch_size = batch_size
        self.max_time_step = max_time_step
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.exploration_steps = exploration_steps

        self.epsilon_decay_step = self.epsilon_start - self.epsilon_end
        self.epsilon_decay_step /= self.exploration_steps

        self.memory = deque(maxlen=50000)

        self._q_network = QNetwork(action_spec, observation_spec)
        self._target_q_network = QNetwork(action_spec, observation_spec)
        self.update_target_model()
        ##
        self.avg_q_max, self.avg_loss = 0, 0
        self._optimizer = optim.Adam(self._q_network.parameters(), lr=lr)
        self.device = device

    # Epsilon-greedy policy for action selection
    def step(self, observations, return_activation=False):
        if np.random.rand() <= self._epsilon:
            # print("Action: random", end=" ")
            q_value = self._q_network(observations, device=self.device)
            action = np.random.randint(self._action_spec)
            greedy = False

        else:
            # print("Action: greedy", end=" ")
            q_value = self._q_network(observations, device=self.device)
            action = np.argmax(q_value.cpu().detach().numpy()[0])
            greedy = True

        # print(f"{ActionType(action).name:10}", end=" ")

        return action, q_value, greedy

    def get_reward(self, observations, next_observations, action, done):
        if self._reward_shaping:
            # Reward
            # reward = -0.01 * (np.power(next_observations["ev"][0][0:3], 2.0).sum(axis=0) + np.power((100 - next_observations["ev"][0][3]) * 0.15, 2.0))
            reward = -0.01 * (np.power(next_observations["ev"][0][0:3], 2.0).sum(axis=0) + np.power((next_observations["ev"][0][3]) * 0.15, 2.0))

            # # Action penalty
            if action not in (
                ActionType.NONE.value,
                ActionType.LEFT.value,
                ActionType.RIGHT.value,
            ):
                reward -= 0.001

            if done:
                reward = -10
                
            # Clip reward
            reward = np.clip(reward, -10, 10, dtype=np.float32)
        else:
            reward = -1.0 if done else 0.0
        # print(f"Reward: {reward:.8}", end=" ")
        # print(f"Reward: {reward:.8}")
        return reward

    ################################## For train ###################################
    # Store sample <s, a, r, s'> in replay memory
    def append_sample(self, sample):  # type: ignore
        self.memory.append(sample)  # type: ignore

    # Train the model with a randomly sampled batch from replay memory
    def train(self):
        if self._epsilon > self.epsilon_end:
            self._epsilon -= self.epsilon_decay_step
        else:
            self._epsilon = self.epsilon_end

        # Randomly sample a batch from memory
        batch = random.sample(self.memory, self.batch_size)

        # Create s, a, s', r, d chunks from the sampled batch
        # images, vectors, actions, rewards, next_images, next_vectors, olf, next_olf = map(concat, zip(*batch))  # type: ignore
        observations, actions, rewards, next_observations, _ = make_batch_observation(
            batch, self._observation_spec
        )

        # images, vectors, actions, rewards, next_images, next_vectors, olf, next_olf = map(batch_samples, batch)  # type: ignore

        # Calculate Q(s_t, a) - the model computes Q(s_t) and selects the column of the taken action.
        # These are the actions selected for each batch state according to policy_net.
        state_action_values = self._q_network(observations, device=self.device).gather(
            1, torch.from_numpy(actions).to(self.device)
        )

        # Calculate V(s_{t+1}) for all next states
        # The expected values of the actions for non_final_next_states are calculated based on the "previous" target_net.
        # Use max(1)[0] to select the best reward.
        # This is merged based on the mask to have the expected state value or 0 if the state is terminal.
        next_state_values = (
            self._target_q_network(next_observations, device=self.device)
            .max(1)[0]
            .detach()
        )
        # Calculate expected Q value
        expected_state_action_values = (next_state_values.cpu() * self._gamma) + rewards
        expected_state_action_values = torch.Tensor.to(
            expected_state_action_values, dtype=torch.float32
        ).to(self.device)

        # Calculate Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        loss.backward()
        for param in self._q_network.parameters():
            param.grad.data.clamp_(-1, 1)
        self._optimizer.step()
        self._optimizer.zero_grad()

        self.avg_loss = loss.cpu().detach().numpy()

    # Update the target model with the weights of the model
    def update_target_model(self):
        self._target_q_network.load_state_dict(self._q_network.state_dict())
