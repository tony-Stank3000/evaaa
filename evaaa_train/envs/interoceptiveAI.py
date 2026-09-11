from __future__ import annotations

import os, random, json
from typing import Any, Dict, List, Optional, Sequence, SupportsFloat, Tuple, Union
from typing import List, Tuple

import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import RenderFrame
# import gym
# from gym import spaces
# from gym.core import RenderFrame

import numpy as np

from mlagents_envs.environment import ActionTuple, UnityEnvironment
from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)
# from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from envs.unity_gym_env import UnityToGymWrapper
os.environ["DISPLAY"] = ":1"


class InteroceptiveAIWrapper(gym.Wrapper):
# class InteroceptiveAIWrapper(Env):
    def __init__(
        self,
        id: str,
        screen_size: None,
        env_cfg: None,
        seed: int | None = None,
    ) -> None:
        
        self.env_cfg = env_cfg
        engineChannel = EngineConfigurationChannel()
        paramChannel = EnvironmentParametersChannel()

        engine_configuration = env_cfg["engine_configuration"]
        environment_parameters = env_cfg["environment_parameters"]
        self.environment_parameters = environment_parameters

        # assigned_port = env_cfg["env"]["env_port"] + random.randint(0, 1000)  # Assign a random port within a range
        assigned_port = env_cfg["env"]["env_port"]

        file_path = os.path.join(env_cfg["env"]["base_dir_"], env_cfg["env"]["env_name"], "Config", "mainConfig.json")
        # Load the JSON file
        with open(file_path, "r") as file:
            data = json.load(file)

        data["configFolderName"] = env_cfg["env"]["config_"]
        data["recordingScreen"]["recordingFolderName"] = env_cfg["env"]['recordingFolder_']
        data["experimentData"]["baseFolderName"] = env_cfg["env"]['baseFolderName_']
        data["experimentData"]["fileNamePrefix"] = env_cfg["env"]['fileNamePrefix_']
        # When the environment_parameters['common']['recordEnable_'] is True or False, set "true" or "false" string to data["recordEnable"] instead of boolean value
        data["recordingScreen"]["recordEnable"] = "true" if env_cfg["env"]['screenRecordEnable_'] else "false"
        data["experimentData"]["recordEnable"] = "true" if env_cfg["env"]['dataRecordEnable_'] else "false"

        # Save the updated JSON file
        with open(file_path, "w") as file:
            json.dump(data, file, indent=4)

        ENV_FILE = os.path.join(env_cfg["env"]["base_dir_"], env_cfg["env"]["env_name"], "build")
        
        unity_env = UnityEnvironment(
            file_name=ENV_FILE,
            seed=seed,
            side_channels=[engineChannel, paramChannel],
            base_port=assigned_port,
            # additional_args=["-logfile", "-"],
        )

        engineChannel.set_configuration_parameters(
            time_scale=engine_configuration["time_scale"],
            width=engine_configuration["width"],
            height=engine_configuration["height"],
        )

        # # Setting pre-defined parameters from Unity
        # for key in environment_parameters.keys():
        #     for parameters in environment_parameters[key].keys():
        #         value = environment_parameters[key][parameters]
        #         paramChannel.set_float_parameter(parameters, float(value))

        unity_env.reset()
        # env = UnityToGymWrapper(unity_env, uint8_visual=True, flatten_branched=False, allow_multiple_obs=True)
        super().__init__(
            UnityToGymWrapper(unity_env, uint8_visual=True, flatten_branched=False, allow_multiple_obs=True)
        )

        # Define observation spaces using gymnasium.Box
        # if environment_parameters["ev"]["useEV"]:
        #     self._vector_obs_infos = {"ev": (environment_parameters["ev"]["evSize"],)}
        # else:
        #     self._vector_obs_infos = {}
        # if environment_parameters["olfactorySensor"]["useOlfactory"]:
        #     self._vector_obs_infos["olfactory"] = (environment_parameters["olfactorySensor"]["olfactoryFeatureSize"],)
        # if environment_parameters["thermoSensor"]["useThermo"]:
        #     self._vector_obs_infos["thermo"] = (environment_parameters["thermoSensor"]["thermoSensorSize"],)
        # if environment_parameters["collisionSensor"]["useCollision"]:
        #     self._vector_obs_infos["collision"] = (environment_parameters["collisionSensor"]["collisionSensorSize"],)
        # if environment_parameters["touchSensor"]["useTouchObs"]:
        #     self._vector_obs_infos["touch"] = (environment_parameters["touchSensor"]["touchSensorSize"],)

        self._vector_obs_infos = {"ev": (environment_parameters["ev"]["evSize"],)}
        self._vector_obs_infos["olfactory"] = (environment_parameters["olfactorySensor"]["olfactoryFeatureSize"],)
        self._vector_obs_infos["thermo"] = (environment_parameters["thermoSensor"]["thermoSensorSize"],)
        self._vector_obs_infos["collision"] = (environment_parameters["collisionSensor"]["collisionSensorSize"],)
        self._vector_obs_infos["touch"] = (environment_parameters["touchSensor"]["touchSensorSize"],)

        self._selected_vector_obs_list = []
        if environment_parameters["ev"]["useEV"]:
            self._selected_vector_obs_list.append("ev")
        if environment_parameters["olfactorySensor"]["useOlfactory"]:
            self._selected_vector_obs_list.append("olfactory")
        if environment_parameters["thermoSensor"]["useThermo"]:
            self._selected_vector_obs_list.append("thermo")
        if environment_parameters["collisionSensor"]["useCollision"]:
            self._selected_vector_obs_list.append("collision")
        if environment_parameters["touchSensor"]["useTouchObs"]:
            self._selected_vector_obs_list.append("touch")

        obs_dict = {}
        if environment_parameters["visualSensor"]["useVisual"]:
            obs_dict["rgb"] = spaces.Box(
                low=self.env.observation_space[0].low,
                high=self.env.observation_space[0].high,
                shape=self.env.observation_space[0].shape,
                dtype=self.env.observation_space[0].dtype,
            )

        start_idx = 0
        for key in self._vector_obs_infos.keys():
            end_idx = start_idx + self._vector_obs_infos[key][0]

            if key in self._selected_vector_obs_list:
                obs_dict[key] = spaces.Box(
                    low=self.env.observation_space[1].low[start_idx:end_idx],
                    high=self.env.observation_space[1].high[start_idx:end_idx],
                    shape=self._vector_obs_infos[key],
                    dtype=self.env.observation_space[1].dtype,
                )

            start_idx = end_idx

        self.observation_space = spaces.Dict(obs_dict)
        self.action_space = spaces.Discrete(self.env.action_space.n)
        self.reward_range = (-np.inf, np.inf)
        self._render_mode = "rgb_array"  # Default render mode

        self.observation_space.seed(seed)
        self.action_space.seed(seed)

    @property
    def render_mode(self) -> str | None:
        return self._render_mode

    def _convert_obs(self, obs: np.ndarray) -> Dict[str, np.ndarray]:
        if self.environment_parameters["visualSensor"]["useVisual"]:
            _obs_dict = {"rgb": obs[0]}
        else:
            _obs_dict = {}  

        start_idx = 0
        for key in self._vector_obs_infos.keys():
            end_idx = start_idx + self._vector_obs_infos[key][0]
            if key in self._selected_vector_obs_list:
                _obs_dict[key] = obs[1][start_idx:end_idx]
            start_idx = end_idx

        # return {"rgb": obs[0], "state": obs[1]}
        return _obs_dict

    def _calculate_rewards(self, obs: np.ndarry, done) -> np.float32:
        if done: 
            reward = np.array(-100, dtype=np.float32)
        else:
            if self.env_cfg["env"]["use_reward_shaping_"]:
                # reward = -0.01 * (np.power(obs[1][:3], 2.0).sum(axis=0) + np.power((100 - obs[1][3]) * 0.15, 2.0))
                if self.env_cfg["env"]["use_homeostatic_reward_"]:
                    reward = -0.01 * (np.power(obs[1][:3], 2.0).sum(axis=0) + np.power((obs[1][3]) * 0.15, 2.0))
                else:
                    reward = np.sum(obs[1][4:14])/10 # reward for the approaching to the resources
                    reward -= np.sum(np.absolute(obs[1][14:22]))/1000 # reward for the temperature reward
                    if obs[1][-2] + obs[1][-3] >= 1: 
                        reward += 100 # reward for the consumption of the resources
                    if obs[1][-1] == 1:
                        reward -= 1 # reward for the collision with the obstacles
            else:
                reward = np.array(0, dtype=np.float32)

                

        # if self.env_cfg["env"]["use_reward_shaping"]:
        #     reward = -0.01 * (np.power(obs[1][:3], 2.0).sum(axis=0) + np.power((100 - obs[1][3]) * 0.15, 2.0))
        # else:
        #     if done:
        #         reward = np.array(-100, dtype=np.float32)
        #     else:
        #         reward = np.array(0, dtype=np.float32)
        return reward

    def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        obs, _, done, info = self.env.step(action)
        return self._convert_obs(obs), self._calculate_rewards(obs, done), done, False, info

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        obs = self.env.reset()
        return self._convert_obs(obs), {}

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        return self.env.render()

    def close(self) -> None:
        return self.env.close()
