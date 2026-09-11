"""
Simple example demonstrating how to interact with the EVAAA environment using ML-Agents.
This example shows basic environment setup, action execution, and observation processing.
See more details about mlagents_envs at https://github.com/Unity-Technologies/ml-agents/tree/main/ml-agents-envs
"""

import os
import numpy as np
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

# Path to the Unity environment build
# The last file name need to be matched with the files without extension.
# For example, if the file name is "build.x86_64", the file name should be "build
env_file = os.path.join('./envs/evaaa/build')

# Initialize side channels for environment configuration
# EngineConfigurationChannel: Controls engine settings like time scale and resolution
# EnvironmentParametersChannel: Allows passing custom parameters to the environment
engineChannel = EngineConfigurationChannel()
paramChannel = EnvironmentParametersChannel()

# Create the Unity environment
# file_name: Path to the Unity build
# seed: Random seed for reproducibility
# side_channels: List of channels for environment configuration
# base_port: Base port for communication (must be unique if running multiple environments)
env = UnityEnvironment(
        file_name=env_file,
        seed=42,
        side_channels=[engineChannel, paramChannel],
        base_port=50000,
        )

# Configure environment parameters
# time_scale: Controls simulation speed (higher = faster)
# width, height: Screen resolution
engineChannel.set_configuration_parameters(
        time_scale=15,
        width=100,
        height=100,
        )

# Reset the environment and get behavior specifications
env.reset()
behavior_name = list(env.behavior_specs)[0]  # Get the first behavior name
spec = env.behavior_specs[behavior_name]     # Get behavior specifications
n_agents = len(env.get_steps(behavior_name)[0])  # Number of agents
branch_sizes = spec.action_spec.discrete_branches  # Action space dimensions

# Print environment specifications
print("spec: ", spec)
print("action_spec: ", branch_sizes)

# Main interaction loop
done = False
for _ in range(1000): 
    # Generate random actions for each agent
    # Action mapping:
    # 0: None
    # 1: Forward
    # 2: Left
    # 3: Right
    # 4: Eat
    random_action = np.column_stack([
        np.random.randint(0, branch_size, size=n_agents)
        for branch_size in branch_sizes
    ])
    
    # Create and execute action
    # ActionTuple is used to format actions for the environment
    action_tuple = ActionTuple()
    action_tuple.add_discrete(random_action)
    env.set_actions(behavior_name, action_tuple)
    env.step()
    
    # Get observations and terminal states
    # In ML-Agents, the environment returns two types of steps:
    # 1. decision_steps: Contains observations for agents that need to make decisions
    #    - These are agents that are still active in the environment
    #    - Similar to Gym's step() return, but for multiple agents
    # 2. terminal_steps: Contains observations for agents that reached terminal states
    #    - These are agents that have completed their episode
    #    - Similar to Gym's done flag, but for multiple agents
    # This is different from Gym where step() returns (obs, reward, done, info)
    decision_steps, terminal_steps = env.get_steps(behavior_name)

    # Process observations for each agent that needs a decision
    # Each agent in decision_steps is still active and needs an action
    for agent_id in decision_steps:
        obs = decision_steps[agent_id].obs
        # Visual observations (images)
        # obs[0]: Camera input from the agent's perspective
        # Shape: (channels, height, width)
        print("visual observation: ", obs[0].shape)
        
        # Vector observations (obs[1]): A concatenated vector containing multiple types of observations
        # Essential Variables (indices 0-3):
        # - 0: Agent's food level
        # - 1: Agent's water level
        # - 2: Agent's temperature
        # - 3: Agent's health (damage)
        print("EV: ", " ".join(map(str, obs[1][0:4])))
        
        # Olfactory observations (indices 4-13):
        # These are based on resource properties within the olfactory sensor range
        # The sensor is a sphere around the agent (olfactorySensorLength radius)
        # For each detected resource (pond, water, food):
        # - The observation is calculated as: ResourceProperty * (1/distance)
        # - This creates a gradient of resource properties based on distance
        # - The 10 values represent different properties of the detected resources
        # - Higher values indicate stronger presence of resources
        # - Zero values indicate no resources in range
        # Note: Unlike directional sensors, these observations are cumulative
        # - They don't provide directional information (no angles)
        # - Instead, they give the total intensity of each property in the detection sphere
        # - This means the agent knows resources are nearby, but not their exact direction
        print("olfactory: ", " ".join(map(str, obs[1][4:14])))
        
        # Temperature readings (indices 14-21):
        # The agent has 8 directional thermal sensors arranged in a circle:
        # - Forward (F)
        # - Backward (B)
        # - Left (L)
        # - Right (R)
        # - Forward-Left (FL)
        # - Forward-Right (FR)
        # - Backward-Left (BL)
        # - Backward-Right (BR)
        # Each sensor provides temperature readings in its direction
        # The values represent the temperature gradient in each direction
        # Higher values indicate warmer temperatures in that direction
        # Zero values indicate no significant temperature difference
        # Note: The thermal sensing system is directional, unlike the olfactory system
        # - Each reading corresponds to a specific direction
        # - This allows the agent to navigate towards or away from temperature sources
        # - The readings are relative to the agent's current position and orientation
        print("temperature: ", " ".join(map(str, obs[1][14:22])))
        
        # Collision information (indices 22-31):
        # The agent uses a sophisticated collision detection system with 100 rays cast in a 360-degree circle
        # These rays are grouped into 10 sectors (10 rays per sector) to create a 10-directional collision map
        # Each sector (indices 22-31) represents a 36-degree arc around the agent
        # The collision observation values indicate:
        # - 0: No collision detected in that sector
        # - >0: Collision detected, with the value representing:
        #   * 1 + impulseMagnitude when a collision is detected
        #   * Higher values indicate stronger collisions (based on impact force)
        # The collision system:
        # - Continuously monitors for obstacles in all directions
        # - Detects both static obstacles and dynamic collisions
        # - Automatically resets collision state when contact is lost
        print("collision: ", " ".join(map(str, obs[1][22:32])))
    
    # Handle agents that have reached terminal states
    # Each agent in terminal_steps has completed its episode
    # This is similar to Gym's done=True, but for multiple agents
    for agent_id in terminal_steps:
        done = True
        print("*** terminal: ", done)
        env.reset()

# Clean up
env.close()
