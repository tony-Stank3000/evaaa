# %%
# python packages
import os, glob, time, datetime, yaml, shutil, pprint, json
from collections import deque
import numpy as np
import sys

# pytorch packages
import torch
from torch.utils.tensorboard import SummaryWriter

# mlagents packages
from mlagents_envs.environment import ActionTuple, UnityEnvironment
from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)

from algos.dqn.DQN_agent import DqnAgent

from algos.dqn.utils.observation_dqn import (
    get_observation_spec,
    get_observations,
    make_input_observation,
)

# Add the override function here
def apply_manual_dqn_overrides(config, argv):
    for arg in argv:
        if arg.startswith("tag="):
            config["code_settings"]["tag"] = arg.split("=", 1)[1]
        elif arg.startswith("env.port="):
            config["code_settings"]["port"] = int(arg.split("=", 1)[1])
        elif arg.startswith("env.time_scale="):
            config["engine_configuration"]["time_scale"] = int(arg.split("=", 1)[1])
        elif arg.startswith("env.width="):
            config["engine_configuration"]["width"] = int(arg.split("=", 1)[1])
        elif arg.startswith("env.height="):
            config["engine_configuration"]["height"] = int(arg.split("=", 1)[1])
        elif arg.startswith("seed="):
            config["code_settings"]["seed"] = int(arg.split("=", 1)[1])
        elif arg.startswith("env.config="):
            config["mainCofnig"]["configFolderName"] = arg.split("=", 1)[1]
        elif arg.startswith("env.screenRecordEnable="):
            val = arg.split("=", 1)[1].lower() == "true"
            config["mainCofnig"]["recordingScreen"]["recordEnable"] = val
        elif arg.startswith("env.dataRecordEnable="):
            val = arg.split("=", 1)[1].lower() == "true"
            config["mainCofnig"]["experimentData"]["recordEnable"] = val
        elif arg.startswith("checkpoint_path="):
            # Split into folder and ckpt name
            ckpt_path = arg.split("=", 1)[1]
            folder, ckpt = os.path.split(ckpt_path)
            config["code_settings"]["load_model"] = folder
            config["code_settings"]["model_ckpt"] = ckpt
    return config

# Update main to use the override logic
def main(config=None):
    now = datetime.datetime.now()
    
    if config is None:
        PROJECT_HOME = "./"
        CONFIG_FILE = "configs/exp/dqn.yaml"
        config_path = os.path.join(PROJECT_HOME, CONFIG_FILE)
        with open(config_path) as f:
            config = yaml.safe_load(f)
        # Apply CLI overrides if any
        config = apply_manual_dqn_overrides(config, sys.argv)
    else:
        # If config is passed, still apply CLI overrides for consistency
        config = apply_manual_dqn_overrides(config, sys.argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pp = pprint.PrettyPrinter(indent=1)
    pp.pprint(config)


    torch.manual_seed(int(config["code_settings"]["seed"]))
    np.random.seed(int(config["code_settings"]["seed"]))

    # Check the time of executing code. This will be unique index for log and saving model.
    
    write_tensorbaord_step = config["code_settings"]["write_tensorbaord_step"]

    tag = config["code_settings"]["tag"]
    record_name = (
        f"{now.year:02d}-{now.month:02d}-{now.day:02d}_{now.hour:02d}-{now.minute:02d}-{now.second:02d}_{tag}"
    )

    log_path = os.path.join(
                        PROJECT_HOME,
                        "logs",
                        "runs",
                        "dqn",
                        record_name,
            )

    # Copying configuration yaml file of the traning to recording folder
    if not config["code_settings"]["tag"] == "debug":
        record_path = os.path.join(PROJECT_HOME, "logs", "runs", "dqn", record_name, "version_0")

        # Create folder if there is not
        os.makedirs(record_path, exist_ok=True)

        # Save the in-memory config (with overrides) as YAML
        archive_config_path = os.path.join(record_path, "config.yaml")
        with open(archive_config_path, "w") as f:
            yaml.safe_dump(config, f, default_flow_style=False)
        print("----------******---------")
        print(f"Saved config file name: {record_name}")
        print(
            f"Saved config full directory: {config_path, archive_config_path}\n"
        )

    # %%
    code_settings = config["code_settings"]
    engine_configuration = config["engine_configuration"]
    environment_parameters = config["environment_parameters"]
    mainCofnig = config["mainCofnig"]

    file_path = os.path.join(
        PROJECT_HOME,
        "envs",
        code_settings["env_name"],
        "Config", 
        "mainConfig.json"
    )
        # Load the JSON file
    with open(file_path, "r") as file:
        data = json.load(file)

    data["configFolderName"] = mainCofnig["configFolderName"]
    data["recordingScreen"]["recordingFolderName"] = mainCofnig["recordingScreen"]["recordingFolderName"]
    data["experimentData"]["baseFolderName"] = mainCofnig["experimentData"]["baseFolderName"]
    data["experimentData"]["fileNamePrefix"] = mainCofnig["experimentData"]["fileNamePrefix"]
    data["recordingScreen"]["recordEnable"] = "true" if mainCofnig["recordingScreen"]["recordEnable"] else "false"
    data["experimentData"]["recordEnable"] = "true" if mainCofnig["experimentData"]["recordEnable"] else "false"

    # Save the updated JSON file
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)

    dir_name = os.path.dirname(file_path)
    shutil.copytree(dir_name, os.path.join(log_path, os.path.basename("env_cfg")), dirs_exist_ok=True)

    # For test mode, no backprop will be done
    if config["code_settings"]["experiment"] == "test":
        is_train = False
    else:
        is_train = True

    base_port = config["code_settings"]["port"]
    env_file = os.path.join(
        PROJECT_HOME,
        "envs",
        code_settings["env_name"],
        "build",
    )

    engineChannel = EngineConfigurationChannel()
    # paramChannel = EnvironmentParametersChannel()

    env = UnityEnvironment(
        file_name=env_file,
        seed=int(code_settings["seed"]),
        # side_channels=[engineChannel, paramChannel],
        side_channels=[engineChannel],
        base_port=base_port,
    )

    engineChannel.set_configuration_parameters(
        time_scale=engine_configuration["time_scale"],
        width=int(engine_configuration["width"]),
        height=int(engine_configuration["height"]),
    )

    # # %%
    # # Setting pre-defined parameters from Unity
    # for key in environment_parameters.keys():
    #     for parameters in environment_parameters[key].keys():
    #         value = environment_parameters[key][parameters]
    #         paramChannel.set_float_parameter(parameters, float(value))

    # Reset environment with parameters
    env.reset()

    # %%
    # Get observation specs
    observation_spec = get_observation_spec(environment_parameters)
    # %%
    # Agent configuration
    agent_parameter = config["agent_parameter"]
    behavior_name = list(env.behavior_specs)[0]
    spec = env.behavior_specs[behavior_name]
    action_spec = spec.action_spec.discrete_branches[0]

    # Initialize agent
    agent = DqnAgent(
        observation_spec=observation_spec,
        action_spec=action_spec,
        # q_network=q_network,
        epsilon=agent_parameter["epsilon"],
        target_update_period=agent_parameter["target_update_period"],
        gamma=agent_parameter["gamma"],
        lr=agent_parameter["learning_rate"],
        reward_shaping=agent_parameter["reward_shaping"],
        batch_size=agent_parameter["batch_size"],
        max_time_step=agent_parameter["max_time_step"],
        epsilon_start=agent_parameter["epsilon_start"],
        epsilon_end=agent_parameter["epsilon_end"],
        exploration_steps=agent_parameter["exploration_steps"],
        device=device,
    )

    # Setting device for model
    agent._q_network.to(device)
    agent._target_q_network.to(device)

    # %%
    # Loading pre-trained model
    if code_settings["load_model"]:
        try:
            saved_model_path = os.path.join(
                PROJECT_HOME,
                code_settings["load_model"],
                code_settings["model_ckpt"],
            )
            model_ckpt = glob.glob(saved_model_path)[0]
        except:
            print("* Cannot find saved model: {}".format(saved_model_path))
            raise FileNotFoundError

        print("Loading saved model: {}".format(model_ckpt))
        agent._q_network.load_state_dict(torch.load(model_ckpt))
        agent._target_q_network.load_state_dict(torch.load(model_ckpt))
        print("#############################################")
        print("#### Loading pre-trained model completed ####")
        print("##############################################")
        # To display complete message, 3 seconds break
        time.sleep(3)

    # If test mode, batch normailization will be test mode
    if code_settings["experiment"] == "test":
        agent._q_network.eval()
        agent._target_q_network.eval()

    # %%
    # Setting variables for traning code
    episode_return_avg = 0
    episode_return_max = 0
    global_step = 0
    survival_time_steps = []

    done_config_to_log_path = False
    done_config_to_save_path = False
    done_network_to_save_path = False

    # Train will start after this global steps
    train_start = int(agent_parameter["train_start"])

    # Maximum episode number
    n_episode = int(agent_parameter["n_episode"])

    # %%
    ###################################### TRAING START #######################################
    for episode in range(1, n_episode + 1):
        # %%
        ## Env reset ##
        time_step = 0
        done = False
        episode_return = 0

        # %%
        # DQN observation que. DQN needs multiple timepoint observation to get temporal information
        obs_que = deque(maxlen=2)

        # Initializing observation
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        for agent_id in decision_steps:
            observations = get_observations(agent_id, decision_steps, observation_spec)
            observations["image"] = np.transpose(observations["image"], [0, 2, 3, 1])
        # Appending observations to DQN observation que
        obs_que.append(observations)
        if len(obs_que) == 1:
            obs_que.append(observations)

        # Concatenating each observations
        obs_input = make_input_observation(obs_que, observation_spec)

        # %%
        ######################################  Training loop ######################################
        while not done:
            # debug_log = f"Episode: {episode}/{n_episode} "
            # debug_log += f"Global step: {global_step} "
            # debug_log += f"Time step: {time_step} "
            # debug_log += f"Epsilon: {agent._epsilon:.5f} "
            # print(debug_log, end=" ")

            # %%
            # Get action from agent
            action, _, _ = agent.step(obs_input)  # Calculate action using DQN_agent's step function

            # Making action tuple for unity environment
            action_tuple = ActionTuple()
            action_tuple.add_discrete(np.resize(action, (len(decision_steps), 1)))

            ## Apply action to environment
            env.set_actions(behavior_name, action_tuple)
            env.step()

            # Get next step from environment
            decision_steps, terminal_steps = env.get_steps(behavior_name)

            # %%
            # Get next observation
            # If agent in terminal step, episode will end
            for agent_id in terminal_steps:
                done = True
                next_observations = get_observations(
                    agent_id, terminal_steps, observation_spec
                )
                next_observations["image"] = np.transpose(
                    next_observations["image"], [0, 2, 3, 1]
                )
            for agent_id in decision_steps:
                global_step += 1
                time_step += 1

                # If timestep over maximum, episode will end
                if time_step == agent.max_time_step:
                    done = True
                next_observations = get_observations(
                    agent_id, decision_steps, observation_spec
                )
                next_observations["image"] = np.transpose(
                    next_observations["image"], [0, 2, 3, 1]
                )

            # %%
            # Appending observations to DQN observation que
            obs_que.append(next_observations)
            # Concatenating each observations
            next_obs_input = make_input_observation(obs_que, observation_spec)

            # %%
            # Get Reward
            reward = agent.get_reward(
                observations, next_observations, action, done
            )  # Calculated in DQN_agent's get_reward function

            # %%
            if is_train:
                if not done:
                    # Add experience to replay memory (Store observed transitions)
                    sample = dict()
                    sample["observations"] = obs_input
                    sample["action"] = action
                    sample["reward"] = reward
                    sample["next_observations"] = next_obs_input
                    sample["done"] = done
                    agent.append_sample(sample)

                # Training model: Backpropagation
                if len(agent.memory) > train_start:
                    grads = agent.train()

                    # Updating DQN target model for each pre-defincd period
                    if global_step % agent._target_update_period == 0:
                        agent.update_target_model()

            # Recording total reward in episode
            episode_return += reward

            # Next observation become Old
            observations = next_observations
            obs_input = next_obs_input

        # %%
        ########## Recording tensorboard logs, configuration file, training model weights ##########
        # Recording traninig
        if is_train:
            if done:
                # print("Episode done!")
                # print(f"Total rewards for episode {episode} is {episode_return}")
                survival_time_steps.append(time_step)

                # Recording Tensorbaord
                if (
                    global_step > write_tensorbaord_step
                    and not config["code_settings"]["tag"] == "debug"
                ):
                    # Tensorboard log directory
                    log_dir = os.path.join(
                        PROJECT_HOME,
                        "logs",
                        "runs",
                        "dqn",
                        record_name,
                        'version_0'
                    )

                    if episode % code_settings["write_tensorboard_each_episodes"] == 0:
                        # Writing traning results to tensorboard
                        writer = SummaryWriter(log_dir=log_dir)
                        writer.add_scalar(
                            "Total Reward/Episode", episode_return, episode
                        )
                        writer.add_scalar("Duration/Episode", time_step, episode)
                        writer.add_scalar(
                            "Average Loss/Episode",
                            agent.avg_loss / float(time_step),
                            episode,
                        )
                        writer.add_scalar("Epsilon/Episode", agent._epsilon, episode)

                        # Get one node's weight for debugging purpose
                        for name, p in agent._q_network.named_parameters():
                            if name == "fc_cat.weight":
                                param = p.cpu().detach().numpy()
                        writer.add_scalar("tracking weight", param[0, 0], episode)

                    # Copying configuration yaml file of the traning to tensorboard log folder
                    if (
                        not done_config_to_log_path
                        and not config["code_settings"]["tag"] == "debug"
                    ):
                        shutil.copyfile
                        (
                            os.path.join(record_path, record_name),
                            os.path.join(log_dir, "config.yaml"),
                        )
                        done_config_to_log_path = True

        # %%
        # Saving models weight for each pre-defined episode numbers
        if (
            episode % code_settings["save_model_each_episodes"] == 0
            and not config["code_settings"]["tag"] == "debug"
        ):
            save_model_path = os.path.join(
                        PROJECT_HOME,
                        "logs",
                        "runs",
                        "dqn",
                        record_name,
                        'version_0',
                        'checkpoint'
            )

            os.makedirs(save_model_path, exist_ok=True)
            torch.save(
                agent._q_network.state_dict(), f"{save_model_path}/ckpt_{episode}.ckpt"
            )

            # # Copying configuration yaml file of the traning to saving model folder
            # if not done_config_to_save_path:
            #     shutil.copyfile(
            #         os.path.join(record_path, record_name),
            #         os.path.join(save_model_path, "config.yaml"),
            #     )
            #     done_config_to_save_path = True


    env.close()
    print("Experiment done. Close environment")


if __name__ == "__main__":
    main()
