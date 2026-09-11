# %%
import os
import numpy as np
import torch
from collections import deque
from datetime import datetime

# %%
# detecting observation
def get_observation_spec(environment_param):
    observation_spec = dict()
    observation_spec["EV_SIZE"] = 2
    if environment_param["visualSensor"]["useVisual"]:
        observation_spec["USE_VIS"] = True
        observation_spec["VIS_H"]: int(environment_param["visualSensor"]["visualHight"])
        observation_spec["VIS_W"]: int(environment_param["visualSensor"]["visualWidth"])

    if environment_param["olfactorySensor"]["useOlfactory"]:
        observation_spec["USE_OLF"] = True
        observation_spec["OLF_SIZE"] = int(
            environment_param["olfactorySensor"]["olfactoryFeatureSize"]
        )
    else:
        observation_spec["OLF_SIZE"] = False

    if environment_param["thermoSensor"]["useThermo"]:
        observation_spec["EV_SIZE"] += 1
        observation_spec["THERMO_SIZE"] = int(
            environment_param["thermoSensor"]["thermoSensorSize"]
        )
        observation_spec["USE_TEMP"] = True
    else:
        observation_spec["USE_TEMP"] = False

    if environment_param["collisionSensor"]["useCollision"]:
        observation_spec["EV_SIZE"] += 1
        observation_spec["USE_COLL"] = True
        observation_spec["COLL_SIZE"] = int(
            environment_param["collisionSensor"]["collisionFeatureSize"]
        )
    else:
        observation_spec["COLL_SIZE"] = False

    if environment_param["touchSensor"]["useTouchObs"]:
        observation_spec["USE_TOUCH"] = True
        observation_spec["TOUCH_SIZE"] = 1
    else:
        observation_spec["USE_TOUCH"] = False


    return observation_spec


def get_observations(agent_id, steps, observation_spec):
    observations = dict()
    vec_obs_idx = observation_spec["EV_SIZE"]
    observations["ev"] = np.array(
        [steps[agent_id].obs[1][0:vec_obs_idx]], dtype=np.float32
    )
    # vec_obs_idx += 1  # Need to ignore HP EV.

    if observation_spec["USE_VIS"]:
        # image_que = deque(maxlen=2)
        image = steps[agent_id].obs[0]

        image_obs = np.transpose(image, axes=[2, 0, 1])
        image_obs = np.expand_dims(image_obs, axis=0)
        observations["image"] = image_obs

    if observation_spec["USE_OLF"]:
        observations["olfactory"] = np.array(
            [
                steps[agent_id].obs[1][
                    vec_obs_idx : vec_obs_idx + observation_spec["OLF_SIZE"]
                ]
            ],
            dtype=np.float32,
        )
        vec_obs_idx += observation_spec["OLF_SIZE"]

    if observation_spec["USE_TEMP"]:
        observations["temperature"] = np.array(
            [
                steps[agent_id].obs[1][
                    vec_obs_idx : vec_obs_idx + observation_spec["THERMO_SIZE"]
                ]
            ],
            dtype=np.float32,
        )
        vec_obs_idx += observation_spec["THERMO_SIZE"]

    if observation_spec["USE_COLL"]:

        observations["collision"] = np.array(
            [
                steps[agent_id].obs[1][
                    vec_obs_idx : vec_obs_idx + observation_spec["COLL_SIZE"]
                ]
            ],
            dtype=np.float32,
        )
        vec_obs_idx += observation_spec["COLL_SIZE"]

    if observation_spec["USE_TOUCH"]:
        observations["touch"] = np.array(
            [
                steps[agent_id].obs[1][
                    vec_obs_idx : vec_obs_idx + observation_spec["TOUCH_SIZE"]
                ]
            ],
            dtype=np.float32,
        )
        vec_obs_idx += observation_spec["TOUCH_SIZE"]


    return observations


def make_input_observation(obs_que, observation_spec):
    input_observations = dict()
    input_observations["ev"] = np.array(
        np.concatenate((obs_que[0]["ev"], obs_que[1]["ev"]), axis=-1), dtype=np.float32
    )

    if observation_spec["USE_VIS"]:
        input_observations["image"] = np.array(
            np.concatenate((obs_que[0]["image"], obs_que[1]["image"]), axis=1),
            dtype=np.float32,
        )

    if observation_spec["USE_OLF"]:
        input_observations["olfactory"] = np.array(
            np.concatenate((obs_que[0]["olfactory"], obs_que[1]["olfactory"]), axis=-1),
            dtype=np.float32,
        )

    if observation_spec["USE_TEMP"]:
        input_observations["temperature"] = np.array(
            np.concatenate(
                (obs_que[0]["temperature"], obs_que[1]["temperature"]), axis=-1
            ),
            dtype=np.float32,
        )

    if observation_spec["USE_COLL"]:
        input_observations["collision"] = np.array(
            np.concatenate((obs_que[0]["collision"], obs_que[1]["collision"]), axis=-1),
            dtype=np.float32,
        )

    if observation_spec["USE_TOUCH"]:
        input_observations["touch"] = np.array(
            np.concatenate((obs_que[0]["touch"], obs_que[1]["touch"]), axis=-1),
            dtype=np.float32,
        )

    return input_observations


def make_batch_observation(batch, observation_spec):
    invalid_entries = []
    valid_entries = []

    for b in batch:
        if (
            isinstance(b, dict)
            and "observations" in b
            and isinstance(b["observations"], dict)
            and "ev" in b["observations"]
        ):
            valid_entries.append(b)
        else:
            invalid_entries.append(b)
            print("Invalid entry:", b)

    if invalid_entries:
        time = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = "/media/nas01/projects/Interoceptive-AI/interoceptive-ai/batch_error"
        file_path = os.path.join(base_dir, time, "batch.txt")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "a") as f:
            f.write(f"batch: {batch}\n\n")

    # if not valid_entries:
    #     raise ValueError("No valid entries found in batch")

    observations = dict()
    observations["ev"] = np.concatenate(
        [b["observations"]["ev"] for b in batch], axis=0
    )

    if observation_spec["USE_VIS"]:
        observations["image"] = np.concatenate(
            [b["observations"]["image"] for b in batch], axis=0
        )

    if observation_spec["USE_OLF"]:
        observations["olfactory"] = np.concatenate(
            [b["observations"]["olfactory"] for b in batch], axis=0
        )

    if observation_spec["USE_TEMP"]:
        observations["temperature"] = np.concatenate(
            [b["observations"]["temperature"] for b in batch], axis=0
        )

    if observation_spec["USE_COLL"]:
        observations["collision"] = np.concatenate(
            [b["observations"]["collision"] for b in batch], axis=0
        )

    if observation_spec["USE_TOUCH"]:
        observations["touch"] = np.concatenate(
            [b["observations"]["touch"] for b in batch], axis=0
        )


    actions = np.expand_dims([b["action"] for b in batch], axis=1)
    rewards = np.array([b["reward"] for b in batch])

    next_observations = dict()
    next_observations["ev"] = np.concatenate(
        [b["next_observations"]["ev"] for b in batch], axis=0
    )

    if observation_spec["USE_VIS"]:
        next_observations["image"] = np.concatenate(
            [b["next_observations"]["image"] for b in batch], axis=0
        )

    if observation_spec["USE_OLF"]:
        next_observations["olfactory"] = np.concatenate(
            [b["next_observations"]["olfactory"] for b in batch], axis=0
        )

    if observation_spec["USE_TEMP"]:
        next_observations["temperature"] = np.concatenate(
            [b["next_observations"]["temperature"] for b in batch], axis=0
        )

    if observation_spec["USE_COLL"]:
        next_observations["collision"] = np.concatenate(
            [b["next_observations"]["collision"] for b in batch], axis=0
        )
        
    if observation_spec["USE_TOUCH"]:
        next_observations["touch"] = np.concatenate(
            [b["next_observations"]["touch"] for b in batch], axis=0
        )


    dones = np.array([b["done"] for b in batch])

    return observations, actions, rewards, next_observations, dones