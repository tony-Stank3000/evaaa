# Minecraft bridge protocol (version 1)

This is the interface between the Python Gymnasium adapter and a future Minecraft
Java-side bridge. It does not claim a Java mod/server has been implemented.
A live session requires a bridge built for the actual Minecraft Java version and
chosen mod loader.

## Transport and safety

- TCP, one UTF-8 JSON object per line; every response ends with a newline.
- Default Python endpoint is 127.0.0.1:8765. Keep it on loopback.
- Every message includes protocol_version=1.
- The bridge validates action IDs 0 through 4, advances a bounded number of
  game ticks per action, and never interprets action data as shell/code.
- The Python adapter uses finite socket timeouts and rejects malformed,
  oversized, non-finite, or incompatible messages.

## Handshake

Client: {"type":"hello","protocol_version":1,"client":"evaaa_minecraft_env"}

Server: {"type":"hello_ack","protocol_version":1,"bridge":"implementation-name"}

## Reset

Client: {"type":"reset","protocol_version":1,"seed":123}

The seed is optional. The bridge should reset a controlled test world to a known
state when deterministic reset is supported. If not, disclose this in info.

Response shape:
{"type":"observation","protocol_version":1,"observation":{"rgb":[...],"ev":[0,0,0,0],"olfactory":[0,0,0,0,0,0,0,0,0,0],"thermo":[0,0,0,0,0,0,0,0],"collision":[0,0,0,0,0,0,0,0,0,0],"touch":[0]},"info":{}}

The RGB ellipsis is illustrative only; the actual payload must be a numeric
nested array of shape [64,64,3], with integer pixels in [0,255]. Vector values
must be finite and match the lengths below.

## Step

Client: {"type":"step","protocol_version":1,"action":1,"action_name":"forward"}

Server:
{"type":"step_result","protocol_version":1,"observation":{},"reward":0.0,"terminated":false,"truncated":false,"info":{}}

The response observation uses the same schema as reset. Reward must be finite.
terminated means a task terminal state; truncated means an external time/step
limit. Reward and terminal flags come from the bridge; Python does not invent
Minecraft mechanics or reward values.

## Action mapping copied from upstream Unity source

| ID | Name | Existing Unity behavior |
|---:|---|---|
| 0 | none | stop movement |
| 1 | forward | move forward |
| 2 | left | rotate left |
| 3 | right | rotate right |
| 4 | eat | attempt resource consumption |

The Minecraft implementation must document its tick duration, movement speed,
turn angle, and resource interaction. It must not claim physical equivalence
with Unity without a measured mapping.

## Observation contract

- rgb: [64,64,3], RGB pixels [0,255], uint8.
- ev: [4], intended to correspond to food, water, temperature, health.
- olfactory: [10], resource-sensing features.
- thermo: [8], directional thermal features.
- collision: [10], directional collision features.
- touch: [1], touch/contact signal.

These shapes match the upstream default experiment configuration, not every
possible EVAAA configuration. The initial adapter requires the complete set.
Exact Minecraft sensor semantics and scaling remain to be implemented and
documented against the selected experiment.

## Reward and fidelity

The Unity source's reward function returns the negative Euclidean distance of
four resource levels from a zero set point, each normalized by its configured
maximum. Do not blindly copy this formula until Minecraft resource units,
ranges, update equations, and terminal conditions are mapped. A faithful port
must preserve those configured equations; a Minecraft-native reward must be
versioned as a separate experiment and not described as an exact reproduction.
