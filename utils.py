import math
from random import uniform as randfloat

import gym
import soccer_twos
from ray.rllib import MultiAgentEnv

from shaped_env import RewardShapingWrapper


class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    """
    A RLLib wrapper so our env can inherit from MultiAgentEnv.
    """

    pass

class ShapedRLLibWrapper(gym.core.Wrapper):
    """
    A RLLib wrapper so our env can inherit from MultiAgentEnv.
    """
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        shaping_type = "normal" # hardcoded, change when needed
        if shaping_type == "goalie_1":
            ball_pos = info[0]['ball_info']['position']
            # calculate midpoint between ball and our own goal
            ball_wt = 0.3
            mid_pos = [
                ball_wt * ball_pos[0] - (1-ball_wt) * 15.5,
                ball_wt * ball_pos[1]]
            # make penalty proportional to distance between us and the midpoint
            player_pos = info[0]['player_info']['position']
            dist = math.dist(mid_pos, player_pos)
            if reward[0] < -0.5:
                # a goal was scored on us, keep some penalty
                reward[0] = -dist * 0.0001 - 0.3
            else:
                # if not, completely replace reward
                reward[0] = -dist * 0.0001
            return obs, reward, done, info
        elif shaping_type == "goalie_2":
            ball_pos = info[0]['ball_info']['position']
            player_pos = info[0]['player_info']['position']
            new_reward = 0
            if player_pos[0] > -4: # stay on our side of the field
                new_reward -= 0.01
            if ball_pos[0] > 3: # try to clear the ball
                new_reward += 0.001
            if reward[0] < -0.5: # don't get scored on
                new_reward += reward[0]
            # move the ball towards the other side
            new_reward += info[0]['ball_info']['velocity'][0] * 0.001
            reward[0] = new_reward
            return obs, reward, done, info
        else:
            ball_vel = info[0]['ball_info']['velocity'][0] * 0.002
            reward[0] += ball_vel
            reward[1] += ball_vel
            reward[2] -= ball_vel
            reward[3] -= ball_vel
            # reward += info['ball_info']['velocity'][0] * 0.002
            return obs, reward, done, info

def create_shaped_rllib_env(env_config: dict = {}):
    """
    Creates a RLLib environment and prepares it to be instantiated by Ray workers.
    Args:
        env_config: configuration for the environment.
            You may specify the following keys:
            - variation: one of soccer_twos.EnvType. Defaults to EnvType.multiagent_player.
            - opponent_policy: a Callable for your agent to train against. Defaults to a random policy.
    """
    if hasattr(env_config, "worker_index"):
        env_config["worker_id"] = (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )
    env = ShapedRLLibWrapper(soccer_twos.make(**env_config))
    if "multiagent" in env_config and not env_config["multiagent"]:
        # is multiagent by default, is only disabled if explicitly set to False
        return env
    return RLLibWrapper(env)

def create_rllib_env(env_config: dict = {}):
    """
    Creates a RLLib environment and prepares it to be instantiated by Ray workers.
    Args:
        env_config: configuration for the environment.
            You may specify the following keys:
            - variation: one of soccer_twos.EnvType. Defaults to EnvType.multiagent_player.
            - opponent_policy: a Callable for your agent to train against. Defaults to a random policy.
    """
    if hasattr(env_config, "worker_index"):
        env_config["worker_id"] = (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )
    env = soccer_twos.make(**env_config)
    # env = TransitionRecorderWrapper(env)
    if "multiagent" in env_config and not env_config["multiagent"]:
        # is multiagent by default, is only disabled if explicitly set to False
        return env
    return RLLibWrapper(env)


def sample_vec(range_dict):
    return [
        randfloat(range_dict["x"][0], range_dict["x"][1]),
        randfloat(range_dict["y"][0], range_dict["y"][1]),
    ]


def sample_val(range_tpl):
    return randfloat(range_tpl[0], range_tpl[1])


def sample_pos_vel(range_dict):
    _s = {}
    if "position" in range_dict:
        _s["position"] = sample_vec(range_dict["position"])
    if "velocity" in range_dict:
        _s["velocity"] = sample_vec(range_dict["velocity"])
    return _s


def sample_player(range_dict):
    _s = sample_pos_vel(range_dict)
    if "rotation_y" in range_dict:
        _s["rotation_y"] = sample_val(range_dict["rotation_y"])
    return _s
