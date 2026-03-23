import pickle
import numpy as np
from pathlib import Path

import ray
import torch.nn as nn
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from soccer_twos import EnvType

from utils import (create_rllib_env, create_shaped_rllib_env, sample_player,
                   sample_pos_vel)


class PPOTrainerWithInitWeights(PPOTrainer):
    def setup(self, config):
        striker_path = config.pop("striker_path")
        goalie_path = config.pop("goalie_path")
        super().setup(config)

        with open(striker_path, "rb") as f:
            striker_weights = pickle.load(f)
        with open(goalie_path, "rb") as f:
            goalie_weights = pickle.load(f)
        
        # print([(key, striker_weights[key].shape) for key in striker_weights.keys()])
        hidden_wt_key = "_hidden_layers.0._model.0.weight"
        hidden_bias_key = "_hidden_layers.0._model.0.bias"
        logit_wt_key = "_logits._model.0.weight"
        value_wt_key = "_value_branch._model.0.weight"
        extra_hidden = self.get_weights(["striker"])["striker"][hidden_wt_key].shape[0] - striker_weights[hidden_wt_key].shape[0]
        # print(extra_hidden)
        linear_1 = nn.Linear(336, extra_hidden)
        linear_2 = nn.Linear(336, extra_hidden)
        striker_weights[hidden_wt_key] = np.concatenate([
            striker_weights[hidden_wt_key],
            linear_1.weight.detach().numpy()
        ], axis=0)
        striker_weights[hidden_bias_key] = np.concatenate([
            striker_weights[hidden_bias_key],
            linear_1.bias.detach().numpy()
        ])
        striker_weights[logit_wt_key] = np.concatenate([
            striker_weights[logit_wt_key],
            np.zeros((27, extra_hidden))
        ], axis=1)
        striker_weights[value_wt_key] = np.concatenate([
            striker_weights[value_wt_key],
            np.zeros((1, extra_hidden))
        ], axis=1)

        goalie_weights[hidden_wt_key] = np.concatenate([
            goalie_weights[hidden_wt_key],
            linear_2.weight.detach().numpy()
        ], axis=0)
        goalie_weights[hidden_bias_key] = np.concatenate([
            goalie_weights[hidden_bias_key],
            linear_2.bias.detach().numpy()
        ])
        goalie_weights[logit_wt_key] = np.concatenate([
            goalie_weights[logit_wt_key],
            np.zeros((27, extra_hidden))
        ], axis=1)
        goalie_weights[value_wt_key] = np.concatenate([
            goalie_weights[value_wt_key],
            np.zeros((1, extra_hidden))
        ], axis=1)

        # Set local worker weights.
        print("------- Initializing weights from .pkl --------")
        self.set_weights({
            "striker": striker_weights,
            "opponent_striker": striker_weights
        })
        self.set_weights({
            "goalie": goalie_weights,
            "opponent_goalie": goalie_weights
        })

        # Important in older RLlib: propagate to remote rollout workers too.
        self.workers.sync_weights()

if __name__ == '__main__':
    ray.init()
    tune.registry.register_env("Soccer", create_shaped_rllib_env)
    temp_env = create_shaped_rllib_env({"flatten_branched" : True})
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()
    trainer = PPOTrainerWithInitWeights(
        env="Soccer",
        config={            
            "framework": "torch",
            "striker_path": "/Users/matthewspillman/Documents/_5th/Spring/Deep Reinforcement Learning/soccer-twos-starter/striker_agent/weights/striker.pkl",
            "goalie_path": "/Users/matthewspillman/Documents/_5th/Spring/Deep Reinforcement Learning/soccer-twos-starter/striker_agent/weights/goalie_2.pkl",
            "multiagent": {
                "policies": {
                    "striker": (None, obs_space, act_space, {}),
                    "goalie": (None, obs_space, act_space, {}),
                    "opponent_striker": (None, obs_space, act_space, {}),
                    "opponent_goalie": (None, obs_space, act_space, {}),
                },
            },
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [1024],
            },
        }
    )