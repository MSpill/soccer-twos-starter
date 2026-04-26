import pickle
from pathlib import Path

import numpy as np
import ray
import yaml
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.examples.policy.random_policy import RandomPolicy
from soccer_twos import EnvType

from copy_for_selfplay import PPOTrainerWithInitWeights
from utils import (create_rllib_env, create_shaped_rllib_env, sample_player,
                   sample_pos_vel)

NUM_ENVS_PER_WORKER = 3

def policy_mapping_fn(agent_id, *args, **kwargs):
    if agent_id == 0:
        return "striker"  # Choose 01 policy for agent_01
    elif agent_id == 1:
        return "goalie"
    elif agent_id == 2:
        return "opponent_striker"
    else:
        return np.random.choice(
            ["opponent_goalie", "opponent_striker"],
            size=1,
            p=[0.75, 0.25]
        )[0]

# class SelfPlayUpdateCallback(DefaultCallbacks):
#     def on_train_result(self, **info):
#         """
#         Update multiagent oponent weights when reward is high enough
#         """
#         if info["result"]["episode_reward_mean"] > 0.5:
#             print("---- Updating opponents!!! ----")
#             trainer = info["trainer"]
#             trainer.set_weights(
#                 {
#                     "opponent_3": trainer.get_weights(["opponent_2"])["opponent_2"],
#                     "opponent_2": trainer.get_weights(["opponent_1"])["opponent_1"],
#                     "opponent_1": trainer.get_weights(["default"])["default"],
#                 }
#             )

have_changed_weights = True
num_weight_updates = 0
running_reward_avg = 0.0

class CurriculumUpdateCallback(DefaultCallbacks):
    def on_train_result(self, **info):
        path = Path("/Users/matthewspillman/Documents/_5th/Spring/Deep Reinforcement Learning/soccer-twos-starter/striker_agent/weights/striker_2.pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(info["trainer"].get_weights(["striker"])["striker"], f)
            print('saved striker weights!')
        path = Path("/Users/matthewspillman/Documents/_5th/Spring/Deep Reinforcement Learning/soccer-twos-starter/striker_agent/weights/goalie_3.pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(info["trainer"].get_weights(["goalie"])["goalie"], f)
            print('saved goalie weights!')
        global have_changed_weights, num_weight_updates, running_reward_avg
        if not have_changed_weights:
            print('trying to update weights')
            trainer = info["trainer"]
            old_trainer = PPOTrainer(
                config = {
                    "framework": "torch",
                    "num_gpus": 0,
                    "num_workers": 0,
                    "env": "Soccer",
                    "env_config": {
                        "variation": EnvType.team_vs_policy,
                        "flatten_branched": True,
                        "single_player": True,
                    },
                    "model": {
                        "vf_share_layers": True,
                        "fcnet_hiddens": [512],
                        # "fcnet_activation": "relu",
                    }
            })
            old_trainer.restore("/Users/matthewspillman/Documents/_5th/Spring/Deep Reinforcement Learning/soccer-twos-starter/ray_results/PPO_SP/PPO_Soccer_e492d_00000_0_2026-03-17_22-40-57/checkpoint_000869/checkpoint-869")
            weights = old_trainer.get_weights(['default_policy'])['default_policy']
            print('updating weights')
            trainer.set_weights({"striker": weights})
            trainer.set_weights({"opponent_striker": weights})
            print("updated weights")
            # old_trainer = PPOTrainer(
            #     config={
            #         # system settings
            #         "num_gpus": 0,
            #         "num_workers": 0,
            #         "log_level": "INFO",
            #         "framework": "torch",
            #         # RL setup
            #         "multiagent": {
            #             "policies": {
            #                 "default": (None, obs_space, act_space, {}),
            #                 "dummy": (RandomPolicy, obs_space, act_space, {}),
            #                 "opponent": (None, obs_space, act_space, {}),
            #             },
            #             "policy_mapping_fn": tune.function(policy_mapping_fn),
            #             "policies_to_train": ["default"],
            #         },
            #         "env": "Soccer",
            #         "env_config": {
            #             "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            #             "flatten_branched": True
            #         },
            #         "model": {
            #             "vf_share_layers": True,
            #             "fcnet_hiddens": [512],
            #             # "fcnet_activation": "relu"
            #         },
            #     },
            # )
            # old_trainer.restore("/Users/matthewspillman/Documents/_5th/Spring/Deep Reinforcement Learning/soccer-twos-starter/ray_results/PPO_selfplay_rec/PPO_Soccer_15248_00000_0_2026-03-20_16-19-22/checkpoint_003753/checkpoint-3753")
            # weights = old_trainer.get_weights(['default'])['default']
            # trainer.set_weights({"goalie": weights})
            # trainer.set_weights({"opponent_goalie": weights})
            have_changed_weights = True
        running_reward_avg = 0.8 * running_reward_avg + 0.2 * info["result"]["policy_reward_mean"]["striker"]
        print(f'running avg is {running_reward_avg}')
        if running_reward_avg > 0.35:
            running_reward_avg = 0
            print("---- Updating opponents!!! ----")
            trainer = info["trainer"]
            trainer.set_weights(
                {
                    "opponent_striker": trainer.get_weights(["striker"])["striker"],
                    "opponent_goalie": trainer.get_weights(["goalie"])["goalie"],
                }
            )
            num_weight_updates += 1
        print(f"Have updated weights {num_weight_updates} times")
        info["result"]["episode_reward_mean"] = info["result"]["policy_reward_mean"]["striker"] * 2

if __name__ == "__main__":
    ray.init()

    tune.registry.register_env("Soccer", create_shaped_rllib_env)
    temp_env = create_shaped_rllib_env({"flatten_branched" : True})
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    analysis = tune.run(
        "PPO",
        name="PPO_selfplay_new",
        config={
            # system settings
            "num_gpus": 0,
            "num_workers": 2,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            "callbacks": CurriculumUpdateCallback,
            # RL setup
            "multiagent": {
                "policies": {
                    "striker": (None, obs_space, act_space, {}),
                    "goalie": (None, obs_space, act_space, {}),
                    "opponent_striker": (None, obs_space, act_space, {}),
                    "opponent_goalie": (None, obs_space, act_space, {}),
                },
                "policy_mapping_fn": tune.function(policy_mapping_fn),
                "policies_to_train": ["striker", "goalie"],
            },
            "env": "Soccer",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "flatten_branched": True
            },
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [1024],
                # "fcnet_activation": "relu"
            },
            "train_batch_size": 16000,
            "batch_mode": "complete_episodes",
            # "striker_path": "/Users/matthewspillman/Documents/_5th/Spring/Deep Reinforcement Learning/soccer-twos-starter/striker_agent/weights/striker.pkl",
            # "goalie_path": "/Users/matthewspillman/Documents/_5th/Spring/Deep Reinforcement Learning/soccer-twos-starter/striker_agent/weights/goalie_2.pkl"
        },
        stop={"time_total_s": 3600 * 216},  # 2h
        checkpoint_freq=20,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        restore="/Users/matthewspillman/Documents/_5th/Spring/Deep Reinforcement Learning/soccer-twos-starter/ray_results/PPO_selfplay_new/PPO_Soccer_6faba_00000_0_2026-03-28_09-23-21/checkpoint_003000/checkpoint-3000"
        # restore="/Users/mathewspillman/Documents/_5th/Spring/Deep Reinforcement Learning/soccer-twos-starter/ray_results/PPO_selfplay_new/PPO_Soccer_ac4f3_00000_0_2026-03-25_11-44-36/checkpoint_002320/checkpoint-2320"
        # restore="/Users/matthewspillman/Documents/_5th/Spring/Deep Reinforcement Learning/soccer-twos-starter/ray_results/PPO_selfplay_new/PPO_Soccer_7fbbb_00000_0_2026-03-23_23-49-46/checkpoint_000520/checkpoint-520"
        # restore="/Users/matthewspillman/Documents/_5th/Spring/Deep Reinforcement Learning/soccer-twos-starter/ray_results/PPO_selfplay_new/PPO_Soccer_be6b7_00000_0_2026-03-23_17-24-58/checkpoint_000160/checkpoint-160"
        # restore="ray_results/PPO_selfplay_rec/PPO_Soccer_419fc_00000_0_2026-03-21_17-31-01/checkpoint_000160/checkpoint-160"
        # restore="ray_results/PPO_selfplay_rec/PPO_Soccer_a79f3_00000_0_2026-03-21_17-12-23/checkpoint_000004/checkpoint-4"
        # restore="ray_results/PPO_selfplay_rec/PPO_Soccer_15248_00000_0_2026-03-20_16-19-22/checkpoint_003753/checkpoint-3753"
        # restore="ray_results/PPO_selfplay_rec/PPO_Soccer_98026_00000_0_2026-03-20_16-08-43/checkpoint_000002/checkpoint-2"
        # restore="ray_results/PPO_selfplay_rec/PPO_Soccer_bd183_00000_0_2026-03-19_23-27-35/checkpoint_002427/checkpoint-2427"
        # restore="./ray_results/PPO_selfplay_twos_2/PPO_Soccer_a8b44_00000_0_2021-09-18_11-13-55/checkpoint_000600/checkpoint-600",
    )

    # Gets best trial based on max accuracy across all training iterations.
    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print(best_trial)
    # Gets best checkpoint for trial based on accuracy.
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(best_checkpoint)
    print("Done training")
