import pickle
from pathlib import Path

import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from soccer_twos import EnvType

from utils import create_rllib_env

NUM_ENVS_PER_WORKER = 3



class SaveWeightsCallback(DefaultCallbacks):
    def on_train_result(self, **info):
        path = Path("/Users/matthewspillman/Documents/_5th/Spring/Deep Reinforcement Learning/soccer-twos-starter/striker_agent/weights/striker.pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(info["trainer"].get_weights(["default_policy"])["default_policy"], f)
            print('saved weights!')

if __name__ == "__main__":
    ray.init()

    tune.registry.register_env("Soccer", create_rllib_env)

    analysis = tune.run(
        "PPO",
        name="PPO_SP",
        config={
            # system settings
            "num_gpus": 0,
            "num_workers": 2,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            "callbacks": SaveWeightsCallback,
            # RL setup
            "env": "Soccer",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "variation": EnvType.team_vs_policy,
                "multiagent": False,
                "single_player": True,
                "flatten_branched": True,
                "opponent_policy": lambda *_: 0,
            },
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [512],
            },
            "rollout_fragment_length": 500,
            "train_batch_size": 12000,
        },
        stop={
            # "timesteps_total": 20000000,  # 15M
            "time_total_s": 10 * 3600, # 4h
        },
        checkpoint_freq=100,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        restore="ray_results/PPO_SP/PPO_Soccer_e492d_00000_0_2026-03-17_22-40-57/checkpoint_000869/checkpoint-869",
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
    print(best_checkpoint)
    print("Done training")
    print("Done training")
