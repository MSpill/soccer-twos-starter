import ray
from ray import tune
from ray.rllib.agents.registry import get_agent_class
from soccer_twos import EnvType

from utils import create_rllib_env

NUM_ENVS_PER_WORKER = 1

checkpoint_path = "ray_results/PPO_curriculum/PPO_Soccer_78b2e_00000_0_2026-03-18_12-21-09/checkpoint_000165/checkpoint-165"

if __name__ == "__main__":
    ray.init()

    tune.registry.register_env("Soccer", create_rllib_env)
    temp_env = create_rllib_env()
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    algo_name = "PPO"   # or "DQN", etc.
    cls = get_agent_class(algo_name)
    config={
        # system settings
        "num_gpus": 0,
        "num_workers": 1,
        "num_envs_per_worker": NUM_ENVS_PER_WORKER,
        "log_level": "INFO",
        "framework": "torch",
        # RL setup
        "env": "Soccer",
        "env_config": {
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "variation": EnvType.team_vs_policy,
            "multiagent": False,
            "flatten_branched": True,
            "single_player": True,
            "opponent_policy": lambda *_: 0,
        },
        "model": {
            "vf_share_layers": True,
            "fcnet_hiddens": [512, 512],
            "fcnet_activation": "relu",
        },
        "rollout_fragment_length": 5000,
        "batch_mode": "complete_episodes",
    }
    trainer = cls(config=config)
    trainer.restore(checkpoint_path)
    print('restored successfully')
    model = trainer.get_policy().model.policy_net
    print(model)
