# import random

# import ray
from gym_unity.envs import ActionFlattener
# from ray import tune
# from ray.rllib.agents.registry import get_agent_class
from soccer_twos import AgentInterface

# from utils import create_rllib_env, sample_player, sample_pos_vel

from .network import MyModel

NUM_ENVS_PER_WORKER = 1
# checkpoint_path = "ray_results/PPO_SP/PPO_Soccer_e492d_00000_0_2026-03-17_22-40-57/checkpoint_000869/checkpoint-869"

class TeamAgent(AgentInterface):
    """
    An agent definition for policies trained with DQN on `team_vs_policy` variation with `single_player=True`.
    """

    def __init__(self, env):
        super().__init__()
        print("initializing")
        self.striker = MyModel("TEAM69_AGENT/weights/striker_2.pkl")
        self.goalie = MyModel("TEAM69_AGENT/weights/goalie_3.pkl")
        # if not ray.is_initialized():
        #     ray.init()
        # tune.registry.register_env("Soccer", create_rllib_env)
        self.flattener = ActionFlattener(env.action_space.nvec)
        # algo_name = "PPO"
        # cls = get_agent_class(algo_name)
        # config = {
        #     "framework": "torch",
        #     "num_workers": 0,
        #     "env": "Soccer",
        #     "env_config": {
        #         "variation": EnvType.team_vs_policy,
        #         "flatten_branched": True,
        #         "single_player": True,
        #     },
        #     "model": {
        #         "vf_share_layers": True,
        #         "fcnet_hiddens": [512],
        #         # "fcnet_activation": "relu",
        #     },
        # }
        # self.trainer = cls(config=config)
        # self.trainer.restore(checkpoint_path)
    
    def act(self, observation):
        """The act method is called when the agent is asked to act.
        Args:
            observation: a dictionary where keys are team member ids and
                values are their corresponding observations of the environment,
                as numpy arrays.
        Returns:
            action: a dictionary where keys are team member ids and values
                are their corresponding actions, as np.arrays.
        """
        actions = {}
        actions[0] = self.flattener.lookup_action(self.striker.get_action(observation[0]))
        actions[1] = self.flattener.lookup_action(self.goalie.get_action(observation[1]))
        # actions[1] = [0, 0, 0]
        return actions



