import numpy as np
import soccer_twos


class RewardShapingWrapper:
    def __init__(self, env, alpha=0.01):
        self.env = env
        self.alpha = alpha

        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def close(self):
        return self.env.close()

    def _shape(self, info, rewards):
        shaped = dict(rewards)

        # You will need to inspect the exact structure of `info`
        # in your setup, but docs say it includes ball/player position
        # and velocity.
        ball = info.get("ball_info", {}) or info.get("ball", {})
        players = info.get("players_info", {}) or info.get("players", {})

        ball_pos = np.array(ball.get("position", [0.0, 0.0]), dtype=float)
        ball_vel = np.array(ball.get("velocity", [0.0, 0.0]), dtype=float)

        # Example heuristic:
        # reward blue team when ball moves right, orange when left.
        # You must confirm field orientation in your env.
        progress = ball_vel[0]

        for agent_id in rewards:
            if agent_id in [0, 1]:      # blue team
                shaped[agent_id] += self.alpha * progress
            elif agent_id in [2, 3]:    # orange team
                shaped[agent_id] -= self.alpha * progress

        return shaped

    def step(self, action):
        obs, rewards, done, info = self.env.step(action)
        rewards = self._shape(info, rewards)
        return obs, rewards, done, info