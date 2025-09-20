import numpy as np

from air_hockey_challenge.framework import AgentBase
from rust_agent import Agent

def build_agent(env_info, **kwargs):
    """
    Function where an Agent that controls the environments should be returned.
    The Agent should inherit from the mushroom_rl Agent base env.

    Args:
        env_info (dict): The environment information
        kwargs (any): Additionally setting from agent_config.yml
    Returns:
         (AgentBase) An instance of the Agent
    """
    return RustAgentWrapper(env_info, **kwargs)

class RustAgentWrapper(AgentBase):
    def __init__(self, env_info, **kwargs):
        super().__init__(env_info, **kwargs)

        self.rsAgent = Agent(
            env_info["dt"],
            env_info["puck_pos_ids"],
            env_info["puck_vel_ids"],
            env_info["joint_pos_ids"],
            env_info["joint_vel_ids"],
            env_info["opponent_ee_ids"],
            env_info["table"]["length"],
            env_info["table"]["width"],
            env_info["table"]["goal_width"],
            env_info["puck"]["radius"],
            env_info["mallet"]["radius"],
            list(env_info["robot"]["joint_pos_limit"][0]),
            list(env_info["robot"]["joint_pos_limit"][1]),
            list(env_info["robot"]["joint_vel_limit"][0]),
            list(env_info["robot"]["joint_vel_limit"][1]),
            list(env_info["robot"]["joint_acc_limit"][0]),
            list(env_info["robot"]["joint_acc_limit"][1]),
            list(env_info["robot"]["base_frame"][0].flatten()),
        )

    def reset(self):
        self.rsAgent.reset()

    def draw_action(self, observation):
        action = self.rsAgent.draw_action(observation)
        return np.asarray(action).reshape(2, 7)
