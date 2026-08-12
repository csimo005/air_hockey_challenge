import numpy as np

from air_hockey_challenge.framework import AgentBase
from rust_agent import Agent, EnvInfo, RobotInfo


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

        rsEnvInfo = EnvInfo.from_dict(env_info)
        rsRobotInfo = RobotInfo.from_dict(env_info["robot"])
        self.rsAgent = Agent(
            rsEnvInfo,
            rsRobotInfo,
            list(env_info["robot"]["base_frame"][0].flatten()),
        )

    def reset(self):
        self.rsAgent.reset()

    def draw_action(self, observation):
        action = self.rsAgent.draw_action(observation)
        return np.asarray(action).reshape(2, 7)
