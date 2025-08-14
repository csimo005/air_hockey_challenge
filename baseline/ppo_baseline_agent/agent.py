import os
import jax
from sbx import PPO
from air_hockey_challenge.framework.agent_base import AgentBase
from air_hockey_challenge.utils import forward_kinematics
import pickle
import numpy as np

from air_hockey_challenge.utils.kinematics import inverse_kinematics
from baseline.baseline_agent.kalman_filter import PuckTracker
from pathlib import Path

jax.config.update("jax_platform_name", "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def build_agent(env_info, **kwargs):
    return PPOBaselineAgent(env_info, **kwargs)


class PPOAgent:
    def __init__(self, env_info, path):
        self.env_info = env_info
        with open(path / "best_model.zip", "rb") as f:
            self.model = PPO.load(f, device="cpu")
            print(f"Using device {self.model.device}")
        with open(path / "vecnormalize.pkl", "rb") as f:
            self.normalizer = pickle.load(f)

    def draw_action(self, obs):
        obs = self.filter_obs(obs)
        planned_world_pos = forward_kinematics(
            self.env_info["robot"]["robot_model"],
            self.env_info["robot"]["robot_data"],
            self.interp_pos,
        )[0]
        obs = np.hstack(
            [
                obs,
                self.interp_pos,
                self.interp_vel,
                self.last_acceleration,
                planned_world_pos,
            ]
        )

        norm_obs = self.normalizer.normalize_obs(obs)
        action, _ = self.model.predict(norm_obs, deterministic=True)
        action /= 10

        new_vel = self.interp_vel + action

        jerk = (
            2 * (new_vel - self.interp_vel - self.last_acceleration * 0.02) / (0.02**2)
        )
        new_pos = (
            self.interp_pos
            + self.interp_vel * 0.02
            + (1 / 2) * self.last_acceleration * (0.02**2)
            + (1 / 6) * jerk * (0.02**3)
        )

        self.interp_pos = new_pos
        self.interp_vel = new_vel
        self.last_acceleration += jerk * 0.02

        abs_action = np.vstack([np.hstack([new_pos, 0]), np.hstack([new_vel, 0])])
        return abs_action

    def reset(self, obs):
        self.last_acceleration = np.repeat(0.0, 6)
        self.interp_pos = obs[self.env_info["joint_pos_ids"]][:-1]
        self.interp_vel = obs[self.env_info["joint_vel_ids"]][:-1]

    def filter_obs(self, obs):
        return np.hstack([obs[0:2], obs[3:5], obs[6:12], obs[13:19]])
    
class PrepareAgent(PPOAgent):
    def __init__(self, env_info, path):
        super().__init__(env_info, path)
    
    def reset(self, obs):
        super().reset(obs)
        self.counter = 0
        self.done = False

    def draw_action(self, obs):
        self.counter += 1
        if self.counter > 75:
            self.done = True

        puck_pos = obs[self.env_info["puck_pos_ids"]].copy()
        puck_vel = obs[self.env_info["puck_vel_ids"]].copy()
        puck_pos[0] -= 1.51
        if np.abs(puck_vel[:2]).max() > 0.2 or (abs(puck_pos[1]) < 0.41 and puck_pos[0] > -0.8) or (puck_pos[0] > -0.5):
            self.done = True
        
        return super().draw_action(obs)


    
class ResetAgent:
    def __init__(self, env_info):
        self.env_info = env_info

    def draw_action(self, obs):
        if self.first_stop:
            if np.max(np.abs(obs[self.env_info["joint_vel_ids"]])) < 0.05:
                self.first_stop = False
            else:
                self.dq *= 0.75
                natural_q = self.q + self.dq * 0.02
                ee_pos = forward_kinematics(self.env_info["robot"]["robot_model"], self.env_info["robot"]["robot_data"], natural_q)[0]
                ee_pos[2] = 0.1645
                proper_q = inverse_kinematics(self.env_info["robot"]["robot_model"], self.env_info["robot"]["robot_data"], ee_pos, initial_q=self.q)[1]
                diff_q = (proper_q - self.q) / 0.02
                lims = self.env_info["robot"]["joint_vel_limit"] * 0.8
                while not np.all(np.abs(diff_q) < lims[1]):
                    diff_q *= 0.95
                self.dq = self.dq * 0.5 + diff_q * 0.5
                self.q += self.dq * 0.02
                return np.vstack([self.q, self.dq])

        if self.roll_to_count or self.at_init:
            self.roll_count -= 1
            if self.roll_count == 0:
                self.at_init = True
            return np.vstack([self.target_q, [0 for _ in range(7)]])
        diff_q = obs[self.env_info["joint_pos_ids"]] - self.target_q
        if np.linalg.norm(diff_q) < 0.01:
            self.roll_to_count = True

        joint_ret_dq = (self.target_q - self.q) / 0.02
        lims = self.env_info["robot"]["joint_vel_limit"] * 0.65
        while not (np.abs(joint_ret_dq) - lims[1] < 0).all() or (np.abs(joint_ret_dq) - np.abs(self.dq) > 0.15).any():
            joint_ret_dq *= 0.95
        des_q = self.q + joint_ret_dq * 0.02
        ee_pos = forward_kinematics(self.env_info["robot"]["robot_model"], self.env_info["robot"]["robot_data"], des_q)[0]
        ee_pos[2] = np.clip(ee_pos[2], 0.16, 0.169)
        safe_des_q = inverse_kinematics(self.env_info["robot"]["robot_model"], self.env_info["robot"]["robot_data"], ee_pos, initial_q=des_q)[1]
        diff_q = (safe_des_q - self.q) / 0.02
        while not (np.abs(diff_q) - lims[1] < 0).all():
            diff_q *= 0.95
        self.dq = self.dq * 0.35 + diff_q * 0.65
        self.q += self.dq * 0.02
        return np.vstack([self.q, self.dq])
    
    def reset(self, obs):
        self.at_init = False
        self.first_stop = False
        self.roll_to_count = False
        self.roll_count = 3
        self.q = obs[self.env_info["joint_pos_ids"]]
        self.dq = obs[self.env_info["joint_vel_ids"]]
        self.ee_pos = forward_kinematics(self.env_info["robot"]["robot_model"], self.env_info["robot"]["robot_data"], self.q)[0]
        self.target_ee = np.array([0.65, 0, 0.1645])
        self.target_q = np.array([0., -0.1961, 0., -1.8436, 0., 0.9704, 0.])

class StateMachine:
    def __init__(self, env_info, ik, prepare_agent):
        self.env_info = env_info
        self.side_margin = 0.41
        self.ik = ik
        self.prepare_agent = prepare_agent

    def reset(self, obs):
        self.state = "ik"
    
    def update_state(self, obs):
        ee_pos = forward_kinematics(self.env_info["robot"]["robot_model"], self.env_info["robot"]["robot_data"], obs[self.env_info["joint_pos_ids"]])[0]
        if self.state == "hit":
            puck_pos = obs[self.env_info["puck_pos_ids"]]
            puck_vel = obs[self.env_info["puck_vel_ids"]]
            if puck_pos[0] - 1.51 > -0.2 or (puck_pos[0] - 1.51) + puck_vel[0] * 0.5 > -0.2:
                self.state = "ik"
                return True
            if puck_pos[0] <= ee_pos[0] or (abs(puck_pos[1]) > self.side_margin or abs(puck_pos[1] + puck_vel[1] * 0.75) > self.side_margin):
                self.state = "ik"
                return True
        elif self.state == "defend":
            puck_pos = obs[self.env_info["puck_pos_ids"]]
            puck_vel = obs[self.env_info["puck_vel_ids"]]
            if (puck_vel[0] > -0.2) or (puck_pos[0] < ee_pos[0]):
                self.state = "ik"
                return True
        elif self.state == "prepare":
            puck_pos = obs[self.env_info["puck_pos_ids"]]
            puck_vel = obs[self.env_info["puck_vel_ids"]]
            if self.prepare_agent.done:
                self.state = "ik"
                return True
        elif self.state == "ik":
            if self.ik.at_init:
                puck_pos = obs[self.env_info["puck_pos_ids"]]
                puck_vel = obs[self.env_info["puck_vel_ids"]]
                if (puck_pos[0] - 1.51 < -0.2 and np.abs(puck_vel[:2]).max() < 0.05) and ((puck_pos[0] - 1.51 <= -0.8) or (abs(puck_pos[1]) > self.side_margin)):
                    self.state = "prepare"
                    return True
                if ((puck_pos[0] - 1.51 < 0.3 and puck_vel[0] < -0.5) or puck_vel[0] < -1.5) and ee_pos[0] < puck_pos[0]:
                    self.state = "defend"
                    return True
                if (puck_pos[0] - 1.51 < -0.2 and (puck_pos[0] - 1.51) + puck_vel[0] * 1 < -0.2) and puck_vel[0] < 0.5 and abs(puck_vel[1]) < 0.5 and not (abs(puck_pos[1]) > self.side_margin or abs(puck_pos[1] + puck_vel[1] * 0.75) > self.side_margin) and (puck_pos[0] - 1.51) + puck_vel[0] * 0.75 > -0.8:
                    self.state = "hit"
                    return True 
        return False
    
class PPOBaselineAgent(AgentBase):
    def __init__(self, env_info, agent_id=1, **kwargs):
        super().__init__(env_info, agent_id, **kwargs)
        self.env_info = env_info
        self.tracker = PuckTracker(env_info)

        trained_models_dir = Path(__file__).parent / "trained_models"

        self.ik = ResetAgent(env_info)
        self.hit_agent = PPOAgent(env_info, trained_models_dir / "hit")
        self.defend_agent = PPOAgent(env_info, trained_models_dir / "defend")
        self.prepare_agent = PrepareAgent(env_info, trained_models_dir / "prepare")
        self.sm = StateMachine(env_info, self.ik, self.prepare_agent)
        self.reset()
    
    def reset_state(self, observation):
        self.tracker.reset(observation[self.env_info["puck_pos_ids"]])
        self.last_obs = observation

    def reset(self):
        self.step = 0

    def draw_action(self, original_obs):
        if self.step == 0:
            self.reset_state(original_obs)
            self.sm.reset(original_obs)
            self.ik.reset(original_obs)

        passed_obs = original_obs.copy()
        if np.all(passed_obs[self.env_info["puck_vel_ids"]] == 0):
            state, _, _ = self.tracker.get_prediction(0.02)
            self.tracker.step(state[[0,1,4]])
        else:
            self.tracker.step(passed_obs[self.env_info["puck_pos_ids"]])

        passed_obs[self.env_info["puck_pos_ids"]] = self.tracker.state[[0,1,4]]
        passed_obs[self.env_info["puck_vel_ids"]] = self.tracker.state[[2,3,5]]
        
        if self.sm.update_state(passed_obs):
            if self.sm.state == "hit":
                self.hit_agent.reset(passed_obs)
            elif self.sm.state == "defend": 
                self.defend_agent.reset(passed_obs)
            elif self.sm.state == "ik": 
                self.ik.reset(passed_obs)
            elif self.sm.state == "prepare":
                self.prepare_agent.reset(passed_obs)
        
        if self.sm.state == "hit":
            action = self.hit_agent.draw_action(passed_obs)
        elif self.sm.state == "defend": 
            action = self.defend_agent.draw_action(passed_obs)
        elif self.sm.state == "ik": 
            action = self.ik.draw_action(passed_obs)
        elif self.sm.state == "prepare":
            action = self.prepare_agent.draw_action(passed_obs)

        self.step += 1

        action[0][-1] = 0
        action[1][-1] = 0

        return action

if __name__ == "__main__":
    from air_hockey_challenge.framework import AirHockeyChallengeWrapper
    from pathlib import Path

    # path = Path(
    #     "/home/donat/projects/air_hockey_challenge/baseline/ppo_baseline_agent/sbx_checkpoints/defend/defend-new"
    # )

    # # path = Path(
    # #     "/home/donat/projects/air_hockey_challenge/baseline/ppo_baseline_agent/sbx_checkpoints/hit/hit-41"
    # # )
    
    path = Path(
        "/home/donat/projects/air_hockey_challenge/baseline/ppo_baseline_agent/sbx_checkpoints/prepare/prepare-19"
    )
    env = AirHockeyChallengeWrapper(env="prepare")
    env.seed(2)
    agent = PPOAgent(env_info=env.env_info, path=path)
    # # # agent = BaselineAgent(env_info=env.env_info, agent_id=1, only_tactic="defend")

    obs = env.reset()
    agent.reset(obs)
    # # agent.reset()

    done = False

    env.render()

    while True:
        for _ in range(300):
            action = agent.draw_action(obs)
            obs, reward, done, info = env.step(action)
            env.render()
            if done:
                if env.check_success(obs):
                    print("Success!")
                break

        obs = env.reset()
        agent.reset(obs)
        # agent.reset()
        env.render()
