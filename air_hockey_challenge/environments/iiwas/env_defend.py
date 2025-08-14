import numpy as np

from air_hockey_challenge.environments.iiwas.env_single import AirHockeySingle


class AirHockeyDefend(AirHockeySingle):
    """
        Class for the air hockey defending task.
        The agent should stop the puck at the line x=-0.6.
    """
    def __init__(self, gamma=0.99, horizon=500, viewer_params={}, **kwargs):
        self.init_velocity_range = (1, 5)

        self.start_range = np.array([[0.4, 0.75], [-0.4, 0.4]])  # Table Frame
        super().__init__(gamma=gamma, horizon=horizon, viewer_params=viewer_params)

    def setup(self, obs):
        puck_pos = np.random.rand(2) * (self.start_range[:, 1] - self.start_range[:, 0]) + self.start_range[:, 0]

        lin_vel = np.random.uniform(self.init_velocity_range[0], self.init_velocity_range[1])
        angle = np.random.uniform(-0.75, 0.75)

        puck_vel = np.zeros(3)
        puck_vel[0] = -np.cos(angle) * lin_vel
        puck_vel[1] = np.sin(angle) * lin_vel
        puck_vel[2] = np.random.uniform(-10, 10, 1)

        self._write_data("puck_x_pos", puck_pos[0])
        self._write_data("puck_y_pos", puck_pos[1])
        self._write_data("puck_x_vel", puck_vel[0])
        self._write_data("puck_y_vel", puck_vel[1])
        self._write_data("puck_yaw_vel", puck_vel[2])

        super(AirHockeyDefend, self).setup(obs)

    def reward(self, state, action, next_state, absorbing):
        return 0

    def is_absorbing(self, state):
        puck_pos, puck_vel = self.get_puck(state)
        # If puck is over the middle line and moving towards opponent
        if puck_pos[0] > 0 and puck_vel[0] > 0:
            return True
        if np.linalg.norm(puck_vel[:2]) < 0.1:
            return True
        return super().is_absorbing(state)