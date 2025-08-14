import numpy as np

from air_hockey_challenge.environments.iiwas.env_single import AirHockeySingle


class AirHockeyHitSingle(AirHockeySingle):
    """
        Class for the air hockey hitting task.
    """
    def __init__(self, gamma=0.99, horizon=500, moving_init=True, viewer_params={}, **kwargs):
        """
            Constructor
            Args:
                moving_init(bool, False): If true, initialize the puck with inital velocity.
        """

        super().__init__(gamma=gamma, horizon=horizon, viewer_params=viewer_params)

        self.moving_init = moving_init
        hit_width = self.env_info['table']['width'] / 2 - self.env_info['puck']['radius'] - \
                    self.env_info['mallet']['radius'] * 2
        self.hit_range = np.array([[-0.7, -0.2], [-hit_width, hit_width]])  # Table Frame
        self.init_velocity_range = (0, 0.5)  # Table Frame


    def setup(self, obs):
        # Initial position of the puck
        puck_pos = np.random.rand(2) * (self.hit_range[:, 1] - self.hit_range[:, 0]) + self.hit_range[:, 0]

        self._write_data("puck_x_pos", puck_pos[0])
        self._write_data("puck_y_pos", puck_pos[1])

        if self.moving_init:
            while True:
                lin_vel = np.random.uniform(self.init_velocity_range[0], self.init_velocity_range[1])
                angle = np.random.uniform(-np.pi / 2 - 0.1, np.pi / 2 + 0.1)
                puck_vel = np.zeros(3)
                puck_vel[0] = -np.cos(angle) * lin_vel
                puck_vel[1] = np.sin(angle) * lin_vel
                puck_vel[2] = np.random.uniform(-2, 2, 1)
                if np.abs(puck_vel[1] * 1 + puck_pos[1]) < 0.43:
                    break

            self._write_data("puck_x_vel", puck_vel[0])
            self._write_data("puck_y_vel", puck_vel[1])
            self._write_data("puck_yaw_vel", puck_vel[2])

        super(AirHockeyHitSingle, self).setup(obs)

    def reward(self, state, action, next_state, absorbing):
        return 0

    def is_absorbing(self, obs):
        puck_pos, puck_vel = self.get_puck(obs)
        # Stop if the puck bounces back on the opponents wall
        if puck_pos[0] > 0 and puck_vel[0] < 0:
            return True
        return super(AirHockeyHitSingle, self).is_absorbing(obs)


if __name__ == '__main__':
    env = AirHockeyHitSingle(moving_init=True)
    env.reset()

    steps = 0
    while True:
        action = np.zeros(7)

        observation, reward, done, info = env.step(action)
        env.render()
        if done or steps > env.info.horizon:
            steps = 0
            env.reset()
