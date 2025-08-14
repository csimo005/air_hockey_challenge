from air_hockey_challenge.environments.brax_envs.env_single import AirHockeySingle
from jax import numpy as jnp
import jax
from brax.envs.base import State


class AirHockeyDefend(AirHockeySingle):
    def __init__(self, timestep=1 / 1000, n_intermediate_steps=20):
        super().__init__(timestep=timestep, n_intermediate_steps=n_intermediate_steps)

        self.init_velocity_range = (1, 5)
        self.start_range = jnp.array([[0.4, 0.75], [-0.4, 0.4]])  # Table Frame

    def _setup(self, rng: jax.Array) -> tuple[jax.Array, jax.Array]:
        qpos, qvel = super()._setup(rng)

        rng, rng1, rng2, rng3, rng4 = jax.random.split(rng, 5)

        puck_pos = (
            jax.random.uniform(rng1, shape=(2))
            * (self.start_range[:, 1] - self.start_range[:, 0])
            + self.start_range[:, 0]
        )

        lin_vel = jax.random.uniform(
            rng2, minval=self.init_velocity_range[0], maxval=self.init_velocity_range[1]
        )
        angle = jax.random.uniform(rng3, minval=-0.75, maxval=0.75)

        puck_vel = jnp.zeros(3)
        puck_vel = puck_vel.at[0].set(-jnp.cos(angle) * lin_vel)
        puck_vel = puck_vel.at[1].set(jnp.sin(angle) * lin_vel)
        puck_vel = puck_vel.at[2].set(jax.random.uniform(rng4, minval=-10, maxval=10))

        qpos = qpos.at[self.puck_ids[:2]].set(puck_pos)
        qvel = qvel.at[self.puck_ids].set(puck_vel)

        return qpos, qvel

    def _is_absorbing(self, state: State)-> tuple[State, jax.Array]:
        puck_pos, puck_vel = self.get_puck(state.info["internal_obs"])

        parent_done = super()._is_absorbing(state)

        done = jnp.logical_or.reduce(
            jnp.array([
                parent_done[1],
                jnp.logical_and(puck_pos[0] > 0, puck_vel[0] > 0), # If puck is over the middle line and moving towards opponent
                jnp.linalg.norm(puck_vel[:2]) < 0.1, # If puck is almost stationary
            ])
        )

        return state, done * 1.0