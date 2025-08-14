import jax
from air_hockey_challenge.environments.brax_envs.env_single import AirHockeySingle
from jax import numpy as jnp
from brax.envs.base import State


class AirHockeyHitSingle(AirHockeySingle):
    def __init__(self, moving_init=True, timestep=1 / 1000, n_intermediate_steps=20):
        super().__init__(timestep=timestep, n_intermediate_steps=n_intermediate_steps)

        self.moving_init = moving_init
        hit_width = (
            self.env_info["table"]["width"] / 2
            - self.env_info["puck"]["radius"]
            - self.env_info["mallet"]["radius"] * 2
        )
        self.hit_range = jnp.array(
            [[-0.7, -0.2], [-hit_width, hit_width]]
        )  # Table Frame
        self.init_velocity_range = (0, 0.5)  # Table Frame

    def _setup(self, rng: jax.Array) -> tuple[jax.Array, jax.Array]:
        qpos, qvel = super()._setup(rng)

        rng, rng1 = jax.random.split(rng, 2)

        puck_pos = (
            jax.random.uniform(rng1, shape=(2,))
            * (self.hit_range[:, 1] - self.hit_range[:, 0])
            + self.hit_range[:, 0]
        )

        qpos = qpos.at[self.puck_ids[:2]].set(puck_pos)

        if self.moving_init:
            def cond_fun(val):
                rng, puck_pos, puck_vel = val
                return jnp.abs(puck_vel[1] * 1 + puck_pos[1]) >= 0.43

            def body_fun(val):
                rng, puck_pos, puck_vel = val
                rng, rng2, rng3, rng4 = jax.random.split(rng, 4)
                lin_vel = jax.random.uniform(rng2, minval=self.init_velocity_range[0], maxval=self.init_velocity_range[1])
                angle = jax.random.uniform(rng3, minval=-jnp.pi / 2 - 0.1, maxval=jnp.pi / 2 + 0.1)
                puck_vel = puck_vel.at[0].set(-jnp.cos(angle) * lin_vel)
                puck_vel = puck_vel.at[1].set(jnp.sin(angle) * lin_vel)
                puck_vel = puck_vel.at[2].set(jax.random.uniform(rng4, minval=-2, maxval=2))
                return rng, puck_pos, puck_vel
            
            puck_vel = jnp.zeros(3)
            rng, _, puck_vel = body_fun((rng, puck_pos, puck_vel)) # Initial call to set puck velocity
            rng, _, puck_vel = jax.lax.while_loop(
                cond_fun, body_fun, (rng, puck_pos, puck_vel)
            )
            qvel = qvel.at[self.puck_ids].set(puck_vel)

        return qpos, qvel
    
    def _is_absorbing(self, state: State) -> tuple[State, jax.Array]:
        puck_pos, puck_vel = self.get_puck(state.info["internal_obs"])
        
        parent_done = super()._is_absorbing(state)

        done = jnp.logical_or(
            jnp.logical_and(puck_pos[0] > 0, puck_vel[0] < 0), # if puck bounces back on the opponent's wall
            parent_done[1],
        )

        return state, done * 1.0