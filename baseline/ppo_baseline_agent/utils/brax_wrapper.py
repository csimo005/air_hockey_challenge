from typing import Optional

import jax
import numpy as np
from brax.envs.base import PipelineEnv, Wrapper, State
from brax.io import image as brax_image
from gymnasium import spaces

from stable_baselines3.common.vec_env import VecEnv
import copy
from jax import numpy as jnp

# from https://gist.github.com/araffin/a7a576ec1453e74d9bb93120918ef7e7
class BraxSB3Wrapper(VecEnv):
    """A wrapper that converts batched Brax Env to one that follows SB3 VecEnv API."""

    def __init__(
        self,
        env: PipelineEnv,
        seed: int = 0,
        backend: Optional[str] = None,
        keep_infos: bool = True,
    ) -> None:
        self._env = env
        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "video.frames_per_second": 1 / self._env.dt,
        }
        self.render_mode = "rgb_array"
        if not hasattr(self._env, "batch_size"):
            raise ValueError("underlying env must be batched")

        self.num_envs = self._env.batch_size
        self.seed(seed)
        self.backend = backend
        self._state = None
        self.keep_infos = keep_infos
        self.default_infos = [{} for _ in range(self.num_envs)]

        obs = np.inf * np.ones(self._env.observation_size, dtype=np.float32)
        self.observation_space = spaces.Box(-obs, obs, dtype=np.float32)

        action = jax.tree.map(np.array, self._env.sys.actuator.ctrl_range)
        self.action_space = spaces.Box(action[:, 0], action[:, 1], dtype=np.float32)

        def reset(key):
            key1, key2 = jax.random.split(key)
            state = self._env.reset(key2)
            return state, state.obs, key1

        self._reset = jax.jit(reset, backend=self.backend)

        def step(state, action):
            state = self._env.step(state, action)
            # Note: they don't seem to handle truncation properly
            info = {**state.metrics, **state.info}
            return state, state.obs, state.reward, state.done, info

        self._step = jax.jit(step, backend=self.backend)

    def reset(self) -> np.ndarray:
        self._state, obs, self._key = self._reset(self._key)
        return np.array(obs)

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        # TODO: add last observation too?
        self._state, obs, rewards, dones, info = self._step(self._state, self.actions)
        # Convert from dict of list to list of dicts
        if self.keep_infos:
            # May be slow with many envs
            infos = self.to_list(info)
        else:
            infos = self.default_infos

        return np.array(obs), np.array(rewards), np.array(dones).astype(bool), infos

    def seed(self, seed: int = 0) -> None:
        self._key = jax.random.PRNGKey(seed)

    def render(self, mode: str = "human") -> None:
        if mode == "rgb_array":
            if self._state is None:
                raise RuntimeError("Must call reset or step before rendering")
            return brax_image.render_array(
                self._env.sys, self._state.pipeline_state, 256, 256
            )
        else:
            # Use opencv to render
            return super().render(mode="human")

    def get_images(self):
        state_list = [self._state.take(i).pipeline_state for i in range(self.num_envs)]
        return brax_image.render_array(self._env.sys, state_list, width=256, height=256)

    def env_is_wrapped(self, wrapper_class, indices=None):
        # For compatibility with eval and monitor helpers
        return [False]

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError("Setting attributes is not supported.")

    def get_attr(self, attr_name, indices=None):
        # resolve indices
        if indices is None:
            indices = slice(None)
            num_indices = self.num_envs
        else:
            num_indices = len(indices)
        attr_val = getattr(self, attr_name)
        return [attr_val] * num_indices

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        raise NotImplementedError()

    def close(self) -> None:
        pass

    def to_list(self, info_dict: dict):
        infos = [dict.fromkeys(info_dict.keys()) for _ in range(self.num_envs)]
        # From https://github.com/isaac-sim/IsaacLab
        # fill-in information for each sub-environment
        # note: This loop becomes slow when number of environments is large.
        for idx in range(self.num_envs):
            # fill-in bootstrap information
            # TODO: use "truncation" key
            # infos[idx]["TimeLimit.truncated"] = truncated[idx] and not terminated[idx]
            # TODO: use first-obs?
            # infos[idx]["terminal_observation"] = None
            # fill-in information from extras
            for key, value in info_dict.items():
                try:
                    infos[idx][key] = value[idx]
                except TypeError:
                    # Note: doesn't work for State object
                    pass
        # return list of dictionaries
        return infos
    
class AutoResetWrapper(Wrapper):
  """Automatically resets Brax envs that are done."""

  def reset(self, rng: jax.Array) -> State:
    state = self.env.reset(rng)
    state.info['first_pipeline_state'] = state.pipeline_state
    state.info['first_obs'] = state.obs
    state.info['first_info'] = copy.deepcopy(state.info)
    return state

  def step(self, state: State, action: jax.Array) -> State:
    if 'steps' in state.info:
      steps = state.info['steps']
      steps = jnp.where(state.done, jnp.zeros_like(steps), steps)
      state.info.update(steps=steps)
    state = state.replace(done=jnp.zeros_like(state.done))
    state = self.env.step(state, action)

    def where_done(x, y):
      done = state.done
      if done.shape and done.shape[0] != x.shape[0]:
        return y
      if done.shape:
        done = jnp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
      return jnp.where(done, x, y)

    pipeline_state = jax.tree.map(
        where_done, state.info['first_pipeline_state'], state.pipeline_state
    )
    obs = jax.tree.map(where_done, state.info['first_obs'], state.obs)
    info = jax.tree.map(
        where_done, state.info['first_info'] | {'first_info': state.info['first_info']}, state.info
    )
    return state.replace(pipeline_state=pipeline_state, obs=obs, info=info)