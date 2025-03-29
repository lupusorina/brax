# inverted_pendulum.py

import jax
import jax.numpy as jp
from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils import epath

class InvertedPendulum(PipelineEnv):
  """
  A single pendulum anchored at the origin with continuous torque in [-2, 2].
  
  State = [ x, y, theta_dot ], where:
    - x = cos(theta),
    - y = sin(theta),
    - theta_dot = angular velocity.
  
  Reward each step = - [ theta^2 + 0.1 * theta_dot^2 + 0.001 * torque^2 ],
  with hinge angle limited to [-8, 8].
  """

  def __init__(self, backend='generalized', **kwargs):
    xml_path = epath.resource_path('brax') / 'envs/assets/classic_IP.xml'
    sys = mjcf.load(xml_path)

    n_frames = 2

    if backend in ['spring', 'positional']:
      sys = sys.tree_replace({'opt.timestep': 0.005})
      n_frames = 4

    kwargs['n_frames'] = kwargs.get('n_frames', n_frames)

    super().__init__(sys=sys, backend=backend, **kwargs)

  def reset(self, rng: jax.Array) -> State:
    """Resets the pendulum to a random angle in [-pi, pi], velocity in [-8, 8]."""
    rng, rng1, rng2 = jax.random.split(rng, 3)

    theta = jax.random.uniform(rng1, shape=(self.sys.q_size()), minval=-jp.pi, maxval=jp.pi)
    theta_dot = jax.random.uniform(rng2, shape=(self.sys.qd_size()), minval=-8.0, maxval=8.0)

    # Brax pipeline state uses (q, qd)
    q = self.sys.init_q + theta
    qd = theta_dot
    pipeline_state = self.pipeline_init(q, qd)

    # Create the first observation
    obs = self._get_obs(q, qd)
    reward,done = jp.zeros(2)
    metrics = {}

    return State(pipeline_state, obs, reward, done, metrics)

  def step(self, state: State, action: jax.Array) -> State:
    """Applies a torque in [-2,2], advances physics, and returns new state."""
    # Clip input to our allowed torque range:
    action_min = self.sys.actuator.ctrl_range[:,0]
    action_max = self.sys.actuator.ctrl_range[:,1]
    torque = jp.clip(action, min=action_min, max=action_max)

    pipeline_state = self.pipeline_step(state.pipeline_state, torque)
    q = pipeline_state.q      # [theta]
    qd = pipeline_state.qd    # [theta_dot]
    obs = self._get_obs(q, qd)

    theta     = q[0]
    theta_dot = qd[0]
    
    # Reward = -( theta^2 + 0.1*theta_dot^2 + 0.001*torque^2 )
    cost = theta**2 + 0.1 * (theta_dot**2) + 0.001 * (torque**2)
    reward = -cost.squeeze(-1)

    done = 0.

    return state.replace(
        pipeline_state=pipeline_state,
        obs=obs,
        reward=reward,
        done=done
    )

  def _get_obs(self, q, qd) -> jp.ndarray:
    """Returns [ x, y, theta_dot ], with x=cos(theta), y=sin(theta)."""
    theta = q[0]
    theta_dot = qd[0]
    x = jp.cos(theta)
    y = jp.sin(theta)
    return jp.array([x, y, theta_dot])

  @property
  def action_size(self) -> int:
    """We have 1D action (the torque)."""
    return 1
