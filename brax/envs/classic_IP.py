import jax
import jax.numpy as jp
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils import epath


class InvertedPendulum(PipelineEnv):
  """
  A single pendulum anchored at the origin with continuous torque in [-2, 2],
  matching the MuJoCo model defined in 'classic_IP.xml'.

  The pendulum's joint is 'sphere_joint', with axis=(0,1,0). That means it
  rotates in the x-z plane, and the angle is stored in pipeline_state.q[0].
  We define:

    theta = q[0],  theta_dot = qd[0].

  Observations:
    [cos(theta), sin(theta), theta_dot]

  Reward each step:
    R = -( theta^2 + 0.1*theta_dot^2 + 0.001*torque^2 )

  Hinge angle is limited by the XML's ctrlrange to [-2,2] torque.
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

    # Random initial angle theta and angular velocity theta_dot
    theta = jax.random.uniform(
        rng1, shape=(self.sys.q_size(),), minval=-jp.pi, maxval=jp.pi
    )
    theta_dot = jax.random.uniform(
        rng2, shape=(self.sys.qd_size(),), minval=-8.0, maxval=8.0
    )

    # Brax pipeline state uses (q, qd)
    q = self.sys.init_q + theta
    qd = theta_dot
    pipeline_state = self.pipeline_init(q, qd)

    # Create the first observation
    obs = self._get_obs(q, qd)
    reward = jp.zeros(())
    done = jp.zeros(())
    metrics = {}

    return State(pipeline_state, obs, reward, done, metrics)

  def step(self, state: State, action: jax.Array) -> State:
    """Applies a torque in [-2,2], advances physics, and returns new state."""
    # Clip input to our allowed torque range:
    ctrl_range = self.sys.actuator.ctrl_range  # shape [1,2] 
    torque = jp.clip(action, ctrl_range[:, 0], ctrl_range[:, 1])

    pipeline_state = self.pipeline_step(state.pipeline_state, torque)
    q = pipeline_state.q      # [theta]
    qd = pipeline_state.qd    # [theta_dot]

    obs = self._get_obs(q, qd)

    theta = q[0]
    theta_dot = qd[0]
    cost = theta**2 + 0.1 * (theta_dot**2) + 0.001 * (torque**2)
    reward = -cost.squeeze()

    done = jp.zeros(())
    return state.replace(
        pipeline_state=pipeline_state,
        obs=obs,
        reward=reward,
        done=done
    )

  def _get_obs(self, q: jp.ndarray, qd: jp.ndarray) -> jp.ndarray:
    """
    Observations = [ cos(theta), sin(theta), theta_dot ],
    where q[0] = theta, qd[0] = theta_dot.
    """
    theta = q[0]
    norm_theta = (theta + jp.pi) % (2.0 * jp.pi) - jp.pi
    theta_dot = qd[0]
    clamped_theta_dot = jp.clip(theta_dot, -8.0, 8.0)
    x = jp.cos(norm_theta)
    y = jp.sin(norm_theta)
    return jp.array([x, y, clamped_theta_dot])

  @property
  def action_size(self) -> int:
    """We have a single torque actuator in the XML => 1D action."""
    return 1
