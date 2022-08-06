from gym.envs.registration import register

# Mujoco
# ----------------------------------------
register(
    'SwimmerVel-v2',
    entry_point='envs.utils.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'envs.mujoco.swimmer:SwimmerVelEnv'}
)

register(
    'AntVel-v3',
    entry_point='envs.utils.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'envs.mujoco.ant:AntVelEnv'}
)

register(
    'AntDir-v3',
    entry_point='envs.utils.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'envs.mujoco.ant:AntDirEnv'}
)

register(
    'AntPos-v2',
    entry_point='envs.utils.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'envs.mujoco.ant:AntPosEnv'}
)

register(
    'HalfCheetahVel-v3',
    entry_point='envs.utils.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'envs.mujoco.half_cheetah:HalfCheetahVelEnv'}
)

register(
    'HalfCheetahDir-v3',
    entry_point='envs.utils.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'envs.mujoco.half_cheetah:HalfCheetahDirEnv'}
)

# 2D Navigation
# ----------------------------------------

register(
    '2DNavigation-v1',
    entry_point='envs.navigation:Navigation2DEnv',
    max_episode_steps=100
)
