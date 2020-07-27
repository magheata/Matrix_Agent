from gym.envs.registration import register

register(
    id='MatrixEnv-v0',
    entry_point='env.MatrixEnv:Matrix'
)
