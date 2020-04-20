from gym.envs.registration import register

register(
    id='MatrixEnv-v0',
    entry_point='env.MatrixEnv:Matrix',
    kwargs={'actions': 4, 'dimension': 5, 'goal': (1, 4)}
)
