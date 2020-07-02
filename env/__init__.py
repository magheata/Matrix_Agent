from gym.envs.registration import register

register(
    id='MatrixEnv-v0',
    entry_point='env.MatrixEnv:Matrix',
    kwargs={'dimension': 5, 'goal': (1, 4), 'start_state': (0, 0)}
)
