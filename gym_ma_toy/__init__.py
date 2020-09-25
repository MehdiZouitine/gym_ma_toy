from gym.envs.registration import register

register(
    id='toy_grid_coop-v0',
    entry_point='gym_ma_toy.envs:ToyGridCoop',
)