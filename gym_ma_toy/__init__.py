from gym.envs.registration import register

register(
    id="team_catcher-v0",
    entry_point="gym_ma_toy.envs:TeamCatcher",
)