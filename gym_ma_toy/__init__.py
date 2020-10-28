from gym.envs.registration import register
from .envs.coop import team_catcher_base

# No mobile targets
register(
    id="team_catcher-v0",
    entry_point="gym_ma_toy.envs:TeamCatcherBase",
    kwargs={
        "grid_size": 64,
        "nb_agents_hv": 256,
        "nb_agents_diag": 0,
        "nb_targets": 128,
        "nb_mobiles": 0,
    },
)

# Mobile targets
register(
    id="team_catcher-v1",
    entry_point="gym_ma_toy.envs:TeamCatcherBase",
    kwargs={
        "grid_size": 64,
        "nb_agents_hv": 256,
        "nb_agents_diag": 0,
        "nb_targets": 128,
        "nb_mobiles": 32,
    },
)

# Mixed Agents  and mobile targets
register(
    id="team_catcher-v2",
    entry_point="gym_ma_toy.envs:TeamCatcherBase",
    kwargs={
        "grid_size": 64,
        "nb_agents_hv": 128,
        "nb_agents_diag": 128,
        "nb_targets": 128,
        "nb_mobiles": 32,
    },
)
