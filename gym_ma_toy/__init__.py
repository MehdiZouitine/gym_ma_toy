from gym.envs.registration import register
from .envs.coop import team_catcher_base

# No mobile targets
register(
    id="team_catcher-v0",
    entry_point="gym_ma_toy.envs:TeamCatcherBase",
    kwargs={
        "grid_size": 64,
        "nb_agents_hv": 16,
        "nb_agents_diag": 0,
        "nb_targets": 32,
        "nb_mobiles": 0,
    },
)

# Mobile targets
register(
    id="team_catcher-v1",
    entry_point="gym_ma_toy.envs:TeamCatcherBase",
    kwargs={
        "grid_size": 64,
        "nb_agents_hv": 16,
        "nb_agents_diag": 0,
        "nb_targets": 16,
        "nb_mobiles": 16,
    },
)

# Mixed Agents  and mobile targets
register(
    id="team_catcher-v2",
    entry_point="gym_ma_toy.envs:TeamCatcherBase",
    kwargs={
        "grid_size": 64,
        "nb_agents_hv": 8,
        "nb_agents_diag": 8,
        "nb_targets": 16,
        "nb_mobiles": 16,
    },
)

# Partially observable
register(
    id="team_catcher-v3",
    entry_point="gym_ma_toy.envs:TeamCatcherBase",
    kwargs={
        "grid_size": 64,
        "nb_agents_hv": 8,
        "nb_agents_diag": 8,
        "nb_targets": 16,
        "nb_mobiles": 16,
        "fow_agents_hv": 2,
        "fow_agents_diag": 4,
    },
)
