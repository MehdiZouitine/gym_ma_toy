from gymnasium.envs.registration import register
# from .envs import team_catcher_base

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


register(
    id="team_catcher-easy-lvl0-v0",
    entry_point="gym_ma_toy.envs:TeamCatcherBase",
    kwargs={
        "grid_size": 16,
        "nb_agents_hv": 16,
        "nb_agents_diag": 0,
        "nb_targets": 32,
        "nb_mobiles": 0,
    },
)

register(
    id="team_catcher-easy-lvl1-v0",
    entry_point="gym_ma_toy.envs:TeamCatcherBase",
    kwargs={
        "grid_size": 16,
        "nb_agents_hv": 8,
        "nb_agents_diag": 0,
        "nb_targets": 8,
        "nb_mobiles": 0,
    },
)


register(
    id="team_catcher-easy-lvl2-v0",
    entry_point="gym_ma_toy.envs:TeamCatcherBase",
    kwargs={
        "grid_size": 16,
        "nb_agents_hv": 2,
        "nb_agents_diag": 0,
        "nb_targets": 2,
        "nb_mobiles": 0,
    },
)


register(
    id="team_catcher-medium-lvl0-v0",
    entry_point="gym_ma_toy.envs:TeamCatcherBase",
    kwargs={
        "grid_size": 64,
        "nb_agents_hv": 32,
        "nb_agents_diag": 0,
        "nb_targets": 64,
        "nb_mobiles": 0,
    },
)


register(
    id="team_catcher-medium-lvl1-v0",
    entry_point="gym_ma_toy.envs:TeamCatcherBase",
    kwargs={
        "grid_size": 64,
        "nb_agents_hv": 32,
        "nb_agents_diag": 16,
        "nb_targets": 64,
        "nb_mobiles": 32,
    },
)


register(
    id="team_catcher-medium-lvl2-v0",
    entry_point="gym_ma_toy.envs:TeamCatcherBase",
    kwargs={
        "grid_size": 64,
        "nb_agents_hv": 4,
        "nb_agents_diag": 4,
        "nb_targets": 4,
        "nb_mobiles": 4,
    },
)


register(
    id="team_catcher-hard-lvl0-v0",
    entry_point="gym_ma_toy.envs:TeamCatcherBase",
    kwargs={
        "grid_size": 32,
        "nb_agents_hv": 8,
        "nb_agents_diag": 8,
        "nb_targets": 8,
        "nb_mobiles": 8,
        "fow_agents_hv": 2,
        "fow_agents_diag": 4,
    },
)

register(
    id="team_catcher-hard-lvl1-v0",
    entry_point="gym_ma_toy.envs:TeamCatcherBase",
    kwargs={
        "grid_size": 84,
        "nb_agents_hv": 32,
        "nb_agents_diag": 16,
        "nb_targets": 64,
        "nb_mobiles": 32,
        "fow_agents_hv": 2,
        "fow_agents_diag": 4,
    },
)

register(
    id="team_catcher-hard-lvl2-v0",
    entry_point="gym_ma_toy.envs:TeamCatcherBase",
    kwargs={
        "grid_size": 256,
        "nb_agents_hv": 256,
        "nb_agents_diag": 256,
        "nb_targets": 256,
        "nb_mobiles": 256,
        "fow_agents_hv": 2,
        "fow_agents_diag": 4,
    },
)
