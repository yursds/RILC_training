# ------------------------------------------------- #
# save entrypoint of custom gymnasium environments
# ------------------------------------------------- #
import gymnasium as gym


env_id  = 'Env_RILC_old'
path    = 'classes.environments.env_rlilc_mjc_old:Env_RILC'
gym.register(
    id=env_id,
    entry_point=path,
)


env_id  = 'Env_RL_old'
path    = 'classes.environments.env_rlilc_mjc_old:Env_RL'
gym.register(
    id=env_id,
    entry_point=path,
)


env_id  = 'Env_RILC'
path    = 'classes.environments.env_rlilc_mjc:Env_RILC'
gym.register(
    id=env_id,
    entry_point=path,
)


env_id  = 'Env_RL'
path    = 'classes.environments.env_rlilc_mjc:Env_RL'
gym.register(
    id=env_id,
    entry_point=path,
)


env_id  = 'Env_RILC_LISS'
path    = 'classes.environments.env_rlilc_mjc:Env_RILC_LISS'
gym.register(
    id=env_id,
    entry_point=path,
)


env_id  = 'Env_RL_LISS'
path    = 'classes.environments.env_rlilc_mjc:Env_RL_LISS'
gym.register(
    id=env_id,
    entry_point=path,
)


env_id  = 'Env_RILC_RANGE'
path    = 'classes.environments.env_rlilc_mjc:Env_RILC_RANGE'
gym.register(
    id=env_id,
    entry_point=path,
)


env_id  = 'Env_RL_RANGE'
path    = 'classes.environments.env_rlilc_mjc:Env_RL_RANGE'
gym.register(
    id=env_id,
    entry_point=path,
)

