# ------------------------------------------------- #
# save entrypoint of custom gymnasium environments
# ------------------------------------------------- #
import gymnasium as gym


env_id  = 'Env_RILC'
path    = 'classes.environments.env_rlilc_mjc:Env_RILC'
gym.register(
        id=env_id,
        entry_point=path,
    )


env_id  = 'Env_classic_RL'
path    = 'classes.environments.env_rlilc_mjc:Env_classic_RL'
gym.register(
        id=env_id,
        entry_point=path,
    )
