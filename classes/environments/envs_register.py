# ------------------------------------------------- #
# save entrypoint of custom gymnasium environments
# ------------------------------------------------- #
import gymnasium as gym


def _register(env_id, path):
    """Register a gymnasium environment only if not already registered."""
    if env_id not in gym.envs.registry:
        gym.register(id=env_id, entry_point=path)


_register('Env_RILC',       'classes.environments.env_rlilc_mjc:Env_RILC')
_register('Env_RL',         'classes.environments.env_rlilc_mjc:Env_RL')
_register('Env_RILC_LISS',  'classes.environments.env_rlilc_mjc:Env_RILC_LISS')
_register('Env_RL_LISS',    'classes.environments.env_rlilc_mjc:Env_RL_LISS')
_register('Env_RILC_RANGE', 'classes.environments.env_rlilc_mjc:Env_RILC_RANGE')
_register('Env_RL_RANGE',   'classes.environments.env_rlilc_mjc:Env_RL_RANGE')
