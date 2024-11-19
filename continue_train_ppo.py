__credits__ = ["Yuri De Santis"]
# import warnings
# warnings.filterwarnings("ignore")

from __init__                               import *

import gymnasium                            as gym

# STABLE_BASELINES3
from stable_baselines3.common.vec_env       import DummyVecEnv, VecMonitor, VecNormalize, SubprocVecEnv
from stable_baselines3                      import PPO

# my functions and classes
from classes.callbacks.custom_callback      import CB4TB

from stable_baselines3.common.callbacks     import CallbackList, CheckpointCallback, EvalCallback

parent_str = "model"
dat_str = "..." 

step_str = "best_model/best_model.zip"
 
print(dat_str)

model_str = parent_str + "/" + dat_str + "/" + step_str

if __name__ == '__main__':

    visual = False
    
    yaml_str        = parent_str+ "/" +dat_str+ "/" +'config.yaml'
    config:dict     = load_config(yaml_str)

    env_id: str     = config['env_id']
    taskT: float    = config['taskT']
    stayT: float    = config['stayT']
    n_envs: int     = config['n_envs']
    n_ep_reset: int = config['n_ep_reset']
    n_update: int   = config['n_update']
    scaling: int    = config['scaling']
    f_robot: int    = config['f_robot']
    le: float       = config['le']
    lde: float      = config['lde']
    ldde: float     = config['ldde']
    kp: float       = config['kp']
    kv: float       = config['kv']
    fl_noILC: bool  = config['fl_noILC']
    chunks: int     = config['chunks']
    seed: int       = config['seed']
    pi:list[float]  = config['pi']
    vf:list[float]  = config['vf']
    config['extra'] = ''
    
    model_rl = PPO.load(model_str)
        
    samples = convert_parameters_for_training(taskT=taskT, freq_policy=f_robot/scaling)
    _PPO_batch          = n_envs*samples*n_ep_reset   # dimension of samples for one update -> n_envs*steps
    PPO_n_steps         = samples*n_ep_reset          # steps to update for single environment
    PPO_minibatch_size  = _PPO_batch//chunks
    PPO_total_timesteps = _PPO_batch*n_update
        
    print("")
    print_config(config=config)
    print("")
    print(f"freq policy     -> frequency of policy update                    : {f_robot/scaling}")
    print(f"samples         -> samples for 1 episode for 1 environment       : {samples}")
    print(f"samples4update  -> samples for 1 update for 1 environment        : {samples*n_ep_reset}")
    print(f"batch size      -> samples for 1 update of policy = n_envs*steps : {_PPO_batch}")
    print(f"chunks          -> number of mini-batch size                     : {chunks}")
    print(f"mini-batch size -> samples for mini-update of policy             : {PPO_minibatch_size} ")
    print(f"num update      -> number of policy update                       : {n_update}")
    print(f"total steps     -> total samples for training                    : {PPO_total_timesteps} ")
    print("")
    
    if _PPO_batch % chunks != 0:
        raise("mini-bach size not compatible for batch size")
    
    env = DummyVecEnv([
        lambda: gym.make(
            id         = env_id, 
            taskT      = taskT,
            f_robot    = f_robot,
            scaling    = scaling,
            n_ep_reset = n_ep_reset,
            le         = le,
            lde        = lde,
            ldde       = ldde,
            kp         = kp,
            kv         = kv,
            fl_noILC   = fl_noILC,
            seed       = seed,) 
        for i in range(n_envs)])
    
    env = VecMonitor(env)
    
    mycallback = CB4TB(reset_epN=n_ep_reset, modelFolder="model", logFolder="log", checkFreq=PPO_n_steps)
    with open(mycallback.modelsPath+'/config.yaml', 'w') as file:
        yaml.dump(config, file)
    
    callback = CallbackList([mycallback,])
    
    model_rl.set_env(env)
    
    model_rl.learn(total_timesteps  = PPO_total_timesteps,
                callback            = callback,
                tb_log_name         = "train",
                reset_num_timesteps = False, 
                progress_bar        = True,
                )