__credits__ = ["Yuri De Santis"]
# import warnings
# warnings.filterwarnings("ignore")
import os
import sys

# Add local directory to path
sys.path.append(os.getcwd())

from __init__                               import *

import gymnasium                            as gym
import torch

# STABLE_BASELINES3
from stable_baselines3.common.vec_env       import DummyVecEnv, VecMonitor, VecNormalize, SubprocVecEnv

from stable_baselines3                      import PPO as rl_algo
# from stable_baselines3                      import A2C as rl_algo

# my functions and classes
from classes.callbacks.custom_callback      import CB4TB

from stable_baselines3.common.callbacks     import CallbackList, CheckpointCallback, EvalCallback

if __name__ == '__main__':
    
    yaml_str = 'config/softleg_env_mjc.yaml'
    section  = 'rl'

    config:dict   = load_config_section(yaml_str, section)

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
    chunks: int     = config['chunks']
    seed: int       = config['seed']
    pi:list[float]  = config['pi']
    vf:list[float]  = config['vf']
    
    config['extra'] = ''
    
    samples = convert_parameters_for_training(taskT=taskT, freq_policy=f_robot/scaling)
    _rl_algo_batch          = n_envs*samples*n_ep_reset   # dimension of samples for one update -> n_envs*steps
    rl_algo_n_steps         = samples*n_ep_reset          # steps to update for single environment
    rl_algo_minibatch_size  = _rl_algo_batch//chunks
    rl_algo_total_timesteps = _rl_algo_batch*n_update
        
    print("")
    print_config(config=config)
    print("")
    print(f"freq policy     -> frequency of policy update                    : {f_robot/scaling}")
    print(f"samples         -> samples for 1 episode for 1 environment       : {samples}")
    print(f"samples4update  -> samples for 1 update for 1 environment        : {samples*n_ep_reset}")
    print(f"batch size      -> samples for 1 update of policy = n_envs*steps : {_rl_algo_batch}")
    print(f"chunks          -> number of mini-batch size                     : {chunks}")
    print(f"mini-batch size -> samples for mini-update of policy             : {rl_algo_minibatch_size} ")
    print(f"num update      -> number of policy update                       : {n_update}")
    print(f"total steps     -> total samples for training                    : {rl_algo_total_timesteps} ")
    
    # if _rl_algo_batch % chunks != 0:
    #     raise("mini-bach size not compatible for batch size")
    
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
            seed       = seed,) 
        for i in range(n_envs)])
    
    env = VecMonitor(env)
    
    mycallback = CB4TB(reset_epN=n_ep_reset, modelFolder="model", logFolder="log", checkFreq=rl_algo_n_steps*10)
    with open(mycallback.modelsPath+'/config.yaml', 'w') as file:
        yaml.dump(config, file)
    
    # checkpoint_callback = CheckpointCallback(save_freq=1000, save_path="./logs/")
    # eval_callback = EvalCallback(
    #     eval_env             = env, 
    #     best_model_save_path = "./logs/best_model",
    #     log_path             = "./logs/results",
    #     eval_freq            = rl_algo_minibatch_size,
    #     n_eval_episodes      = 1)
    # Create the callback list
    
    callback = CallbackList([mycallback,])
    
    # default
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=pi, vf=vf))
    
    model = rl_algo(
        policy          = "MlpPolicy",
        n_steps         = rl_algo_n_steps, # for a single env
        batch_size      = rl_algo_minibatch_size,
        env             = env,
        verbose         = 0,
        tensorboard_log = mycallback.logPath,
        use_sde         = False,
        # ent_coef        = 0.0,
        # gamma           = 0.99,
        seed            = seed,
        policy_kwargs   = policy_kwargs,
        device          = 'cuda'
        )
    
    print(model.policy_kwargs)
    print(model.policy.net_arch)
    print(model.policy)
    
    #model.policy =  pre_model.policy
    model.learn(total_timesteps     = rl_algo_total_timesteps,
                callback            = callback,
                tb_log_name         = "train",
                reset_num_timesteps = False, 
                progress_bar        = True,
                )
    
    # all_qf_visit = []
    # for env_single in env.unwrapped.envs:
    #     qf_visit = env_single.get_wrapper_attr("qf_list")
    #     print("qf visit: ", qf_visit)
    #     all_qf_visit.append(qf_visit)
        
    # import pickle

    # with open("qf_visit_training_rilc.pkl", "wb") as f:
    #     pickle.dump(all_qf_visit, f)