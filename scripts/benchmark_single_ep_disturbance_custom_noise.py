import os
import time
import pickle
import numpy as np

import mujoco
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts import benchmark_mismatch as bm
from scripts.benchmark_mismatch import run_experiment


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'log')
IMG_DIR = os.path.join(os.path.dirname(__file__), '..', 'img')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)


def run_single_episode_disturbance_custom_noise(target_episodes=None, magnitude=0.5, mode_list=None, disturbance_type='impulse', 
                                                 noise_distribution='gaussian', noise_params=None, target_episode=None):
    """Run benchmarks where a disturbance is applied only during specified episodes.

    Uses monkeypatching of `mujoco.mj_resetData` and `mujoco.mj_step` so
    `benchmark_mismatch.run_experiment` can be executed without editing it.
    
    Args:
        target_episodes: list of episode indices to apply disturbance, or a single int. 
                        If None, defaults to [3]. Overrides `target_episode` if provided.
        target_episode: deprecated, use target_episodes instead. Backward compatibility.
        magnitude: scalar torque magnitude applied to each actuator (signed noise will be used).
        disturbance_type: 'impulse' (single fast-step at episode start) or 'continuous' (all steps in episode)
        noise_distribution: 'gaussian', 'uniform', 'exponential', 'laplace', or custom function
        noise_params: dict of additional parameters for the distribution (e.g., {'loc': 0, 'scale': 1} for gaussian)
        mode_list: list of controller modes to test
    """
    # Handle backward compatibility: if target_episode is provided, use it
    if target_episode is not None and target_episodes is None:
        target_episodes = target_episode
    
    if target_episodes is None:
        target_episodes = [3]
    
    # Ensure target_episodes is a list
    if isinstance(target_episodes, int):
        target_episodes = [target_episodes]
    
    if mode_list is None:
        mode_list = ["ILC", "NOILC", "RL", "RILC"]

    if noise_params is None:
        noise_params = {}

    # Save originals
    orig_reset = mujoco.mj_resetData
    orig_step = mujoco.mj_step

    state = {'ep': -1, 'applied': False}

    # History of applied disturbance: dict keyed by episode number
    episode_disturbances = {}  # {ep_num: [list of noise norms per step]}
    episode_per_actuator = {}  # {ep_num: [list of per-step actuator vectors]}
    
    # Current episode's history (transient)
    disturbance_history = []  
    per_actuator_history = []

    rng = np.random.default_rng(12345)

    def generate_noise(size):
        """Generate noise according to specified distribution"""
        if noise_distribution == 'gaussian':
            loc = noise_params.get('loc', 0.0)
            scale = noise_params.get('scale', magnitude)
            return rng.normal(loc=loc, scale=scale, size=size)
        
        elif noise_distribution == 'uniform':
            low = noise_params.get('low', -magnitude)
            high = noise_params.get('high', magnitude)
            return rng.uniform(low=low, high=high, size=size)
        
        elif noise_distribution == 'exponential':
            scale = noise_params.get('scale', magnitude)
            noise = rng.exponential(scale=scale, size=size)
            # Apply random sign
            signs = rng.choice([-1, 1], size=size)
            return noise * signs
        
        elif noise_distribution == 'laplace':
            loc = noise_params.get('loc', 0.0)
            scale = noise_params.get('scale', magnitude)
            return rng.laplace(loc=loc, scale=scale, size=size)
        
        elif noise_distribution == 'truncated_gaussian':
            loc = noise_params.get('loc', 0.0)
            scale = noise_params.get('scale', magnitude)
            limit = noise_params.get('limit', 3 * scale)
            noise = rng.normal(loc=loc, scale=scale, size=size)
            return np.clip(noise, -limit, limit)
        
        elif callable(noise_distribution):
            # Allow custom function: noise_distribution(rng, size, magnitude, **noise_params)
            return noise_distribution(rng, size, magnitude, **noise_params)
        
        else:
            raise ValueError(f"Unknown noise distribution: {noise_distribution}")

    def wrapped_reset(model, data):
        # If we just finished a target episode, save its history before clearing
        if state['ep'] in target_episodes and disturbance_history:
            episode_disturbances[state['ep']] = disturbance_history.copy()
            episode_per_actuator[state['ep']] = [list(a) for a in per_actuator_history]
        
        # increment episode counter and clear applied flag
        state['ep'] += 1
        state['applied'] = False
        # clear per-episode disturbance history when starting a new episode
        disturbance_history.clear()
        per_actuator_history.clear()
        return orig_reset(model, data)

    def wrapped_step(model, data, nstep=1):
        # If we're in any target episode, apply disturbance according to type
        try:
            if state['ep'] in target_episodes:
                ctrl = np.asarray(data.ctrl).flatten()
                if disturbance_type == 'impulse':
                    if not state['applied']:
                        noise = generate_noise(ctrl.size)
                        data.ctrl[:] = (ctrl + noise).tolist()
                        # record noise
                        disturbance_history.append(float(np.linalg.norm(noise)))
                        per_actuator_history.append(noise.tolist())
                        state['applied'] = True
                    else:
                        # after impulse has been applied, record zero for remaining steps
                        disturbance_history.append(0.0)
                        per_actuator_history.append([0.0] * len(ctrl))
                else:  # continuous
                    noise = generate_noise(ctrl.size)
                    data.ctrl[:] = (ctrl + noise).tolist()
                    disturbance_history.append(float(np.linalg.norm(noise)))
                    per_actuator_history.append(noise.tolist())
        except Exception:
            pass
        return orig_step(model, data, nstep=nstep)

    results = {'Nominal': {}, 'SingleEpDisturb': {}}

    try:
        # 1) Nominal runs - WITHOUT wrappers active
        for mode in mode_list:
            print(f"Running nominal: {mode}")
            try:
                rmse_nom = run_experiment(mode=mode, mismatch=False, n_ep_reset=30)
                results['Nominal'][mode] = rmse_nom
            except Exception as e:
                print(f"Nominal run failed for {mode}: {e}")
                results['Nominal'][mode] = []

        # 2) Now activate wrappers and run with disturbance
        mujoco.mj_resetData = wrapped_reset
        mujoco.mj_step = wrapped_step
        
        for mode in mode_list:
            # Reset episode counter for each controller so disturbance is applied at the same relative episode
            state['ep'] = -1
            state['applied'] = False
            print(f"Running with single-episode disturbance: {mode} (ep={target_episodes}, noise_dist={noise_distribution})")
            try:
                rmse_dist = run_experiment(mode=mode, mismatch=False, n_ep_reset=30)
                results['SingleEpDisturb'][mode] = rmse_dist
            except Exception as e:
                print(f"Disturbed run failed for {mode}: {e}")
                results['SingleEpDisturb'][mode] = []

    finally:
        # Restore originals
        mujoco.mj_resetData = orig_reset
        mujoco.mj_step = orig_step

    # Plot RMSE results
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
    colors = {"ILC": "orange", "NOILC": "dodgerblue", "RL": "green", "RILC": "tomato"}
    all_vals = []

    for idx, key in enumerate(['Nominal', 'SingleEpDisturb']):
        ax = axs[idx]
        for mode in mode_list:
            vals = results[key].get(mode, [])
            if vals:
                ax.plot(vals, label=mode, color=colors.get(mode, 'black'), linewidth=2)
                ax.scatter(range(len(vals)), vals, marker='o', facecolors='none', edgecolors=colors.get(mode, 'black'))
                all_vals.extend(vals)
        ep_str = f"Ep {target_episodes[0]}" if len(target_episodes) == 1 else f"Ep {target_episodes}"
        ax.set_title('Nominal' if key=='Nominal' else f'Disturb {ep_str}')
        ax.set_xlabel('Episode')
        if idx==0:
            ax.set_ylabel('RMSE [rad]')
        ax.grid(True)

    if all_vals:
        positives = [v for v in all_vals if v is not None and v > 0]
        if len(positives)==0:
            y_min, y_max = 1e-6, 1e-3
        else:
            y_min = max(min(positives)*0.8, 1e-6)
            y_max = max(positives)*1.2
        for ax in axs:
            ax.set_yscale('log')
            ax.set_ylim(y_min, y_max)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(mode_list))
    plt.tight_layout(rect=[0,0,1,0.9])
    ep_tag = '_'.join(map(str, target_episodes)) if len(target_episodes) > 1 else str(target_episodes[0])
    img_path = os.path.join(IMG_DIR, f'single_ep_disturb_custom_ep{ep_tag}_mag{magnitude:.3f}_{noise_distribution}.pdf')
    fig.savefig(img_path, dpi=200)
    print(f"Saved plot to {img_path}")

    if not episode_disturbances:
        print(f"WARNING: No disturbance history recorded for episodes {target_episodes}")

    return results


if __name__ == '__main__':
    # Example 1: Gaussian noise (default)
    # run_single_episode_disturbance_custom_noise(target_episodes=[0, 5, 10], magnitude=0.02, 
    #                                              mode_list=["ILC", "NOILC", "RL", "RILC"], 
    #                                              disturbance_type='continuous',
    #                                              noise_distribution='gaussian')
    
    # Example 2: Uniform noise
    # run_single_episode_disturbance_custom_noise(target_episodes=[0, 5, 10], magnitude=0.02,
    #                                              mode_list=["ILC", "NOILC", "RL", "RILC"],
    #                                              disturbance_type='continuous',
    #                                              noise_distribution='uniform',
    #                                              noise_params={'low': -0.02, 'high': 0.02})
    
    # Example 3: Exponential noise with sign
    t_ep = [n for n in range(0, 30)]
    run_single_episode_disturbance_custom_noise(target_episodes=t_ep, magnitude=0.02,
                                                 mode_list=["ILC", "NOILC", "RL", "RILC"],
                                                 disturbance_type='continuous',
                                                 noise_distribution='exponential',
                                                 noise_params={'scale': 0.02})
    
    # Example 4: Laplace (heavy-tailed) noise
    # run_single_episode_disturbance_custom_noise(target_episodes=[0, 5, 10], magnitude=0.02,
    #                                              mode_list=["ILC", "NOILC", "RL", "RILC"],
    #                                              disturbance_type='continuous',
    #                                              noise_distribution='laplace',
    #                                              noise_params={'loc': 0.0, 'scale': 0.01})
