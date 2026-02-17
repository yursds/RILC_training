import os
import sys

# Add local directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from __init__ import *
import torch
import mujoco
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import functools
import numpy as np
from stable_baselines3 import PPO

from benchmark_utils import *

# --- Plotting Style ---
setup_plotting()


from classes.controllers.ilc import ILC_base
from classes.controllers.noilc import NOILC

from classes.robots.manipulator_RR import Sim_RR
from classes.environments.env_rlilc_mjc import Env_RILC as ENV


# Simulation Param
f_robot = 100
scaling = 2
taskT = 1.0
n_ep_initial = 20
n_ep_switch = 20
n_ep_total = n_ep_initial + n_ep_switch
kp = 0.0 # KPI=0 to isolate ILC performance
kv = 0.25 

# Gains from config.yaml
le_cfg = 0.0002
lde_cfg = 0.0004
ldde_cfg = 0.0

# Trajectories
QF_A = torch.tensor([[2.4], [-1.4]])
QF_B = torch.tensor([[1.], [1.5]]) # New Target

# Episodes to trace for plotting
TRACE_EPISODES = [0, n_ep_initial-1, n_ep_initial, n_ep_total-1]

# --- Helper ---
def plot_action_contributions(traces, episode_idx, alg_name, suffix=""):
    """
    Plots action contributions for a specific episode.
    traces: dict with keys 'uMB', 'uFB', 'uILC', 'uRL' containing lists of tensors
    """
    if episode_idx not in traces:
        return
        
    data = traces[episode_idx]
    
    # Convert lists to stacked tensors and numpy
    uMB = torch.stack(data['uMB']).numpy()
    uFB = torch.stack(data['uFB']).numpy()
    uILC = torch.stack(data['uILC']).numpy()
    uRL = torch.stack(data['uRL']).numpy()
    
    # Sums
    uTot = uMB + uFB + uILC + uRL
    uRL_ILC = uRL + uILC
    
    plt.figure(figsize=(18, 3))
    
    # 1. uTOT
    plt.subplot(1, 6, 1)
    plt.plot(uTot[:,0], label="J1")
    plt.plot(uTot[:,1], label="J2")
    plt.title(f"uTOT")
    plt.xlabel("steps")
    plt.grid()
    plt.legend()
    
    # 2. uRL + uILC
    plt.subplot(1, 6, 2)
    plt.plot(uRL_ILC[:,0])
    plt.plot(uRL_ILC[:,1])
    plt.title(f"uRL+uILC")
    plt.xlabel("steps")
    plt.grid()
    
    # 3. uMB
    plt.subplot(1, 6, 3)
    plt.plot(uMB[:,0])
    plt.plot(uMB[:,1])
    plt.title(f"uMB")
    plt.xlabel("steps")
    plt.grid()
    
    # 4. uILC
    plt.subplot(1, 6, 4)
    plt.plot(uILC[:,0])
    plt.plot(uILC[:,1])
    plt.title(f"uILC")
    plt.xlabel("steps")
    plt.grid()
    
    # 5. uFB
    plt.subplot(1, 6, 5)
    plt.plot(uFB[:,0])
    plt.plot(uFB[:,1])
    plt.title(f"uFB")
    plt.xlabel("steps")
    plt.grid()
    
    # 6. uRL
    plt.subplot(1, 6, 6)
    plt.plot(uRL[:,0])
    plt.plot(uRL[:,1])
    plt.title(f"uRL")
    plt.xlabel("steps")
    plt.grid()
    
    plt.suptitle(f"{alg_name} - Episode {episode_idx} Actions")
    plt.tight_layout()
    
    img_dir = os.path.join(os.path.dirname(__file__), '..', 'img')
    os.makedirs(img_dir, exist_ok=True)
    save_path = os.path.join(img_dir, f"actions_{alg_name}_ep{episode_idx}{suffix}.pdf")
    plt.savefig(save_path)
    print(f"Saved action plot to {save_path}")
    plt.close()

def plot_evolution(traces, alg_name, suffix=""):
    """
    Plots evolution of trajectories and torques over episodes on a single figure.
    """
    episodes = sorted(traces.keys())
    if not episodes:
        return
        
    plt.figure(figsize=(16, 10))
    
    # Use a colormap for episodes to show progression
    colors = plt.cm.viridis(np.linspace(0, 1, len(episodes)))
    
    for j in range(2): # 2 Joints
        # Trajectory Subplot
        plt.subplot(2, 2, j+1)
        plt.title(f"Joint {j+1}: Trajectory Evolution")
        
        for i, ep in enumerate(episodes):
            data = traces[ep]
            if not data['q']: continue
            
            # Extract
            q_act = torch.stack(data['q']).numpy()[:, j]
            q_ref = torch.stack(data['q_ref']).numpy()[:, j]
            t = np.arange(len(q_act))
            
            # Plot Ref (Only once per phase or style it?)
            # Plotting ref for every ep ensures we see the switch
            plt.plot(t, q_ref, color='k', linestyle=':', alpha=0.3, zorder=0) 
            
            # Plot Act
            plt.plot(t, q_act, color=colors[i], label=f"Ep {ep}", linewidth=1.5)
            
        plt.ylabel("Angle [rad]")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Torque Subplot
        plt.subplot(2, 2, j+3) # 3 and 4
        plt.title(f"Joint {j+1}: Torque Evolution (uTot)")
        
        for i, ep in enumerate(episodes):
            data = traces[ep]
            if not data['uMB']: continue
            
            # Calculate Total Torque
            uMB = torch.stack(data['uMB']).numpy()[:, j]
            uFB = torch.stack(data['uFB']).numpy()[:, j]
            uILC = torch.stack(data['uILC']).numpy()[:, j]
            uRL = torch.stack(data['uRL']).numpy()[:, j]
            uTot = uMB + uFB + uILC + uRL
            
            t = np.arange(len(uTot))
            
            plt.plot(t, uTot, color=colors[i], label=f"Ep {ep}", linewidth=1.5)
            
        plt.ylabel("Torque [Nm]")
        plt.xlabel("Step")
        plt.grid(True, alpha=0.3)
        # plt.legend() # Legend already in Traj plot matching colors
        
    plt.suptitle(f"{alg_name} - Evolution{suffix}")
    plt.tight_layout()
    
    img_dir = os.path.join(os.path.dirname(__file__), '..', 'img')
    os.makedirs(img_dir, exist_ok=True)
    save_path = os.path.join(img_dir, f"evolution_{alg_name}{suffix}.pdf")
    plt.savefig(save_path)
    print(f"Saved evolution plot to {save_path}")
    plt.close()

def plot_interaction(traces, episode_idx, alg_name, suffix="", gt_trace=None, rilc_fresh_trace=None):
    """
    Overlays uRL and uILC, and shows Trajectories.
    """
    if episode_idx not in traces:
        return
        
    data = traces[episode_idx]
    
    # Extract Data (convert to numpy)
    uILC = torch.stack(data['uILC']).numpy()
    uRL = torch.stack(data['uRL']).numpy()
    uFB = torch.stack(data['uFB']).numpy()
    uMB = torch.stack(data['uMB']).numpy()
    
    q_act = torch.stack(data['q']).numpy()
    q_ref = torch.stack(data['q_ref']).numpy()
    
    uTot = uMB + uFB + uILC + uRL
    uSum = uILC + uRL
    
    t = np.arange(len(uILC))
    
    # Process GT
    uTot_GT = None
    if gt_trace is not None:
        try:
           gt_uILC = torch.stack(gt_trace['uILC']).numpy()
           gt_uRL = torch.stack(gt_trace['uRL']).numpy()
           gt_uFB = torch.stack(gt_trace['uFB']).numpy()
           gt_uMB = torch.stack(gt_trace['uMB']).numpy()
           uTot_GT = gt_uILC + gt_uRL + gt_uFB + gt_uMB
        except Exception as e: print(f"GT process error: {e}")

    # Process Fresh
    uTot_Fresh = None
    if rilc_fresh_trace is not None:
        try:
            fr_uILC = torch.stack(rilc_fresh_trace['uILC']).numpy()
            fr_uRL = torch.stack(rilc_fresh_trace['uRL']).numpy()
            fr_uFB = torch.stack(rilc_fresh_trace['uFB']).numpy()
            fr_uMB = torch.stack(rilc_fresh_trace['uMB']).numpy()
            uTot_Fresh = fr_uILC + fr_uRL + fr_uFB + fr_uMB
        except Exception as e: print(f"Fresh process error: {e}")

    
    plt.figure(figsize=(14, 8))

    
    for j in range(2): # Joints
        # 1. Trajectory
        plt.subplot(2, 2, j*2 + 1)
        plt.plot(t, q_ref[:, j], label="q_ref", color="black", linestyle="--", linewidth=2)
        plt.plot(t, q_act[:, j], label=f"q_{alg_name}", color="blue", linewidth=1.5)
        plt.title(f"Joint {j+1} - Trajectory")
        plt.ylabel("Angle [rad]")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 2. Torques
        plt.subplot(2, 2, j*2 + 2)
        
        # Components
        if alg_name == "RILC":
            plt.plot(t, uILC[:, j], label="uILC", color="orange", linestyle="--", alpha=0.7)
            plt.plot(t, uRL[:, j], label="uRL", color="tomato", linestyle="-", alpha=0.7)
        elif alg_name == "ILC" or alg_name == "NOILC": # For ILC/NOILC, uRL is zero, so only plot uILC
            plt.plot(t, uILC[:, j], label="uILC", color="orange", linestyle="--", alpha=0.7)
            
        # Total Comparisons
        if uTot_GT is not None:
             plt.plot(t, uTot_GT[:, j], label="GT (Converged)", color="black", linestyle=":", linewidth=2, alpha=0.6)
             
        if uTot_Fresh is not None:
             plt.plot(t, uTot_Fresh[:, j], label="Fresh Start", color="cyan", linestyle="-.", linewidth=2, alpha=0.6)

        # Current Total
        plt.plot(t, uTot[:, j], label="uTot (Curr)", color="green", linestyle="-.", alpha=0.5)
        
        plt.title(f"Joint {j+1} - Torques")
        plt.ylabel("Torque [Nm]")
        plt.xlabel("Step")
        plt.grid(True, alpha=0.3)
        plt.legend()

    plt.suptitle(f"{alg_name} - Episode {episode_idx} Interaction{suffix}")
    plt.tight_layout()
    
    img_dir = os.path.join(os.path.dirname(__file__), '..', 'img')
    os.makedirs(img_dir, exist_ok=True)
    save_path = os.path.join(img_dir, f"interaction_{alg_name}_ep{episode_idx}{suffix}.pdf")
    plt.savefig(save_path)
    print(f"Saved interaction plot to {save_path}")
    plt.close()

def plot_all_trajectories(all_traces, fresh_trace=None, suffix="", dt=0.01):
    """
    Plots comparative trajectories for ILC, NOILC, RILC (and RL).
    Uses global TRACE_EPISODES for episode selection.
    """
    # Use global TRACE_EPISODES and check availability
    key_episodes = TRACE_EPISODES[:]
    if all_traces:
        # Filter to only available episodes
        available_eps = set()
        for alg_traces in all_traces.values():
            available_eps.update(alg_traces.keys())
        key_episodes = sorted([ep for ep in key_episodes if ep in available_eps])
    
    # Reduced width as requested
    plt.figure(figsize=(18, 10))
    
    alg_colors = {
        "ILC": "orange",
        "NOILC": "dodgerblue",
        "RILC": "tomato",
        "RL": "purple"
    }
    
    # Line styles per algorithm to visually distinguish approaches
    alg_linestyles = {
        "ILC": "-",    # solid
        "NOILC": "--", # dashed
        "RILC": "-.",  # dash-dot
        "RL": ":"      # dotted
    }
    # 1. Determine Y-Limits per Joint (Shared Scaling)
    y_lims = [None, None]
    for j in range(2):
        vals = []
        # Check all algorithms and episodes
        for alg, traces in all_traces.items():
            for ep in key_episodes:
                if ep in traces:
                    vals.append(torch.stack(traces[ep]['q']).numpy()[:, j])
                    vals.append(torch.stack(traces[ep]['q_ref']).numpy()[:, j])
        
        # Check Fresh Trace
        if fresh_trace:
             for ep in key_episodes:
                 if ep >= n_ep_initial:
                     fr_ep = ep - n_ep_initial
                     if fr_ep in fresh_trace:
                         vals.append(torch.stack(fresh_trace[fr_ep]['q']).numpy()[:, j])

        if vals:
             all_ws = np.concatenate(vals)
             y_min, y_max = np.min(all_ws), np.max(all_ws)
             margin = (y_max - y_min) * 0.1
             y_lims[j] = (y_min - margin, y_max + margin)

    
    for j in range(2): # 2 Joints
        for i, ep in enumerate(key_episodes):
            plt.subplot(2, 4, j*4 + i + 1)
            
            phase = "Ref A" if ep < n_ep_initial else "Ref B (Switch)"
            title = f"Joint {j+1}: Ep {ep} ({phase})"
            plt.title(title)
            
            ref_plotted = False
            
            # Plot Algorithms
            for alg in sorted(all_traces.keys()):
                traces = all_traces[alg]
                if ep not in traces: continue
                
                data = traces[ep]
                q_act = torch.stack(data['q']).numpy()[:, j]
                t = np.arange(len(q_act)) * dt
                
                # Plot Ref once
                if not ref_plotted:
                    q_ref = torch.stack(data['q_ref']).numpy()[:, j]
                    plt.plot(t, q_ref, color='k', linestyle=':', label="Ref", linewidth=2, zorder=10)
                    ref_plotted = True
                    
                color = alg_colors.get(alg, 'grey')
                ls = alg_linestyles.get(alg, '-')
                plt.plot(t, q_act, color=color, linestyle=ls, label=f"{alg}", linewidth=1.5, alpha=0.9)
                
            # Plot RILC Fresh
            if fresh_trace is not None and ep >= n_ep_initial:
                fresh_ep = ep - n_ep_initial 
                if fresh_ep in fresh_trace:
                    data = fresh_trace[fresh_ep]
                    q_fresh = torch.stack(data['q']).numpy()[:, j]
                    t = np.arange(len(q_fresh)) * dt
                    plt.plot(t, q_fresh, color='magenta', linestyle='-.', label="RILC Fresh", linewidth=1.5, alpha=0.8)

            if j == 1: plt.xlabel("Time [s]")
            if i == 0: plt.ylabel("Angle [rad]")
            
            # if y_lims[j]:
            #     plt.ylim(y_lims[j])
            
            # Smart Legend Placement - only on first plot of row or outside?
            # User said "in appropriate place". 'best' is usually safest per subplot if not crowded.
            plt.legend(loc='best', fontsize='x-small', framealpha=0.6)
            plt.grid(True, alpha=0.3)

    plt.suptitle(f"Trajectory Comparison{suffix}", fontsize=16)
    plt.tight_layout()
    
    img_dir = os.path.join(os.path.dirname(__file__), '..', 'img')
    os.makedirs(img_dir, exist_ok=True)
    save_path = os.path.join(img_dir, f"trajectories_comparison{suffix}.pdf")
    plt.savefig(save_path)
    print(f"Saved comparative trajectories plot to {save_path}")
    plt.close()


# --- SIMULATION RUNNER ---
def run_experiment(mode="ILC", mismatch=False, scenario="switch", gt_trace=None, rilc_fresh_trace=None):
    # scenario: "switch" or "ref_B"
    print(f"\n--- Running Experiment: {mode} (Mismatch={mismatch}, Scenario={scenario}) ---")
    
    # Init Env/Robot
    f_policy = int(f_robot / scaling)
    samples = int(taskT*f_policy) + 1
    dt_pol = 1/f_policy
    dt_rob = 1/f_robot
    njoint = 2
    
    # Define Trajectories
    traj_A = functools.partial(minjerk, qi = torch.tensor([[0.0], [0.0]]), qf = QF_A, duration = taskT)
    traj_B = functools.partial(minjerk, qi = torch.tensor([[0.0], [0.0]]), qf = QF_B, duration = taskT)
    
    if scenario == "switch":
        total_episodes = n_ep_total
        initial_traj = traj_A
    elif scenario == "ref_B":
        total_episodes = n_ep_switch
        initial_traj = traj_B
        
    # Env for normalization
    env = ENV(taskT=taskT, f_robot=f_robot, scaling=scaling, 
              le=le_cfg, lde=lde_cfg, ldde=ldde_cfg, kp=kp, kv=kv, n_ep_reset=total_episodes)
    
    model_mj = mujoco.MjModel.from_xml_path(MJC_PATH)
    
    if mismatch:
        # 20% Mismatch applied to simulation model
        print("Applying 20% Mass/Friction Mismatch...")
        model_mj.body_mass[:] = model_mj.body_mass * 1.2
        # model_mj.dof_frictionloss[:] = model_mj.dof_frictionloss * 1.2
    
    data_mj = mujoco.MjData(model_mj)
    frame_skip = int((1/f_robot)/model_mj.opt.timestep)
    
    robot = Sim_RR(urdf_path=URDF_PATH, ee_name='LH_ANKLE')
    
    # Init robot state (Start is always 0,0 for both traj)
    tmp_q = initial_traj(t=0.0)[0].clone()
    tmp_dq = initial_traj(t=0.0)[1].clone()
    robot.setState(q0=tmp_q, dq0=tmp_dq, q=tmp_q, dq=tmp_dq)
    qi = robot.q0.clone()
    qpos_init = robot.q0.flatten().numpy().copy()
    qvel_init = robot.dq0.flatten().numpy().copy()
    
    # Controller Setup
    controller = None
    model_rl = None
    
    if mode == "RL":
        print(f"Loading RL model from {MODEL_RL_STR}")
        model_rl = PPO.load(MODEL_RL_STR)
    
    if mode == "RILC":
        print(f"Loading RILC model from {MODEL_RILC_STR}")
        model_rl = PPO.load(MODEL_RILC_STR)

        
    if mode == "ILC" or mode == "RILC":
        le_tens = torch.tensor(le_cfg * f_policy)
        lde_tens = torch.tensor(lde_cfg * f_policy)
        ldde_tens = torch.tensor(ldde_cfg * f_policy)
        controller = ILC_base(dimU=njoint, samples=samples, Le=le_tens, Lde=lde_tens, Ldde=ldde_tens)
        controller.newEp()
        
    elif mode == "NOILC":
        # Precompute Traj Ref for Linearization
        
        traj_for_lin = initial_traj
        
        u_traj_ref = torch.zeros(njoint, samples)
        q_traj_ref = torch.zeros(njoint, samples)
        dq_traj_ref = torch.zeros(njoint, samples)
        for i in range(samples):
            r_val, dr_val, ddr_val = traj_for_lin(t=i*dt_pol)
            q_traj_ref[:, i] = r_val.flatten()
            dq_traj_ref[:, i] = dr_val.flatten()
            tau_ref = robot.getInvDyn(r_val, dr_val, ddr_val, damp_fl=True)
            u_traj_ref[:, i] = tau_ref.flatten()
            
        G_mat = construct_lifted_model_linearized_nonlinear(robot, q_traj_ref, dq_traj_ref, u_traj_ref, dt=dt_pol, samples=samples, dimU=njoint, use_analytical=True)
        
        Q_mat = 1.0 * torch.eye(njoint * samples)
        R_mat = 1.0 * torch.eye(njoint * samples) 
        controller = NOILC(dimU=njoint, samples=samples, G=G_mat, Q=Q_mat, R=R_mat, threshold=1e-4)
        
        # Initial Guess (Nominal Model for Initial Traj)
        controller.uEp = u_traj_ref.clone()
        controller.best_u = u_traj_ref.clone()
        controller.newEp()

    # RILC Specific History
    uRL_old_ep_ts = torch.zeros(njoint,1,samples)
    uILC_old_ep_ts = torch.zeros(njoint,1,samples)
    uFB_old_ep_ts = torch.zeros(njoint,1,samples)
    e_old_ep_ts = torch.zeros(njoint,1,samples)
    de_old_ep_ts = torch.zeros(njoint,1,samples)
    
    # Main Loop
    rmse_per_ep = []
    
    # Episodes to trace (using global TRACE_EPISODES)
    trace_episodes = sorted([ep for ep in TRACE_EPISODES if ep < total_episodes])
    
    traces = {ep_idx: {'uMB': [], 'uFB': [], 'uILC': [], 'uRL': [], 'q': [], 'q_ref': []} for ep_idx in trace_episodes}

    
    current_traj = initial_traj
    if scenario == "switch":
        print(f"Starting with Trajectory A ({n_ep_initial} eps)...")
    else:
        print(f"Starting with Trajectory B ({n_ep_switch} eps)...")
    
    for ep in range(total_episodes):
        # Check for Trajectory Switch
        if scenario == "switch" and ep == n_ep_initial:
            print(f"--- Switching to Trajectory B (Ep {ep}) ---")
            current_traj = traj_B
            
        # Step ILC/NOILC logic
        if ep == 0:
            pass # Already called newEp
        else:
            if controller:
                controller.stepILC()
            
        # Reset Env
        mujoco.mj_resetData(model_mj, data_mj)
        
        data_mj.qpos = qpos_init
        data_mj.qvel = qvel_init
        mujoco.mj_inverse(model_mj, data_mj)
        data_mj.ctrl[:] = robot.getGravity(q=qi).flatten()
        mujoco.mj_forward(model_mj, data_mj)
        
        dq_old = torch.as_tensor(data_mj.qvel).view(2,1).clone()
        
        # Init Variables for Episode
        uFB = torch.zeros(njoint,1)
        uRL = torch.zeros(njoint,1)
        uRL_old = torch.zeros(njoint,1)
        uILC = torch.zeros(njoint,1)
        uILC_old = torch.zeros(njoint,1)
        
        e_ep = []
        t = 0.0
        
        if controller and hasattr(controller, 'idx'): controller.idx = 0
            
        for i in range(samples):
            # Target (Switched)
            r_, dr_, ddr_ = current_traj(t=t)
            
            # State
            q_curr = torch.zeros(2,1)
            q_curr[0] = torch.from_numpy(data_mj.sensor("q_hip").data)
            q_curr[1] = torch.from_numpy(data_mj.sensor("q_knee").data)
            dq_curr = torch.zeros(2,1)
            dq_curr[0] = torch.from_numpy(data_mj.sensor("dq_hip").data)
            dq_curr[1] = torch.from_numpy(data_mj.sensor("dq_knee").data)
            
            # Noise
            q_curr += 1e-6 * torch.randn(2,1)
            dq_curr += 2.5e-4 * torch.randn(2,1)
            
            dq_old = dq_curr.clone()
            
            # Error
            e_ = angle_normalize(r_ - q_curr)
            de_ = dr_ - dq_curr
            dde_ = ddr_ - (dq_curr - dq_old)*f_robot
            
            e_ep.append(e_.flatten().clone())
            
            # 1. Update ILC Memory (Error)
            if mode == "ILC" or mode == "RILC" or mode == "NOILC":
                controller.updateMemError(e_=e_, de_=de_, dde_=dde_)
            
            # 2. Get ILC Control
            if mode == "NOILC":
                 uILC_raw = controller.getControl() 
                 uILC = uILC_raw
            elif mode == "ILC" or mode == "RILC":
                if ep > 0:
                    uILC_raw = controller.getControl()
                    uILC = uILC_raw
                else:
                    uILC = torch.zeros(2,1)
                    if hasattr(controller, 'idx'): controller.idx += 1
            else:  # RL mode
                uILC = torch.zeros(2,1)
            
            # 3. RL Control (Only RILC and RL)
            if mode == "RILC" or mode == "RL":
                t_pol = t + dt_pol
                if t_pol <= taskT:
                    r_f, dr_f, _ = current_traj(t=t_pol) # Use CURRENT trajectory for RL obs
                    
                    uRL_old_ep = uRL_old_ep_ts[:,:,i]
                    uILC_old_ep = uILC_old_ep_ts[:,:,i]
                    e_old_ep = e_old_ep_ts[:,:,i]
                    de_old_ep = de_old_ep_ts[:,:,i]
                    
                    # Per RL puro: azzera osservazioni ILC
                    if mode == "RL":
                        uILC_obs = torch.zeros(2,1)
                        uILC_old_ep_obs = torch.zeros(2,1)
                        e_old_ep_obs = torch.zeros(2,1)
                        de_old_ep_obs = torch.zeros(2,1)
                    else:  # RILC
                        uILC_obs = uILC
                        uILC_old_ep_obs = uILC_old_ep
                        e_old_ep_obs = e_old_ep
                        de_old_ep_obs = de_old_ep
                    
                    obs = torch.cat([
                        q_curr.flatten(), dq_curr.flatten(),
                        r_f.flatten(), dr_f.flatten(),
                        uRL_old.flatten(), uILC_obs.flatten(), uILC_old_ep_obs.flatten(),
                        e_old_ep_obs.flatten(), de_old_ep_obs.flatten()], dim=0)
                    
                    obs_np = env.normalize_obs(obs)
                    url_pred, _ = model_rl.predict(obs_np, deterministic=True)
                    uRL = env.rescale_action(url_pred).view(-1,1)
                    
                    # Store history
                    uRL_old_ep_ts[:, :, i] = uRL.clone()
                    uILC_old_ep_ts[:, :, i] = uILC.clone()
                    e_old_ep_ts[:, :, i] = e_.clone()
                    de_old_ep_ts[:, :, i] = de_.clone()
            
            # 4. Interpolate and Step
            duRL = resample_u(u_old=uRL_old, u_new=uRL, num_step=scaling)
            duILC = resample_u(u_old=uILC_old, u_new=uILC, num_step=scaling)
            
            uRL_interp = uRL_old.clone()
            uILC_interp = uILC_old.clone()
            
            for _ in range(scaling):
                r_rob, dr_rob, _ = current_traj(t=t) # Use CURRENT trajectory for PD
                
                # Fast Loop State
                q_fast = torch.zeros(2,1)
                q_fast[0] = torch.from_numpy(data_mj.sensor("q_hip").data); q_fast[1] = torch.from_numpy(data_mj.sensor("q_knee").data)
                dq_fast = torch.zeros(2,1)
                dq_fast[0] = torch.from_numpy(data_mj.sensor("dq_hip").data); dq_fast[1] = torch.from_numpy(data_mj.sensor("dq_knee").data)
                
                # PD
                e_fast = angle_normalize(r_rob - q_fast)
                de_fast = dr_rob - dq_fast
                uFB = torch.matmul(torch.diag(torch.tensor([kp, kp])), e_fast) + torch.matmul(torch.diag(torch.tensor([kv, kv])), de_fast)
                uMB = robot.getGravity(q=q_fast)
                
                # Interp Actions
                uRL_interp += duRL
                uILC_interp += duILC
                
                uTot = uMB + uFB + uRL_interp + uILC_interp
                
                data_mj.ctrl[:] = uTot.flatten().numpy()
                mujoco.mj_step(model_mj, data_mj, nstep=frame_skip)
                mujoco.mj_rnePostConstraint(model_mj, data_mj)
                t += dt_rob

                # Logging for plots
                if ep in trace_episodes:
                    traces[ep]['uMB'].append(uMB.flatten().clone())
                    traces[ep]['uFB'].append(uFB.flatten().clone())
                    traces[ep]['uILC'].append(uILC_interp.flatten().clone())
                    traces[ep]['uRL'].append(uRL_interp.flatten().clone())
                    traces[ep]['q'].append(q_fast.flatten().clone())
                    traces[ep]['q_ref'].append(r_rob.flatten().clone())

                
            uRL_old = uRL_interp.clone()
            uILC_old = uILC_interp.clone()
            
            # 5. Update ILC Memory (Input)
            if mode == "ILC" or mode == "RILC" or mode == "NOILC":
                controller.updateMemInput(uFB + uILC)
            
        # Ep End
        rmse = torch.sqrt(torch.mean(torch.stack(e_ep)**2)).item()
        rmse_per_ep.append(rmse)
        print(f"Ep {ep}: RMSE = {rmse:.4f}")
        
    return rmse_per_ep, traces

if __name__ == '__main__':
    # Run Experiments
    controllers = ["ILC", "NOILC", "RILC", "RL"]
    # modes = [("Nominal", False, "--"), ("Mismatch", True, "-")] 
    modes = [("Nominal", False, "--")] # Only Nominal as requested
    
    # Store traces per mode for comparative plotting
    traces_store = { "Nominal": {}, "Mismatch": {} }
    fresh_traces_store = { "Nominal": {}, "Mismatch": {} }
    
    # 1. Reference B Run (Pre-Training Reference) - Optional for Plot?
    # Actually we only care about Switch Scenario for the plot requested.
    # But code runs Ref B first for RMSE baseline.
    
    ref_results = {}
    # Run Ref B loops...
    for ctrl in controllers:
        ref_results[ctrl] = {}
        for mode_name, is_mismatch, _ in modes:
            print(f"Running {ctrl} - {mode_name} (Reference B)...")
            try:
                rmse, traces = run_experiment(ctrl, mismatch=is_mismatch, scenario="ref_B")
                ref_results[ctrl][mode_name] = rmse
                
                # Store RILC fresh traces for comparative plot
                if ctrl == "RILC":
                    fresh_traces_store[mode_name] = traces
                    
            except Exception as e:
                print(f"Failed Reference {ctrl} {mode_name}: {e}")
                # import traceback; traceback.print_exc()

    # 2. Run Switch Experiments
    results = {}
    
    for ctrl in controllers:
        results[ctrl] = {}
        for mode_name, is_mismatch, _ in modes:
            print(f"Running {ctrl} - {mode_name} (Trajectory Switch Test)...")
            try:
                rmse, traces = run_experiment(ctrl, mismatch=is_mismatch, scenario="switch")
                results[ctrl][mode_name] = rmse
                
                # Store traces for plotting
                if mode_name in traces_store:
                    traces_store[mode_name][ctrl] = traces
                    
            except Exception as e:
                print(f"Failed {ctrl} {mode_name}: {e}")
                results[ctrl][mode_name] = []
                import traceback
                traceback.print_exc()

    # Plotting Helper
    def plot_results(log_scale=False):
        # Setup plotting style
        textsize = 18
        labelsize = 16
        plt.rc('font', family='serif', serif='Times')
        plt.rcParams["text.usetex"] = True
        plt.rc('text.latex', preamble=r'\usepackage[utf8]{inputenc} \usepackage{amsmath} \usepackage{amsfonts}')
        plt.rc('xtick', labelsize=textsize)
        plt.rc('ytick', labelsize=textsize)
        plt.rc('axes', titlesize=textsize)
        plt.rc('legend', fontsize=textsize)
        plt.rc('grid', linestyle='-.', alpha=0.5)
        plt.rc('axes', grid=True)
        plt.rcParams['figure.constrained_layout.use'] = True
        
        # Create single figure
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Define colors for controllers
        colors = {
            "ILC": "tab:orange",
            "NOILC": "tab:blue",
            "RL": "tab:green",
            "RILC": "tab:red",
        }
        
        # Define linestyles for modes
        linestyles = {"Nominal": "--", "Switch": "-"}
        
        # Collect all RMSE values for consistent y-axis
        all_rmse_values = []
        
        for ctrl in controllers:
            color = colors.get(ctrl, "black")
            for mode_name, is_mismatch, _ in modes:
                # Plot Switch Data
                if mode_name in results[ctrl] and results[ctrl][mode_name]:
                    linestyle = linestyles[mode_name]
                    rmse_data = results[ctrl][mode_name]
                    label = f"{ctrl} ({mode_name})"
                    
                    # Plot line
                    ax.plot(rmse_data, label=label, linewidth=2, 
                           color=color, linestyle=linestyle)
                    
                    # Plot markers (hollow circles)
                    ax.scatter(range(len(rmse_data)), rmse_data, marker='o', 
                              facecolors='none', edgecolors=color, s=50)
                    
                    all_rmse_values.extend(rmse_data)
                    
                # Plot Reference B Data (offset by n_ep_initial)
                if mode_name in ref_results[ctrl] and ref_results[ctrl][mode_name]:
                    data = ref_results[ctrl][mode_name]
                    x_axis = np.arange(n_ep_initial, n_ep_initial + len(data))
                    
                    # Use dotted line for Ref B
                    ax.plot(x_axis, data, linestyle=':', color=color, alpha=0.7, linewidth=2,
                           label=f"{ctrl} ({mode_name}, Ref B)")
                    ax.scatter(x_axis, data, marker='s', facecolors='none', edgecolors=color, s=40, alpha=0.7)
                    
                    all_rmse_values.extend(data)
        
        ax.set_xlabel('Episode', fontsize=textsize)
        ax.set_ylabel('RMSE [rad]', fontsize=textsize)
        ax.tick_params(axis='x', labelsize=labelsize)
        ax.tick_params(axis='y', labelsize=labelsize)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True)
        
        # Set y-axis limits
        if all_rmse_values:
            y_max = max(all_rmse_values)
            y_min = min(all_rmse_values) if not log_scale else max(min(all_rmse_values), 1e-4)
            if log_scale:
                ax.set_yscale('log')
                ax.set_ylim(y_min * 0.8, y_max * 1.2)
            else:
                ax.set_ylim(-0.05, y_max * 1.1)
        
        # Add legend
        ax.legend(fontsize=textsize-4, frameon=True, loc='best')
        
        plt.tight_layout()
        
        # Save figures
        img_dir = os.path.join(os.path.dirname(__file__), '..', 'img')
        os.makedirs(img_dir, exist_ok=True)
        
        suffix = "_log" if log_scale else ""
        save_base = f"benchmark_switch_traj{suffix}"
        
        plt.savefig(os.path.join(img_dir, f"{save_base}.pdf"), format='pdf', bbox_inches='tight')
        plt.close()

    # Plot Results (RMSE)
    plot_results(log_scale=False)
    plot_results(log_scale=True)
    
    # Plot Comparative Trajectories
    for mode_name, all_traces_n in traces_store.items():
        if all_traces_n: # If not empty
            suffix = f"_{mode_name.lower()}"
            fresh_trace = fresh_traces_store.get(mode_name, None)
            plot_all_trajectories(all_traces_n, fresh_trace=fresh_trace, suffix=suffix)

