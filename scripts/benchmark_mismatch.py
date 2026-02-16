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
n_ep_reset = 20
kp = 0.0 # KPI=0 to isolate ILC performance
kv = 0.25 

# Gains from config.yaml
le_cfg = 0.0002
lde_cfg = 0.0004
ldde_cfg = 0.0

# Trajectory
QF = torch.tensor([[2.4], [-1.4]])


# --- SIMULATION RUNNER ---
def run_experiment(mode="ILC", mismatch=False):
    print(f"\n--- Running Experiment: {mode} (Mismatch={mismatch}) ---")
    
    # Init Env/Robot
    f_policy = int(f_robot / scaling)
    samples = int(taskT*f_policy) + 1
    dt_pol = 1/f_policy
    dt_rob = 1/f_robot
    njoint = 2
    
    # Env for normalization
    env = ENV(taskT=taskT, f_robot=f_robot, scaling=scaling, 
              le=le_cfg, lde=lde_cfg, ldde=ldde_cfg, kp=kp, kv=kv, n_ep_reset=n_ep_reset)
    
    model_mj = mujoco.MjModel.from_xml_path(MJC_PATH)
    
    if mismatch:
        # 20% Mismatch applied to simulation model
        print("Applying 20% Mass/Friction Mismatch...")
        model_mj.body_mass[:] = model_mj.body_mass * 1.2
        # model_mj.dof_frictionloss[:] = model_mj.dof_frictionloss * 1.2
    
    data_mj = mujoco.MjData(model_mj)
    frame_skip = int((1/f_robot)/model_mj.opt.timestep)
    
    robot = Sim_RR(urdf_path=URDF_PATH, ee_name='LH_ANKLE')
    des_traj_at = functools.partial(minjerk, qi = torch.tensor([[0.0], [0.0]]), qf = QF, duration = taskT)
    
    # Init robot state
    tmp_q = des_traj_at(t=0.0)[0].clone()
    tmp_dq = des_traj_at(t=0.0)[1].clone()
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
        print(f"Loading RL model from {MODEL_RILC_STR}")
        model_rl = PPO.load(MODEL_RILC_STR)

        
    if mode == "ILC" or mode == "RILC":
        le_tens = torch.tensor(le_cfg * f_policy)
        lde_tens = torch.tensor(lde_cfg * f_policy)
        ldde_tens = torch.tensor(ldde_cfg * f_policy)
        controller = ILC_base(dimU=njoint, samples=samples, Le=le_tens, Lde=lde_tens, Ldde=ldde_tens)
        controller.newEp()
        
    elif mode == "NOILC":
        # Precompute Traj Ref for Linearization
        u_traj_ref = torch.zeros(njoint, samples)
        q_traj_ref = torch.zeros(njoint, samples)
        dq_traj_ref = torch.zeros(njoint, samples)
        for i in range(samples):
            r_val, dr_val, ddr_val = des_traj_at(t=i*dt_pol)
            q_traj_ref[:, i] = r_val.flatten()
            dq_traj_ref[:, i] = dr_val.flatten()
            tau_ref = robot.getInvDyn(r_val, dr_val, ddr_val, damp_fl=True)
            u_traj_ref[:, i] = tau_ref.flatten()
            
        G_mat = construct_lifted_model_linearized_nonlinear(robot, q_traj_ref, dq_traj_ref, u_traj_ref, dt=dt_pol, samples=samples, dimU=njoint, use_analytical=True)
        Q_mat = 1.0 * torch.eye(njoint * samples)
        R_mat = 0.09 * torch.eye(njoint * samples) 
        controller = NOILC(dimU=njoint, samples=samples, G=G_mat, Q=Q_mat, R=R_mat, threshold=1e-4)
        
        # Initial Guess (Nominal Model)
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
    
    for ep in range(n_ep_reset):
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
            # Target
            r_, dr_, ddr_ = des_traj_at(t=t)
            
            # State
            q_curr = torch.zeros(2,1)
            dq_curr = torch.zeros(2,1)
            q_curr[0] = torch.from_numpy(data_mj.sensor("q_hip").data)
            q_curr[1] = torch.from_numpy(data_mj.sensor("q_knee").data)
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
            
            # 3. RL Control (RILC and RL)
            if mode == "RILC" or mode == "RL":
                t_pol = t + dt_pol
                if t_pol <= taskT:
                    r_f, dr_f, _ = des_traj_at(t=t_pol)
                    
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
                r_rob, dr_rob, _ = des_traj_at(t=t)
                
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
                
            uRL_old = uRL_interp.clone()
            uILC_old = uILC_interp.clone()
            
            # 5. Update ILC Memory (Input)
            if mode == "ILC" or mode == "RILC" or mode == "NOILC":
                controller.updateMemInput(uFB + uILC)
            
        # Ep End
        rmse = torch.sqrt(torch.mean(torch.stack(e_ep)**2)).item()
        rmse_per_ep.append(rmse)
        print(f"Ep {ep}: RMSE = {rmse:.4f}")
        
    return rmse_per_ep

if __name__ == '__main__':
    # Run Experiments
    controllers = ["ILC", "NOILC", "RL", "RILC"]
    modes = [("Nominal", False, "--"), ("Mismatch", True, "-")] # Name, mismatch_bool, linestyle
    
    results = {}
    
    for ctrl in controllers:
        results[ctrl] = {}
        for mode_name, is_mismatch, _ in modes:
            print(f"Running {ctrl} - {mode_name}...")
            try:
                rmse = run_experiment(ctrl, mismatch=is_mismatch)
                results[ctrl][mode_name] = rmse
            except Exception as e:
                print(f"Failed {ctrl} {mode_name}: {e}")
                results[ctrl][mode_name] = []

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
        
        # Create figure with 2 subplots (Nominal and Mismatch)
        fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
        
        # Define colors for controllers
        # colors = {
        #     "ILC": "tab:orange",
        #     "NOILC": "tab:blue",
        #     "RL": "tab:green",
        #     "RILC": "tab:red",
        # }
        colors = {
            "ILC": "orange",
            "NOILC": "dodgerblue",
            "RL": "green",
            "RILC": "tomato",
        }
        # Define linestyles for modes
        linestyles = {"Nominal": "--", "Mismatch": "-"}
        
        # Collect all RMSE values for consistent y-axis
        all_rmse_values = []
        
        for mode_idx, (mode_name, is_mismatch, _) in enumerate(modes):
            ax = axs[mode_idx]
            
            for ctrl in controllers:
                if mode_name in results[ctrl] and results[ctrl][mode_name]:
                    color = colors.get(ctrl, "black")
                    linestyle = linestyles[mode_name]
                    rmse_data = results[ctrl][mode_name]
                    
                    # Plot line
                    line, = ax.plot(rmse_data, label=ctrl, linewidth=2, 
                                   color=color, linestyle=linestyle)
                    
                    # Plot markers (hollow circles)
                    ax.scatter(range(len(rmse_data)), rmse_data, marker='o', 
                              facecolors='none', edgecolors=color, s=50)
                    
                    all_rmse_values.extend(rmse_data)
            
            ax.set_xlabel('Episode', fontsize=textsize)
            if mode_idx == 0:
                ax.set_ylabel('RMSE [rad]', fontsize=textsize)
            
            ax.set_title(mode_name, fontsize=textsize)
            ax.tick_params(axis='x', labelsize=labelsize)
            ax.tick_params(axis='y', labelsize=labelsize)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(True)
        
        # Set consistent y-axis limits
        if all_rmse_values:
            y_max = max(all_rmse_values)
            y_min = min(all_rmse_values) if not log_scale else max(min(all_rmse_values), 1e-4)
            for ax in axs:
                if log_scale:
                    ax.set_yscale('log')
                    ax.set_ylim(y_min * 0.8, y_max * 1.2)
                else:
                    ax.set_ylim(0.0, y_max * 1.1)
        
        # Create common legend
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=len(controllers), 
                  fontsize=textsize, frameon=True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.9])
        
        # Save figures
        img_dir = os.path.join(os.path.dirname(__file__), '..', 'img')
        os.makedirs(img_dir, exist_ok=True)
        
        suffix = "_log" if log_scale else ""
        save_base = f"benchmark_mismatch{suffix}"
        
        plt.savefig(os.path.join(img_dir, f"{save_base}.pdf"), format='pdf', bbox_inches='tight')

    # Plot Linear Scale
    plot_results(log_scale=False)
    
    # Plot Log Scale
    plot_results(log_scale=True)
