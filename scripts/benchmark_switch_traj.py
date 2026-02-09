import os
import sys

# Add local directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from __init__ import *
import torch
import mujoco
from matplotlib import pyplot as plt
import functools
import numpy as np
from stable_baselines3 import PPO

# --- Plotting Style ---
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
FONT_SIZE = 15
plt.rcParams['font.size'] = FONT_SIZE
plt.rcParams['axes.labelsize'] = FONT_SIZE
plt.rcParams['xtick.labelsize'] = FONT_SIZE
plt.rcParams['ytick.labelsize'] = FONT_SIZE
plt.rcParams['legend.fontsize'] = FONT_SIZE
plt.rcParams['figure.titlesize'] = FONT_SIZE

from classes.controllers.ilc import ILC_base
from classes.controllers.noilc import NOILC


from classes.robots.manipulator_RR import Sim_RR
from classes.environments.env_rlilc_mjc import Env_RILC as ENV

# --- Configuration Constants ---
abs_path = os.path.join(os.path.dirname((os.path.abspath(__file__))), '..', 'classes')
URDF_PATH = os.path.join(abs_path, 'robots/robot_models/softleg_urdf/urdf/leg_constrained.urdf')
MJC_PATH = os.path.join(abs_path, 'robots/robot_models/softleg_urdf/mjc/scene_test.xml')

# RL Model Path
parent_str = "model"
dat_str = "rilc_16"
step_str = "best_model/best_model.zip"
model_str = os.path.join(os.getcwd(), parent_str, dat_str, step_str)

# Simulation Param
f_robot = 100
scaling = 2
taskT = 1.0
n_ep_initial = 20
n_ep_switch = 10
n_ep_total = n_ep_initial + n_ep_switch
kp = 0.0 # KPI=0 to isolate ILC performance
kv = 0.25 

# Gains from config.yaml
le_cfg = 0.0002
lde_cfg = 0.0004
ldde_cfg = 0.0

# Trajectories
QF_A = torch.tensor([[2.4], [-1.4]])
QF_B = torch.tensor([[-1.5], [0.0]]) # New Target

# --- Helper Functions ---
def minjerk(qi:torch.Tensor,qf:torch.Tensor,duration:float,t:float) -> list[torch.Tensor,torch.Tensor,torch.Tensor]:
    delta_q = qi-qf
    q_new   = qi + delta_q * (15*(t/duration)**4 - 6*(t/duration)**5 - 10*(t/duration)**3)
    dq_new  = delta_q * (60*(t**3)/(duration**4) - 30*((t**4)/(duration**5)) - 30*(t**2)/(duration**3))
    ddq_new = delta_q * (180*(t**2)/(duration**4) - 120*((t**3)/(duration**5)) - 60*(t/(duration**3)))
    return q_new, dq_new, ddq_new

def angle_normalize(x:torch.Tensor) -> torch.Tensor:
    sx = torch.sin(x)
    cx = torch.cos(x)
    x = torch.atan2(sx,cx)
    return x

def resample_u(u_old:torch.Tensor, u_new:torch.Tensor, num_step:int) -> torch.Tensor:
    du_step = (u_new-u_old)/num_step
    return du_step

# --- Linearization for NOILC ---
def construct_lifted_model_linearized_nonlinear(robot, q_traj, dq_traj, u_traj, dt, samples, dimU, use_analytical=True):
    print(f"Linearizing dynamics along trajectory (Model-Based, Analytical={use_analytical})...")
    As, Bs = [], []
    C = torch.cat([torch.eye(dimU), torch.zeros(dimU, dimU)], dim=1)
    
    for k in range(samples):
        q_k = q_traj[:, k].view(-1,1)
        dq_k = dq_traj[:, k].view(-1,1)
        u_k = u_traj[:, k].view(-1,1)
        
        # Get Continuous A, B from Pinocchio (Analytical)
        Ac, Bc = robot.getAnalyticalLinearizedDynamics(q=q_k, dq=dq_k, u=u_k, damp_fl=True)
        # Discretize (Euler)
        Ak = torch.eye(2*dimU) + Ac * dt
        Bk = Bc * dt
            
        As.append(Ak); Bs.append(Bk)
        
    G = torch.zeros(dimU * samples, dimU * samples)
    print("Building Lifted G Matrix (Model-Based)...")
    for c_t in range(samples):
        Bj = Bs[c_t]
        for input_idx in range(dimU):
            x_curr = Bj[:, input_idx].view(-1,1)
            y_curr = C @ x_curr
            for r_t in range(c_t + 1, samples):
                 row_base = r_t * dimU
                 col_idx = c_t * dimU + input_idx
                 G[row_base:row_base+dimU, col_idx] = y_curr.flatten()
                 if r_t < samples - 1:
                     x_curr = As[r_t] @ x_curr
                     y_curr = C @ x_curr
    return G

# --- SIMULATION RUNNER ---
def run_experiment(mode="ILC", mismatch=False, scenario="switch"):
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
        model_mj.dof_frictionloss[:] = model_mj.dof_frictionloss * 1.2
    
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
    
    if mode == "RILC":
        print(f"Loading RL model from {model_str}")
        model_rl = PPO.load(model_str)
        
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
        R_mat = 0.09 * torch.eye(njoint * samples) 
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
            # Note: For NOILC, we are NOT re-linearizing. We test robustness.
            # For ILC, we continue learning (memory persists). Adaptation test.
            
        # Step ILC/NOILC logic
        if ep == 0:
            pass # Already called newEp
        else:
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
        
        if hasattr(controller, 'idx'): controller.idx = 0
            
        for i in range(samples):
            # Target (Switched)
            r_, dr_, ddr_ = current_traj(t=t)
            
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
            controller.updateMemError(e_=e_, de_=de_, dde_=dde_)
            
            # 2. Get ILC Control
            if mode == "NOILC":
                 uILC_raw = controller.getControl() 
                 uILC = uILC_raw
            else:
                if ep > 0:
                    uILC_raw = controller.getControl()
                    uILC = uILC_raw
                else:
                    uILC = torch.zeros(2,1)
                    if hasattr(controller, 'idx'): controller.idx += 1
            
            # 3. RL Control (Only RILC)
            if mode == "RILC":
                t_pol = t + dt_pol
                if t_pol <= taskT:
                    r_f, dr_f, _ = current_traj(t=t_pol) # Use CURRENT trajectory for RL obs
                    
                    uRL_old_ep = uRL_old_ep_ts[:,:,i]
                    uILC_old_ep = uILC_old_ep_ts[:,:,i]
                    e_old_ep = e_old_ep_ts[:,:,i]
                    de_old_ep = de_old_ep_ts[:,:,i]
                    
                    obs = torch.cat([
                        q_curr.flatten(), dq_curr.flatten(),
                        r_f.flatten(), dr_f.flatten(),
                        uRL_old.flatten(), uILC.flatten(), uILC_old_ep.flatten(),
                        e_old_ep.flatten(), de_old_ep.flatten()], dim=0)
                    
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
                
            uRL_old = uRL_interp.clone()
            uILC_old = uILC_interp.clone()
            
            # 5. Update ILC Memory (Input)
            controller.updateMemInput(uFB + uILC)
            
        # Ep End
        rmse = torch.sqrt(torch.mean(torch.stack(e_ep)**2)).item()
        rmse_per_ep.append(rmse)
        print(f"Ep {ep}: RMSE = {rmse:.4f}")
        
    return rmse_per_ep

if __name__ == '__main__':
    # Run Experiments
    controllers = ["ILC", "NOILC", "RILC"]
    modes = [("Nominal", False, "--"), ("Mismatch", True, "-")] 
    
    results = {}
    
    # 1. Run Switch Experiments
    for ctrl in controllers:
        results[ctrl] = {}
        for mode_name, is_mismatch, _ in modes:
            print(f"Running {ctrl} - {mode_name} (Trajectory Switch Test)...")
            try:
                rmse = run_experiment(ctrl, mismatch=is_mismatch, scenario="switch")
                results[ctrl][mode_name] = rmse
            except Exception as e:
                print(f"Failed {ctrl} {mode_name}: {e}")
                results[ctrl][mode_name] = []
                
    # 2. Run Reference B Experiments (Only Mismatch usually? Or both? Let's do both to match)
    # We store these separately
    ref_results = {}
    for ctrl in controllers:
        ref_results[ctrl] = {}
        for mode_name, is_mismatch, _ in modes:
             print(f"Running {ctrl} - {mode_name} (Reference B)...")
             try:
                 rmse = run_experiment(ctrl, mismatch=is_mismatch, scenario="ref_B")
                 ref_results[ctrl][mode_name] = rmse
             except Exception as e:
                 print(f"Failed Reference {ctrl} {mode_name}: {e}")
                 ref_results[ctrl][mode_name] = []

    # Plotting Helper
    def plot_results(log_scale=False):
        plt.figure(figsize=(10,6))
        
        # Define colors for controllers
        colors = {
            "ILC": "orange",
            "NOILC": "dodgerblue",
            "RILC": "tomato",
        }
        
        for ctrl in controllers:
            color = colors.get(ctrl, "black")
            for mode_name, is_mismatch, linestyle in modes:
                # Plot Switch Data
                if mode_name in results[ctrl] and results[ctrl][mode_name]:
                    label = f"{ctrl} ({mode_name})"
                    plt.plot(results[ctrl][mode_name], marker='o', linestyle=linestyle, color=color, label=label)
                    
                # Plot Reference B Data
                # Offset by n_ep_initial (20)
                if mode_name in ref_results[ctrl] and ref_results[ctrl][mode_name]:
                    data = ref_results[ctrl][mode_name]
                    x_axis = np.arange(n_ep_initial, n_ep_initial + len(data))
                    # Use lighter color or dotted line
                    plt.plot(x_axis, data, linestyle=':', color=color, alpha=0.7) 
                    # No label to avoid clutter, or maybe "Ref" in legend?
                    # The user knows.
            
        from matplotlib.ticker import MaxNLocator
        plt.title(r"Trajectory Switch (Ep 20): Adaptation vs Fresh Start (Dotted)")
        plt.xlabel(r"Episode")
        plt.ylabel(r"RMSE [rad]")
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Draw vertical line at switch
        plt.axvline(x=n_ep_initial, color='k', linestyle='-', alpha=0.3, label="Switch / Start B")
        
        suffix = ""
        if log_scale:
            plt.yscale('log')
            suffix = "_log"
            
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.5)
        
        img_dir = os.path.join(os.path.dirname(__file__), '..', 'img')
        os.makedirs(img_dir, exist_ok=True)
        
        save_base = f"benchmark_switch_traj{suffix}"
        plt.savefig(os.path.join(img_dir, f"{save_base}.pdf"))
        print(f"Saved plot to {save_base}.pdf")

    # Plot Linear Scale
    plot_results(log_scale=False)
    
    # Plot Log Scale
    plot_results(log_scale=True)
