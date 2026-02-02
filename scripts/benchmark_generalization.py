from __init__ import *
import torch
import mujoco
from matplotlib import pyplot as plt
import os
import sys
import functools
import numpy as np
from stable_baselines3 import PPO

# Add local directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from classes.controllers.ilc import ILC_base
from classes.controllers.noilc import NOILC
from classes.controllers.ddilc import DDILC
from classes.robots.manipulator_RR import Sim_RR
from classes.environments.env_rlilc_mjc import Env_RILC as ENV

# --- Configuration Constants ---
abs_path = os.path.join(os.path.dirname((os.path.abspath(__file__))), 'classes')
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
taskT = 1.0 # Duration
n_ep_test = 10 # Testing episodes
kp = 0.0 
kv = 0.25 

# Gains 
le_cfg = 0.0002
lde_cfg = 0.0004
ldde_cfg = 0.0

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
def get_linearized_matrices(robot, q, dq, u, dt, damping=True):
    nq = robot._dim_q
    nu = robot._dim_u
    nx = 2 * nq
    eps = 1e-4
    A = torch.zeros(nx, nx)
    B = torch.zeros(nx, nu)
    
    q_nom = q.clone()
    dq_nom = dq.clone()
    q_bak = robot.q.clone()
    dq_bak = robot.dq.clone()
    
    def get_next_state(q_in, dq_in, u_in):
        robot.setState(q=q_in, dq=dq_in)
        x_next = robot.getNewState(dt=dt, action=u_in, damp_fl=damping)
        return torch.cat([x_next[0], x_next[1]], dim=0)

    x_nom_next = get_next_state(q, dq, u)
    
    for i in range(nq):
        q_p = q.clone(); q_p[i] += eps
        x_p_next = get_next_state(q_p, dq, u)
        A[:, i] = (x_p_next - x_nom_next).flatten() / eps
    for i in range(nq):
        dq_p = dq.clone(); dq_p[i] += eps
        x_p_next = get_next_state(q, dq_p, u)
        A[:, nq + i] = (x_p_next - x_nom_next).flatten() / eps
    for i in range(nu):
        u_p = u.clone(); u_p[i] += eps
        x_p_next = get_next_state(q, dq, u_p)
        B[:, i] = (x_p_next - x_nom_next).flatten() / eps
        
    robot.setState(q=q_bak, dq=dq_bak)
    return A, B

def construct_lifted_model_linearized_nonlinear(robot, q_traj, dq_traj, u_traj, dt, samples, dimU):
    print("Linearizing dynamics along trajectory (Model-Based)...")
    As, Bs = [], []
    C = torch.cat([torch.eye(dimU), torch.zeros(dimU, dimU)], dim=1)
    for k in range(samples):
        q_k = q_traj[:, k].view(-1,1)
        dq_k = dq_traj[:, k].view(-1,1)
        u_k = u_traj[:, k].view(-1,1)
        Ak, Bk = get_linearized_matrices(robot, q_k, dq_k, u_k, dt)
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

# --- GENERALIZATION EXPERIMENT ---
def run_generalization():
    print(f"\n--- Running Generalization Benchmark ---")
    
    # Init Env/Robot Common
    f_policy = int(f_robot / scaling)
    samples = int(taskT*f_policy) + 1
    dt_pol = 1/f_policy
    dt_rob = 1/f_robot
    njoint = 2
    
    frame_skip = int((1/f_robot)/0.002) # approx timestep
    robot = Sim_RR(urdf_path=URDF_PATH, ee_name='LH_ANKLE')
    
    # --- PHASE 1: DEFINITIONS ---
    # Trajectory A (Training)
    QF_A = torch.tensor([[2.4], [-1.4]])
    traj_A = functools.partial(minjerk, qi = torch.tensor([[0.0], [0.0]]), qf = QF_A, duration = taskT)
    
    # Trajectory B (Testing - Different Target)
    QF_B = torch.tensor([[1.5], [-0.8]]) # Smaller amplitude, different direction
    traj_B = functools.partial(minjerk, qi = torch.tensor([[0.0], [0.0]]), qf = QF_B, duration = taskT)
    
    print("Trajectory A: [2.4, -1.4]")
    print("Trajectory B: [1.5, -0.8]")
    
    # Load RL Model (trained on A/Random)
    print(f"Loading RL model...")
    model_rl = PPO.load(model_str)
    env = ENV(taskT=taskT, f_robot=f_robot, scaling=scaling, le=le_cfg, lde=lde_cfg, ldde=ldde_cfg, kp=kp, kv=kv, n_ep_reset=1)

    # --- PHASE 2: PRE-COMPUTATION ON TRAJECTORY A ---
    # We construct NOILC and DDILC controllers *specifically for Trajectory A*
    
    # 2.1 NOILC Setup (Linearized on A)
    u_traj_A = torch.zeros(njoint, samples)
    q_traj_A = torch.zeros(njoint, samples)
    dq_traj_A = torch.zeros(njoint, samples)
    for i in range(samples):
        r_val, dr_val, ddr_val = traj_A(t=i*dt_pol)
        q_traj_A[:, i] = r_val.flatten()
        dq_traj_A[:, i] = dr_val.flatten()
        tau_ref = robot.getInvDyn(r_val, dr_val, ddr_val, damp_fl=True)
        u_traj_A[:, i] = tau_ref.flatten()
        
    print("Constructing NOILC G matrix for Trajectory A...")
    G_NOILC_A = construct_lifted_model_linearized_nonlinear(robot, q_traj_A, dq_traj_A, u_traj_A, dt=dt_pol, samples=samples, dimU=njoint)
    
    # 2.2 DDILC Setup (Identified on A)
    # Using Simulation Model for ID (Nominal or Mismatched doesn't matter for the "Process", 
    # but let's use Nominal here just to show purely generalization issue, or Mismatched to show both)
    # Let's use Mismatched Model for everything to be realistic
    model_mj = mujoco.MjModel.from_xml_path(MJC_PATH)
    model_mj.body_mass[:] = model_mj.body_mass * 1.2 # Mismatch
    data_mj = mujoco.MjData(model_mj)
    
    print("Identifying DDILC G matrix for Trajectory A...")
    # Instantiate just to get G
    gravity_comp = robot.getGravity(q=robot.q0).flatten().numpy()
    
    Q_mat = 10.0 * torch.eye(njoint * samples)
    R_mat = 1.0 * torch.eye(njoint * samples)
    
    ddilc_temp = DDILC(njoint, samples, model_mj, data_mj, u_traj_A, dt_pol, frame_skip, 
                       robot.q0.flatten().numpy(), robot.dq0.flatten().numpy(), gravity_comp, 
                       scaling=scaling, Q=Q_mat, R=R_mat, epsilon=1e-2)
    G_DDILC_A = ddilc_temp.G # Extracted identified G
    
    # --- PHASE 3: TESTING ON TRAJECTORY B ---
    # Helper to run episode on Traj B
    def run_on_B(controller_type, G_matrix=None, policy=None):
        print(f"Testing {controller_type} on Trajectory B...")
        
        # Controller Init (Using G_matrix from A if applicable)
        ctrl = None
        Q_mat = 10.0 * torch.eye(njoint * samples)
        R_mat = 1.0 * torch.eye(njoint * samples) # Conservative
        
        if controller_type == "NOILC":
            ctrl = NOILC(njoint, samples, G_matrix, Q_mat, R_mat, threshold=1e-4) # Use G_A
            # Initial Guess? Usually Traj A Reference or Zero. 
            # If we transfer, we might use Traj A Ref (bad) or Traj B Ref (Model Based Guess).
            # To be fair: Let's assume we know Traj B geometry, so we can compute InvDyn(Traj B) for Feedforward Base,
            # BUT the ILC Update Logic uses G_A.
            # OR: We use u_prev from Traj A? No, that's absurd.
            # Best Case "Transfer": Compute Feedforward for B using Nominal Model, Use G_A for Learning.
            
            # Compute FB for B
            u_traj_B = torch.zeros(njoint, samples)
            for i in range(samples):
                r_val, dr_val, ddr_val = traj_B(t=i*dt_pol)
                tau_ref = robot.getInvDyn(r_val, dr_val, ddr_val, damp_fl=True)
                u_traj_B[:, i] = tau_ref.flatten()
            ctrl.uEp = u_traj_B.clone() # Give it the best head start (Nominal InvDyn of B)
            ctrl.best_u = u_traj_B.clone()
            ctrl.newEp()
            
        elif controller_type == "DDILC":
            # Same: Use G_A but Feedforward for B
            ctrl = NOILC(njoint, samples, G_matrix, Q_mat, R_mat, threshold=1e-4) # DDILC is just NOILC with Identified G
            
            u_traj_B = torch.zeros(njoint, samples)
            for i in range(samples):
                r_val, dr_val, ddr_val = traj_B(t=i*dt_pol)
                tau_ref = robot.getInvDyn(r_val, dr_val, ddr_val, damp_fl=True)
                u_traj_B[:, i] = tau_ref.flatten()
            ctrl.uEp = u_traj_B.clone()
            ctrl.best_u = u_traj_B.clone()
            ctrl.newEp()
            
        elif controller_type == "RILC":
            # RILC uses Policy + ILC.
            # ILC base needs initialized.
            le = torch.tensor(le_cfg * f_policy)
            lde = torch.tensor(lde_cfg * f_policy)
            ctrl = ILC_base(njoint, samples, Le=le, Lde=lde)
            ctrl.newEp()
            
        # Run Episodes
        rmse_history = []
        
        # Storage for RILC
        uRL_old_ep_ts = torch.zeros(njoint,1,samples)
        uILC_old_ep_ts = torch.zeros(njoint,1,samples)
        e_old_ep_ts = torch.zeros(njoint,1,samples)
        de_old_ep_ts = torch.zeros(njoint,1,samples)
        
        for ep in range(n_ep_test):
            if ep > 0:
                ctrl.stepILC()
                
            # Rest Env
            mujoco.mj_resetData(model_mj, data_mj)
            # Init state for B (0,0)
            data_mj.qpos[:] = 0; data_mj.qvel[:] = 0
            mujoco.mj_forward(model_mj, data_mj)
            
            # Variables
            e_ep = []
            dq_old = torch.zeros(2,1)
            t = 0.0
            
            if hasattr(ctrl, 'idx'): ctrl.idx = 0
                
            for i in range(samples):
                # Desired Traj B
                r_, dr_, ddr_ = traj_B(t=t)
                
                # Feedback
                q_curr = torch.from_numpy(np.concatenate([data_mj.sensor("q_hip").data, data_mj.sensor("q_knee").data])).view(2,1).float()
                dq_curr = torch.from_numpy(np.concatenate([data_mj.sensor("dq_hip").data, data_mj.sensor("dq_knee").data])).view(2,1).float()
                q_curr += 1e-6 * torch.randn(2,1); dq_curr += 2.5e-4 * torch.randn(2,1)
                
                # Error
                e_ = angle_normalize(r_ - q_curr)
                de_ = dr_ - dq_curr
                dde_ = ddr_ - (dq_curr - dq_old)*f_robot
                dq_old = dq_curr.clone()
                e_ep.append(e_.flatten().clone())
                
                ctrl.updateMemError(e_, de_, dde_)
                
                # Get Actions
                uILC = torch.zeros(2,1)
                if controller_type in ["NOILC", "DDILC"]:
                     uILC = ctrl.getControl()
                else: 
                     if ep>0: uILC = ctrl.getControl()
                     else: 
                         if hasattr(ctrl, 'idx'): ctrl.idx += 1
                
                uRL = torch.zeros(2,1)
                if controller_type == "RILC":
                     # RILC Logic
                     t_pol = t + dt_pol
                     if t_pol <= taskT:
                        r_f, dr_f, _ = traj_B(t=t_pol) # Future ref from B
                        obs = torch.cat([q_curr, dq_curr, r_f, dr_f, 
                                         torch.zeros(njoint,1), uILC, uILC_old_ep_ts[:,:,i], e_old_ep_ts[:,:,i], de_old_ep_ts[:,:,i]]).flatten()
                        # Note: Simple obs construction, assuming history update later
                        obs_np = env.normalize_obs(obs)
                        url_pred, _ = policy.predict(obs_np, deterministic=True)
                        uRL = env.rescale_action(url_pred).view(-1,1)
                        # Update Hist
                        uRL_old_ep_ts[:,:,i] = uRL.clone(); uILC_old_ep_ts[:,:,i] = uILC.clone()
                        e_old_ep_ts[:,:,i] = e_.clone(); de_old_ep_ts[:,:,i] = de_.clone()
                
                # Step (Scaling)
                # Simplified stepping (no interp for brevity as proof of concept)
                # Just hold
                uTot_base = robot.getGravity(q=q_curr) + uRL + uILC + torch.tensor([[kp], [kp]])*e_ + torch.tensor([[kv], [kv]])*de_
                
                for _ in range(scaling):
                     data_mj.ctrl[:] = uTot_base.flatten().numpy()
                     mujoco.mj_step(model_mj, data_mj, nstep=frame_skip)
                     t+=dt_rob
                     
                ctrl.updateMemInput(uILC) # Only ILC part stored typically or full? ILC usually stores its own contribution + error.
                
            rmse = torch.sqrt(torch.mean(torch.stack(e_ep)**2)).item()
            rmse_history.append(rmse)
        return rmse_history

    # Execute Comparisons
    res_noilc = run_on_B("NOILC", G_matrix=G_NOILC_A)
    res_ddilc = run_on_B("DDILC", G_matrix=G_DDILC_A)
    test_env = None # Dummy
    res_rilc = run_on_B("RILC", policy=model_rl)
    
    # Plot
    plt.figure(figsize=(10,6))
    plt.plot(res_noilc, label="NOILC (Trained on A)")
    plt.plot(res_ddilc, label="DD-ILC (Trained on A)")
    plt.plot(res_rilc, label="RILC (Zero-Shot)")
    plt.title("Generalization to Trajectory B (Without Retraining)")
    plt.xlabel("Episode on Traj B")
    plt.ylabel("RMSE [rad]")
    plt.legend()
    plt.grid()
    img_dir = os.path.join(os.path.dirname(__file__), '..', 'img')
    os.makedirs(img_dir, exist_ok=True)
    plt.savefig(os.path.join(img_dir, "benchmark_generalization.png"))
    print("Generalization Plot saved.")

if __name__ == '__main__':
    run_generalization()
