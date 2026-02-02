from __init__ import *
import torch
import mujoco
from matplotlib import pyplot as plt
import os
import functools
import sys
import numpy as np

# Add local directory to path
sys.path.append(os.getcwd())

from classes.controllers.ddilc import DDILC
from classes.controllers.pd import PD_base
from classes.robots.manipulator_RR import Sim_RR
from classes.environments.env_rlilc_mjc import Env_RILC as ENV

from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

# Configuration constants
abs_path = os.path.join(os.path.dirname((os.path.abspath(__file__))), 'classes')
URDF_PATH = os.path.join(abs_path, 'robots/robot_models/softleg_urdf/urdf/leg_constrained.urdf')
MJC_PATH = os.path.join(abs_path, 'robots/robot_models/softleg_urdf/mjc/scene_test.xml')

# Trajectory parameters
QF = torch.tensor([[2.4], [-1.4]])
TRAJ = "minjerk"

# Helper for MinJerk
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

if __name__ == '__main__':
    visual = True 
    
    kp = 0.4
    kv = 0.25
    scaling = 2
    f_robot = 100
    taskT = 1.0
    n_ep_reset = 20
    
    model = mujoco.MjModel.from_xml_path(MJC_PATH)
    
    # Optional: Apply Mismatch ONLY here if we want to test robustness in this script too
    # For now, keep it nominal or apply same mismatch as comparison
    model.body_mass[:] = model.body_mass * 1.2
    
    data = mujoco.MjData(model)
    
    __actual_dt = model.opt.timestep
    frame_skip = int((1/f_robot)/__actual_dt)
    
    f_policy = int(f_robot / scaling)
    samples = int(taskT*f_policy) + 1
    
    noise_q_dev = 1e-6
    noise_dq_dev = 2.5e-4
    njoint = 2
    
    dt_pol = 1/f_policy
    dt_rob = 1/f_robot
    
    # Robot
    robot = Sim_RR(urdf_path=URDF_PATH, ee_name='LH_ANKLE')
    
    if TRAJ == "minjerk":
        des_traj_at = functools.partial(minjerk, qi = torch.tensor([[0.0], [0.0]]), qf = QF, duration = taskT)
    
    # Logging Lists
    e_list = []
    de_list = []
    dde_list = []
    q_list = []
    dq_list = []
    ddq_list = []
    uILC_list = []
    uFB_list = []
    uMB_list = []
    uRL_list = [] 
    
    r_list = torch.zeros(2, samples)
    dr_list = torch.zeros(2, samples)
    ddr_list = torch.zeros(2, samples)
    
    # Ref Traj
    for i in range(samples):
        t = i*dt_pol
        r, dr, ddr = des_traj_at(t=t)
        r_list[:,i] = r.flatten()
        dr_list[:,i] = dr.flatten()
        ddr_list[:,i] = ddr.flatten()

    # Helpers
    tmp_q = des_traj_at(t=0.0)[0].clone()
    tmp_dq = des_traj_at(t=0.0)[1].clone()
    robot.setState(q0=tmp_q, dq0=tmp_dq, q=tmp_q, dq=tmp_dq)
    qi = robot.q0.clone()
    qvel_init = robot.dq0.flatten().numpy().copy()
    qpos_init = robot.q0.flatten().numpy().copy()
    
    # -----------------------------
    # 1. PRE-COMPUTE TRAJECTORY inputs for Initialization/SysID
    # -----------------------------
    u_traj_ref = torch.zeros(njoint, samples)
    
    for i in range(samples):
        t_val = i * dt_pol
        r_val, dr_val, ddr_val = des_traj_at(t=t_val)
        tau_ref = robot.getInvDyn(r_val, dr_val, ddr_val, damp_fl=True)
        u_traj_ref[:, i] = tau_ref.flatten()

    # -----------------------------
    # 2. INSTANTIATE DD-ILC (Performs SysID)
    # -----------------------------
    q_weight = 10.0 # Aggressive
    r_weight = 1.0  # Stable R
    
    Q_mat = q_weight * torch.eye(njoint * samples)
    R_mat = r_weight * torch.eye(njoint * samples)
    
    gravity_comp = robot.getGravity(q=qi).flatten().numpy()
    
    ddilc_ctrl = DDILC(
        dimU=njoint, 
        samples=samples, 
        model_mj=model, 
        data_mj=data, 
        u_nom=u_traj_ref, 
        dt=dt_pol, 
        frame_skip=frame_skip, 
        q_init=qpos_init, 
        dq_init=qvel_init, 
        gravity_comp=gravity_comp,
        scaling=scaling,
        Q=Q_mat, 
        R=R_mat, 
        threshold=1e-4, 
        epsilon=1e-2
    )
    
    # Initial Guess
    ddilc_ctrl.uEp = u_traj_ref.clone()
    ddilc_ctrl.best_u = u_traj_ref.clone()
    
    ddilc_ctrl.newEp()
    
    if visual:
        mujoco_renderer = MujocoRenderer(model, data, None, 800, 600)
    
    for ep in range(n_ep_reset):
        print(f"Episode {ep}")
        if ep > 0:
            ddilc_ctrl.stepILC()
        
        # Reset
        mujoco.mj_resetData(model, data)
        data.qpos = qpos_init
        data.qvel = qvel_init
        mujoco.mj_inverse(model, data)
        data.ctrl[:] = robot.getGravity(q=qi).flatten()
        mujoco.mj_forward(model, data)
        
        dq_old = torch.as_tensor(data.qvel).view(2,1).clone()
        
        # Per Episode Logs
        e_tmp, de_tmp, dde_tmp = [], [], []
        q_tmp, dq_tmp, ddq_tmp = [], [], []
        uILC_tmp, uFB_tmp, uMB_tmp, uRL_tmp_log = [], [], [], []
        
        t = 0.0
        ddilc_ctrl.idx = 0 
        
        for i in range(samples):
            r_, dr_, ddr_ = des_traj_at(t=t)
            
            q_curr = torch.zeros(2,1)
            dq_curr = torch.zeros(2,1)
            q_curr[0] = torch.from_numpy(data.sensor("q_hip").data)
            q_curr[1] = torch.from_numpy(data.sensor("q_knee").data)
            dq_curr[0] = torch.from_numpy(data.sensor("dq_hip").data)
            dq_curr[1] = torch.from_numpy(data.sensor("dq_knee").data)
            
            q_curr += noise_q_dev * torch.randn(2,1)
            dq_curr += noise_dq_dev * torch.randn(2,1)
            
            ddq_curr = (dq_curr - dq_old)*f_robot
            dq_old = dq_curr.clone()
            
            e_ = angle_normalize(r_ - q_curr)
            de_ = dr_ - dq_curr
            dde_ = ddr_ - ddq_curr
            
            # Update Memory
            ddilc_ctrl.updateMemError(e_=e_, de_=de_, dde_=dde_)
            
            # Control Calculation
            uMB = robot.getGravity(q=q_curr)
            uFB = torch.matmul(torch.diag(torch.tensor([kp, kp])), e_) + \
                  torch.matmul(torch.diag(torch.tensor([kv, kv])), de_)
            
            if ep > 0:
                uILC = ddilc_ctrl.getControl()
            else:
                uILC = torch.zeros(2,1)
                ddilc_ctrl.idx += 1
            
            ddilc_ctrl.updateMemInput(uILC + uFB)
            
            # Log
            e_tmp.append(e_.flatten().clone())
            de_tmp.append(de_.flatten().clone())
            dde_tmp.append(dde_.flatten().clone())
            q_tmp.append(q_curr.flatten().clone())
            dq_tmp.append(dq_curr.flatten().clone())
            ddq_tmp.append(ddq_curr.flatten().clone())
            uMB_tmp.append(uMB.flatten().clone())
            uFB_tmp.append(uFB.flatten().clone())
            uILC_tmp.append(uILC.flatten().clone())
            uRL_tmp_log.append(torch.zeros(2).clone()) 
            
            uTot = uMB + uFB + uILC
            
            for _ in range(scaling):
                data.ctrl[:] = uTot.flatten().numpy()
                mujoco.mj_step(model, data, nstep=frame_skip)
                mujoco.mj_rnePostConstraint(model, data)
                t += dt_rob
            
            if visual:
                mujoco_renderer.render("human")
        
        # Append Ep Logs
        e_list.append(e_tmp)
        de_list.append(de_tmp)
        dde_list.append(dde_tmp)
        q_list.append(q_tmp)
        dq_list.append(dq_tmp)
        ddq_list.append(ddq_tmp)
        uMB_list.append(uMB_tmp)
        uFB_list.append(uFB_tmp)
        uILC_list.append(uILC_tmp)
        uRL_list.append(uRL_tmp_log)

    if visual:
        mujoco_renderer.close()

    # ---- PLOTTING ----
    # 1. Console print
    for i in range(len(e_list)):
        rmse_list = torch.sqrt(torch.mean(torch.stack(e_list[i])**2))
        print(f"ddilc MSE of episode: {i}", rmse_list)
        
    # 2. Detailed Plot First and Last Episode
    for i in [0, n_ep_reset-1]:
        plt.figure(figsize=(8, 8))
        plt.subplot(2,3,1)
        plt.plot(torch.stack(e_list[i]).T[0,:], label="sim e1")
        plt.plot(torch.stack(e_list[i]).T[1,:], label="sim e2")
        plt.xlabel("Time steps")
        plt.ylabel("Error [$rad$]")
        plt.title(f"Error")
        plt.grid()
        plt.subplot(2,3,2)
        plt.plot(torch.stack(de_list[i]).T[0,:], label="sim de1")
        plt.plot(torch.stack(de_list[i]).T[1,:], label="sim de2")
        plt.xlabel("Time steps")
        plt.ylabel("Dot error [$rad/s$]")
        plt.title(f"Dot Error")
        plt.grid()    
        plt.subplot(2,3,3)
        plt.plot(torch.stack(dde_list[i]).T[0,:], label="sim dde1")
        plt.plot(torch.stack(dde_list[i]).T[1,:], label="sim dde2")
        plt.xlabel("Time steps")
        plt.ylabel("DDot error [$rad/s^2$]")
        plt.title(f"DDot Error  ")
        plt.grid()    
        plt.subplot(2,3,4)
        plt.plot(torch.stack(q_list[i]).T[0,:], label="sim q1")
        plt.plot(torch.stack(q_list[i]).T[1,:], label="sim q2")
        plt.plot(r_list[0,:], label="ref q1")
        plt.plot(r_list[1,:], label="ref q2")
        plt.xlabel("Time steps")
        plt.ylabel("Angle [$rad$]")
        plt.legend()
        plt.title(f"Joints' Angle in episode  {i+1}")
        plt.grid()
        plt.subplot(2,3,5)
        plt.plot(torch.stack(dq_list[i]).T[0,:], label="sim dq1")
        plt.plot(torch.stack(dq_list[i]).T[1,:], label="sim dq2")
        plt.plot(dr_list[0,:], label="ref dq1")
        plt.plot(dr_list[1,:], label="ref dq2")
        plt.xlabel("Time steps")
        plt.ylabel("Dot Angle [$rad/s$]")
        plt.title(f"Joints' Dot Angle")
        plt.grid()
        plt.legend()
        plt.subplot(2,3,6)
        plt.plot(torch.stack(ddq_list[i]).T[0,:], label="sim ddq1")
        plt.plot(torch.stack(ddq_list[i]).T[1,:], label="sim ddq2")
        plt.plot(ddr_list[0,:], label="ref ddq1")
        plt.plot(ddr_list[1,:], label="ref ddq2")
        plt.xlabel("Time steps")
        plt.ylabel("DDot Angle [$rad/s^2$]")
        plt.title(f"Joints' DDot Angle")
        plt.legend()
        plt.grid()
        plt.suptitle(f"DD-ILC in  episode {i+1}")
        plt.tight_layout()
        plt.savefig(f"ddilc_detailed_ep_{i}.png")

    # 3. Control Components Plot
    for i in [0, n_ep_reset-1]:
        plt.figure(figsize=(15, 3))
        
        plt.subplot(1, 6, 1)
        uT = torch.stack(uMB_list[i]) + torch.stack(uILC_list[i]) + torch.stack(uFB_list[i])
        plt.plot(uT[:,0])
        plt.plot(uT[:,1])
        plt.title("uTOT")
        plt.grid()
        
        plt.subplot(1, 6, 2)
        plt.plot(torch.stack(uILC_list[i])[:,0])
        plt.plot(torch.stack(uILC_list[i])[:,1])
        plt.title("uRL+uILC")
        plt.grid()
        
        plt.subplot(1, 6, 3)
        plt.plot(torch.stack(uMB_list[i])[:,0])
        plt.plot(torch.stack(uMB_list[i])[:,1])
        plt.title("uMB")
        plt.grid()
        
        plt.subplot(1, 6, 4)
        plt.plot(torch.stack(uILC_list[i])[:,0])
        plt.plot(torch.stack(uILC_list[i])[:,1])
        plt.title("uILC")
        plt.grid()
        
        plt.subplot(1, 6, 5)
        plt.plot(torch.stack(uFB_list[i])[:,0])
        plt.plot(torch.stack(uFB_list[i])[:,1])
        plt.title("uFB")
        plt.grid()
        
        plt.subplot(1, 6, 6)
        plt.plot(torch.stack(uRL_list[i])[:,0])
        plt.plot(torch.stack(uRL_list[i])[:,1])
        plt.title("uRL")
        plt.grid()
        
        plt.suptitle(f"DD-ILC Episode {i+1}")
        plt.tight_layout()
        plt.savefig(f"ddilc_controls_ep_{i}.png")
        
    print("All plots saved.")
