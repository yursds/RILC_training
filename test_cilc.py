"""test_cilc.py — Combined ILC (Tsurumoto et al., IFAC 2023)

Tests CILC on the rigid-flexible model (same setup as benchmark_all_controllers.py).
CILC = basis functions (task-flexible) + residual ILC (high-performance).

Usage:  uv run test_cilc.py          (from RILC_training/)
"""

from __init__ import *
import torch
import mujoco
from matplotlib import pyplot as plt
import os
import functools

sys.path.append(os.getcwd())

from classes.controllers.cilc import CILC
from classes.controllers.pd import PD_base
from classes.robots.manipulator_RR import Sim_RR
from classes.environments.env_rlilc_mjc import Env_RILC as ENV

from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

_HERE = os.path.dirname(os.path.abspath(__file__))
abs_path = os.path.join(_HERE, 'classes')
URDF_PATH = os.path.join(abs_path, 'robots/robot_models/softleg_urdf/urdf/leg_constrained.urdf')
MESH_DIR  = os.path.join(abs_path, 'robots/robot_models/softleg_urdf/meshes')
MJC_PATH  = os.path.join(abs_path, 'robots/robot_models/softleg_urdf/mjc/scene_test.xml')

# Config mirrors benchmark_all_controllers.py CILC section
QF = torch.tensor([[2.0], [-1.0]])
PLOT = True

F_ROBOT = 100
SCALING = 2
TASKT = 1.0
N_EP = 20

KP = 0.0
KV = 0.25

LE_CFG = 0.0002
LDE_CFG = 0.0004

F_POLICY = int(F_ROBOT / SCALING)
SAMPLES = int(TASKT * F_POLICY) + 1
DT_POL = 1.0 / F_POLICY
DT_ROB = 1.0 / F_ROBOT
NJOINT = 2

LOAD_PRETRAINED = False
PRETRAINED_PATH = os.path.join(_HERE, 'model/cilc_pretrain/cilc_batch_trained.pt')


def minjerk(qi, qf, duration, t):
    delta_q = qi - qf
    q_new   = qi + delta_q * (15*(t/duration)**4 - 6*(t/duration)**5 - 10*(t/duration)**3)
    dq_new  = delta_q * (60*(t**3)/(duration**4) - 30*((t**4)/(duration**5)) - 30*(t**2)/(duration**3))
    ddq_new = delta_q * (180*(t**2)/(duration**4) - 120*((t**3)/(duration**5)) - 60*(t/(duration**3)))
    return q_new, dq_new, ddq_new

def angle_normalize(x):
    sx, cx = torch.sin(x), torch.cos(x)
    return torch.atan2(sx, cx)

def resample_u(u_old, u_new, num_step):
    du_step = (u_new - u_old) / num_step
    return du_step


def simulate_episode(model_mj, data_mj, robot, traj_fn, controller,
                     qpos_init, qvel_init):
    mujoco.mj_resetData(model_mj, data_mj)
    data_mj.qpos = qpos_init
    data_mj.qvel = qvel_init
    mujoco.mj_inverse(model_mj, data_mj)
    data_mj.ctrl[:] = robot.getGravity(q=robot.q0).flatten()
    mujoco.mj_forward(model_mj, data_mj)

    frame_skip = int((1.0 / F_ROBOT) / model_mj.opt.timestep)

    uILC_old = torch.zeros(NJOINT, 1)
    dq_old = torch.as_tensor(qvel_init).view(NJOINT, 1).clone()

    if controller is not None and hasattr(controller, "idx"):
        controller.idx = 0

    e_ep, de_ep, dde_ep = [], [], []
    q_ep, dq_ep = [], []
    uILC_ep, uFB_ep, uMB_ep = [], [], []
    t = 0.0
    for i in range(SAMPLES):
        r_, dr_, ddr_ = traj_fn(t=t)

        q_c = torch.zeros(2, 1)
        dq_c = torch.zeros(2, 1)
        try:
            q_c[0] = torch.from_numpy(data_mj.sensor("theta_hip").data) + torch.from_numpy(data_mj.sensor("q_hip").data)
            q_c[1] = torch.from_numpy(data_mj.sensor("theta_knee").data) + torch.from_numpy(data_mj.sensor("q_knee").data)
            dq_c[0] = torch.from_numpy(data_mj.sensor("dtheta_hip").data) + torch.from_numpy(data_mj.sensor("dq_hip").data)
            dq_c[1] = torch.from_numpy(data_mj.sensor("dtheta_knee").data) + torch.from_numpy(data_mj.sensor("dq_knee").data)
        except (KeyError, ValueError):
            q_c[0] = torch.from_numpy(data_mj.sensor("q_hip").data)
            q_c[1] = torch.from_numpy(data_mj.sensor("q_knee").data)
            dq_c[0] = torch.from_numpy(data_mj.sensor("dq_hip").data)
            dq_c[1] = torch.from_numpy(data_mj.sensor("dq_knee").data)

        e_ = angle_normalize(r_ - q_c)
        de_ = dr_ - dq_c
        dde_ = ddr_ - (dq_c - dq_old) * F_ROBOT
        dq_old = dq_c.clone()

        if controller is not None:
            controller.updateMemError(e_=e_, de_=de_, dde_=dde_)
            uILC = controller.getControl()
        else:
            uILC = torch.zeros(NJOINT, 1)

        duILC = resample_u(u_old=uILC_old, u_new=uILC, num_step=SCALING)
        uILC_interp = uILC_old.clone()
        for _ in range(SCALING):
            r_rob, dr_rob, _ = traj_fn(t=t)

            q_f = torch.zeros(2, 1)
            dq_f = torch.zeros(2, 1)
            try:
                q_f[0] = torch.from_numpy(data_mj.sensor("theta_hip").data) + torch.from_numpy(data_mj.sensor("q_hip").data)
                q_f[1] = torch.from_numpy(data_mj.sensor("theta_knee").data) + torch.from_numpy(data_mj.sensor("q_knee").data)
                dq_f[0] = torch.from_numpy(data_mj.sensor("dtheta_hip").data) + torch.from_numpy(data_mj.sensor("dq_hip").data)
                dq_f[1] = torch.from_numpy(data_mj.sensor("dtheta_knee").data) + torch.from_numpy(data_mj.sensor("dq_knee").data)
            except (KeyError, ValueError):
                q_f[0] = torch.from_numpy(data_mj.sensor("q_hip").data)
                q_f[1] = torch.from_numpy(data_mj.sensor("q_knee").data)
                dq_f[0] = torch.from_numpy(data_mj.sensor("dq_hip").data)
                dq_f[1] = torch.from_numpy(data_mj.sensor("dq_knee").data)

            e_fast = angle_normalize(r_rob - q_f)
            de_fast = dr_rob - dq_f
            uFB = (torch.diag(torch.tensor([KP, KP])) @ e_fast
                   + torch.diag(torch.tensor([KV, KV])) @ de_fast)
            uMB = robot.getGravity(q=q_f)
            uILC_interp += duILC
            data_mj.ctrl[:] = (uMB + uFB + uILC_interp).flatten().numpy()
            mujoco.mj_step(model_mj, data_mj, nstep=frame_skip)
            mujoco.mj_rnePostConstraint(model_mj, data_mj)
            t += DT_ROB

        uILC_old = uILC_interp.clone()
        if controller is not None:
            controller.updateMemInput(uFB + uILC)

        e_ep.append(e_.flatten().clone())
        de_ep.append(de_.flatten().clone())
        dde_ep.append(dde_.flatten().clone())
        q_ep.append(q_c.flatten().clone())
        dq_ep.append(dq_c.flatten().clone())
        uILC_ep.append(uILC_interp.flatten().clone())
        uFB_ep.append(uFB.flatten().clone())
        uMB_ep.append(uMB.flatten().clone())

    rmse = torch.sqrt(torch.mean(torch.stack(e_ep) ** 2)).item()
    return rmse, e_ep, de_ep, dde_ep, q_ep, dq_ep, uILC_ep, uFB_ep, uMB_ep


if __name__ == '__main__':
    model_mj = mujoco.MjModel.from_xml_path(MJC_PATH)
    data_mj = mujoco.MjData(model_mj)
    robot = Sim_RR(urdf_path=URDF_PATH, mesh_dir=MESH_DIR, ee_name='LH_ANKLE')
    mujoco_renderer = MujocoRenderer(model_mj, data_mj, None, 800, 600)

    qi = torch.tensor([[0.0], [0.0]])
    traj_fn = lambda t: minjerk(qi, QF, TASKT, t)

    q_arr = torch.zeros(NJOINT, SAMPLES)
    dq_arr = torch.zeros(NJOINT, SAMPLES)
    ddq_arr = torch.zeros(NJOINT, SAMPLES)
    for i in range(SAMPLES):
        r, dr, ddr = traj_fn(t=i * DT_POL)
        q_arr[:, i] = r.flatten()
        dq_arr[:, i] = dr.flatten()
        ddq_arr[:, i] = ddr.flatten()

    r_list = q_arr.clone()
    dr_list = dq_arr.clone()

    qi0, _, _ = traj_fn(t=0.0)
    robot.setState(q0=qi0, dq0=qi0, q=qi0, dq=qi0)
    qpos_init = robot.q0.flatten().numpy().copy()
    qvel_init = robot.dq0.flatten().numpy().copy()

    cilc = CILC(
        dimU=NJOINT,
        basis_names=["ddq", "ddq_other", "dq", "dq_other", "sinq", "sinq_other", "bias"],
        gamma=0.8,
        Le=0.5,
        Lde=0.1,
    )

    if LOAD_PRETRAINED:
        if os.path.exists(PRETRAINED_PATH):
            theta = torch.load(PRETRAINED_PATH, map_location='cpu', weights_only=True)
            cilc.theta = theta
            print(f"Loaded pre-trained theta from {PRETRAINED_PATH}")
        else:
            print(f"Warning: {PRETRAINED_PATH} not found, starting from zero")

    cilc.set_trajectory(q_arr, dq_arr, ddq_arr)
    cilc.newEp()

    e_list, de_list, dde_list = [], [], []
    q_list, dq_list = [], []
    uILC_list, uFB_list, uMB_list = [], [], []
    rmse_hist = []

    for ep in range(N_EP):
        rmse, e_ep, de_ep, dde_ep, q_ep, dq_ep, uILC_ep, uFB_ep, uMB_ep = \
            simulate_episode(model_mj, data_mj, robot, traj_fn,
                             cilc, qpos_init, qvel_init)
        rmse_hist.append(rmse)
        e_list.append(e_ep)
        de_list.append(de_ep)
        dde_list.append(dde_ep)
        q_list.append(q_ep)
        dq_list.append(dq_ep)
        uILC_list.append(uILC_ep)
        uFB_list.append(uFB_ep)
        uMB_list.append(uMB_ep)
        print(f"ep={ep:2d}  RMSE={rmse:.5f}")
        if ep < N_EP - 1:
            cilc.stepILC()

    mujoco_renderer.close()

    print(f"\nFinal theta: {[f'{v:.4f}' for v in cilc.theta.flatten().tolist()]}")
    print(f"\nRMSE progression: ep0={rmse_hist[0]:.5f} -> ep{N_EP-1}={rmse_hist[-1]:.5f}")

    if PLOT:
        plt.figure(figsize=(6, 4))
        plt.plot(range(N_EP), rmse_hist, "o-", color="coral", lw=2, label="C-ILC")
        plt.xlabel("Episode $j$")
        plt.ylabel("RMSE [rad]")
        plt.title("CILC — RMSE progression")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

    for ep_idx in [0, N_EP - 1]:
        fig, axs = plt.subplots(2, 2, figsize=(8, 8))
        axs[0, 0].plot(torch.stack(e_list[ep_idx]).T[0, :], label="e1")
        axs[0, 0].plot(torch.stack(e_list[ep_idx]).T[1, :], label="e2")
        axs[0, 0].set_xlabel("Time steps")
        axs[0, 0].set_ylabel("Error [rad]")
        axs[0, 0].set_title(f"Error — episode {ep_idx + 1}")
        axs[0, 0].grid()
        axs[0, 0].legend()

        axs[0, 1].plot(torch.stack(de_list[ep_idx]).T[0, :], label="de1")
        axs[0, 1].plot(torch.stack(de_list[ep_idx]).T[1, :], label="de2")
        axs[0, 1].set_xlabel("Time steps")
        axs[0, 1].set_ylabel("Dot error [rad/s]")
        axs[0, 1].set_title(f"Dot Error — episode {ep_idx + 1}")
        axs[0, 1].grid()
        axs[0, 1].legend()

        axs[1, 0].plot(torch.stack(q_list[ep_idx]).T[0, :], label="sim q1")
        axs[1, 0].plot(torch.stack(q_list[ep_idx]).T[1, :], label="sim q2")
        axs[1, 0].plot(r_list[0, :], label="ref q1")
        axs[1, 0].plot(r_list[1, :], label="ref q2")
        axs[1, 0].set_xlabel("Time steps")
        axs[1, 0].set_ylabel("Angle [rad]")
        axs[1, 0].set_title(f"Joints' Angle — episode {ep_idx + 1}")
        axs[1, 0].grid()
        axs[1, 0].legend()

        axs[1, 1].plot(torch.stack(dq_list[ep_idx]).T[0, :], label="sim dq1")
        axs[1, 1].plot(torch.stack(dq_list[ep_idx]).T[1, :], label="sim dq2")
        axs[1, 1].plot(dr_list[0, :], label="ref dq1")
        axs[1, 1].plot(dr_list[1, :], label="ref dq2")
        axs[1, 1].set_xlabel("Time steps")
        axs[1, 1].set_ylabel("Dot Angle [rad/s]")
        axs[1, 1].set_title(f"Joints' Dot Angle — episode {ep_idx + 1}")
        axs[1, 1].grid()
        axs[1, 1].legend()

        fig.suptitle(f"CILC episode {ep_idx + 1}")
        fig.tight_layout()

    for ep_idx in [0, N_EP - 1]:
        fig, axs = plt.subplots(1, 6, figsize=(15, 3))

        uT = torch.stack(uMB_list[ep_idx]) + torch.stack(uILC_list[ep_idx]) + torch.stack(uFB_list[ep_idx])
        axs[0].plot(uT[:, 0])
        axs[0].plot(uT[:, 1])
        axs[0].set_title("uTOT")
        axs[0].grid()

        axs[1].plot(torch.stack(uILC_list[ep_idx])[:, 0])
        axs[1].plot(torch.stack(uILC_list[ep_idx])[:, 1])
        axs[1].set_title("uILC")
        axs[1].grid()

        axs[2].plot(torch.stack(uMB_list[ep_idx])[:, 0])
        axs[2].plot(torch.stack(uMB_list[ep_idx])[:, 1])
        axs[2].set_title("uMB")
        axs[2].grid()

        axs[3].plot(torch.stack(uILC_list[ep_idx])[:, 0])
        axs[3].plot(torch.stack(uILC_list[ep_idx])[:, 1])
        axs[3].set_title("uILC")
        axs[3].grid()

        axs[4].plot(torch.stack(uFB_list[ep_idx])[:, 0])
        axs[4].plot(torch.stack(uFB_list[ep_idx])[:, 1])
        axs[4].set_title("uFB")
        axs[4].grid()

        axs[5].plot(torch.zeros(SAMPLES))
        axs[5].set_title("uRL (zero)")
        axs[5].grid()

        fig.suptitle(f"CILC Episode {ep_idx + 1}")
        fig.tight_layout()

    if PLOT:
        plt.show()
    else:
        print("Plots not shown (PLOT = False).")
