import argparse
import functools
import os
import sys
import time
from functools import partial

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch
from scipy.optimize import differential_evolution

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from classes.controllers.ilc import ILC_base
from classes.controllers.pd import PD_base
from classes.robots.manipulator_RR import Sim_RR
from stable_baselines3 import PPO


abs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'classes')
URDF_PATH = os.path.join(abs_path, 'robots/robot_models/softleg_urdf/urdf/leg_constrained.urdf')
MESH_DIR = os.path.join(abs_path, 'robots/robot_models/softleg_urdf/meshes')
MJC_PATH = os.path.join(abs_path, 'robots/robot_models/softleg_urdf/mjc/scene_elastic.xml')

parent_str = "model"
dat_str = "rilc_constrained"
step_str = "best_model/best_model.zip"

QF = torch.tensor([[2.4], [-1.4]])
FL_ILC = True
FL_RL = False
OBS_ILC = False


def angle_normalize(x: torch.Tensor) -> torch.Tensor:
    sx = torch.sin(x)
    cx = torch.cos(x)
    return torch.atan2(sx, cx)


def resample_u(u_old: torch.Tensor, u_new: torch.Tensor, num_step: int) -> torch.Tensor:
    return (u_new - u_old) / num_step


def minjerk(qi: torch.Tensor, qf: torch.Tensor, duration: float, t: float):
    delta_q = qi - qf
    q_new = qi + delta_q * (15 * (t / duration) ** 4 - 6 * (t / duration) ** 5 - 10 * (t / duration) ** 3)
    dq_new = delta_q * (60 * (t ** 3) / (duration ** 4) - 30 * ((t ** 4) / (duration ** 5)) - 30 * (t ** 2) / (duration ** 3))
    ddq_new = delta_q * (180 * (t ** 2) / (duration ** 4) - 120 * ((t ** 3) / (duration ** 5)) - 60 * (t / (duration ** 3)))
    return q_new, dq_new, ddq_new


def run_simulation(
    kp: float, kv: float, le: float, lde: float, ldde: float, lddde: float,
    taskT: float = 1.0,
    n_episodes: int = 50,
    scaling: int = 2,
    f_robot: int = 100,
    qi_init: torch.Tensor = None, 
    qf: torch.Tensor = None,
    verbose: bool = False
) -> tuple[float, list[float]]:

    kp = float(kp)
    kv = float(kv)
    le = float(le)
    lde = float(lde)
    ldde = float(ldde)
    lddde = float(lddde)
    
    model = mujoco.MjModel.from_xml_path(MJC_PATH)
    __actual_dt = model.opt.timestep
    frame_skip = int((1 / f_robot) / __actual_dt)
    
    noise_q_dev = 1e-6
    noise_dq_dev = 2.5e-4
    njoint = 2
    
    if qi_init is None:
        qi_init = torch.tensor([[0.0], [0.0]])
    if qf is None:
        qf = QF
    
    des_traj_at = functools.partial(minjerk, qi=qi_init, qf=qf, duration=taskT)
    
    f_policy = int(f_robot / scaling)
    samples = int(taskT * f_policy) + 1
    dt_pol = 1 / f_policy
    dt_rob = 1 / f_robot
    
    robot = Sim_RR(urdf_path=URDF_PATH, mesh_dir=MESH_DIR, ee_name='LH_ANKLE')
    tmp_q = des_traj_at(t=0.0)[0].clone()
    tmp_dq = des_traj_at(t=0.0)[1].clone()
    robot.setState(q0=tmp_q, dq0=tmp_dq, q=tmp_q, dq=tmp_dq)
    robot.u0 = robot.getGravity(robot.q0).clone()
    qi = robot.q0.clone()
    dqi = robot.dq0.clone()
    
    qi_np = robot.q0.flatten().numpy()
    qpos_init = np.array([qi_np[0], qi_np[0], qi_np[1], qi_np[1]])
    dqi_np = robot.dq0.flatten().numpy()
    qvel_init = np.array([dqi_np[0], dqi_np[0], dqi_np[1], dqi_np[1]])
    
    ldde_ts = torch.tensor(ldde * f_policy, dtype=torch.float32)
    lddde_ts = torch.tensor(lddde * f_policy, dtype=torch.float32)
    lde_ts = torch.tensor(lde * f_policy, dtype=torch.float32)
    le_ts = torch.tensor(le * f_policy, dtype=torch.float32)
    
    conILC = ILC_base(
        dimU=njoint,
        samples=samples,
        Le=le_ts,
        Lde=lde_ts,
        Ldde=ldde_ts,
        Lddde=lddde_ts
    )
    conILC.newEp()
    
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos = qpos_init
    data.qvel = qvel_init
    mujoco.mj_inverse(model, data)
    joint_forces = torch.from_numpy(data.qfrc_inverse).clone()
    data.ctrl[:] = joint_forces[:2]
    mujoco.mj_forward(model, data)
    
    conILC.resetAll()
    
    uRL_old_ep_ts = torch.zeros(njoint, 1, samples)
    uILC_old_ep_ts = torch.zeros(njoint, 1, samples)
    uFB_old_ep_ts = torch.zeros(njoint, 1, samples)
    e_old_ep_ts = torch.zeros(njoint, 1, samples)
    de_old_ep_ts = torch.zeros(njoint, 1, samples)
    
    rmse_list = []
    
    for ep in range(n_episodes):
        t = 0.0
        
        if conILC.episodes == 0:
            conILC.newEp()
        else:
            conILC.stepILC()
        
        uFB = torch.zeros(njoint, 1)
        uRL = torch.zeros(njoint, 1)
        uRL_old = torch.zeros(njoint, 1)
        duRL = torch.zeros(njoint, 1)
        uILC = torch.zeros(njoint, 1)
        uILC_old = torch.zeros(njoint, 1)
        duILC = torch.zeros(njoint, 1)
        
        mujoco.mj_resetData(model, data)
        data.qpos = qpos_init
        data.qvel = qvel_init
        mujoco.mj_inverse(model, data)
        joint_forces = torch.from_numpy(data.qfrc_inverse).clone()
        data.ctrl[:] = robot.getGravity(q=qi).flatten()
        mujoco.mj_forward(model, data)
        dq_old = dqi.clone()
        
        e_tmp = []
        de_tmp = []
        dde_tmp = []
        
        q_tmp = []
        dq_tmp = []
        ddq_tmp = []
        
        theta = torch.zeros(2, 1)
        dtheta = torch.zeros(2, 1)
        
        d = torch.zeros(2, samples, 1) * 0.5 * 0
        
        for i in range(samples):
            r_, dr_, ddr_ = des_traj_at(t=t)
            
            deflection = torch.zeros(2, 1)
            dq_raw = torch.zeros(2, 1)
            theta_sample = torch.zeros(2, 1)
            theta_sample[0] = torch.from_numpy(data.sensor("theta_hip").data)
            theta_sample[1] = torch.from_numpy(data.sensor("theta_knee").data)
            deflection[0] = torch.from_numpy(data.sensor("q_hip").data)
            deflection[1] = torch.from_numpy(data.sensor("q_knee").data)
            dq_raw[0] = torch.from_numpy(data.sensor("dq_hip").data)
            dq_raw[1] = torch.from_numpy(data.sensor("dq_knee").data)
            
            q = theta_sample + deflection
            dq = torch.zeros(2, 1)
            q += noise_q_dev * torch.randn(2, 1)
            dq += noise_dq_dev * torch.randn(2, 1)
            ddq = (dq - dq_old) * f_robot
            
            e_ = r_ - q
            e_ = angle_normalize(e_)
            de_ = dr_ - dq
            dde_ = ddr_ - ddq
            
            e_tmp.append(e_.flatten().clone())
            de_tmp.append(de_.flatten().clone())
            dde_tmp.append(dde_.flatten().clone())
            
            q_tmp.append(q.flatten().clone())
            dq_tmp.append(dq.flatten().clone())
            ddq_tmp.append(ddq.flatten().clone())
            
            q_old = robot.q.clone()
            robot.setState(q=q, dq=dq)
            
            u_delta = uFB + uILC
            conILC.updateMemError(e_=e_, de_=de_, dde_=dde_)
            conILC.updateMemInput(u_delta)
            
            if conILC.episodes != 0 and FL_ILC:
                uilc = conILC.getControl()
                uILC = uilc
            else:
                uILC = torch.zeros(2, 1)
            
            duRL = resample_u(u_old=uRL_old, u_new=uRL, num_step=scaling)
            duILC = resample_u(u_old=uILC_old, u_new=uILC, num_step=scaling)
            uRL_interp = uRL_old
            uILC_interp = uILC_old
            
            if FL_RL:
                uRL_old_ep = uRL_old_ep_ts[:, :, i]
                uILC_old_ep = uILC_old_ep_ts[:, :, i]
                uFB_old_ep = uFB_old_ep_ts[:, :, i]
                e_old_ep = e_old_ep_ts[:, :, i]
                de_old_ep = de_old_ep_ts[:, :, i]
                
                uRL_old_ep_ts[:, :, i] = uRL.clone()
                uILC_old_ep_ts[:, :, i] = uILC.clone()
                uFB_old_ep_ts[:, :, i] = uFB.clone()
                e_old_ep_ts[:, :, i] = e_.clone()
                de_old_ep_ts[:, :, i] = de_.clone()
            
            for _ in range(scaling):
                r_, dr_, ddr_ = des_traj_at(t=t)
                
                deflection[0] = torch.from_numpy(data.sensor("q_hip").data)
                deflection[1] = torch.from_numpy(data.sensor("q_knee").data)
                dq_raw[0] = torch.from_numpy(data.sensor("dq_hip").data)
                dq_raw[1] = torch.from_numpy(data.sensor("dq_knee").data)
                
                q = theta + deflection
                dq = dtheta + dq_raw
                q += noise_q_dev * torch.randn(2, 1)
                dq += noise_dq_dev * torch.randn(2, 1)
                ddq = (dq - dq_old) * f_robot
                dq_old = dq.clone()
                
                theta[0] = torch.from_numpy(data.sensor("theta_hip").data)
                theta[1] = torch.from_numpy(data.sensor("theta_knee").data)
                dtheta[0] = torch.from_numpy(data.sensor("dtheta_hip").data)
                dtheta[1] = torch.from_numpy(data.sensor("dtheta_knee").data)
                
                e_ = r_ - q
                e_ = angle_normalize(e_)
                de_ = dr_ - dq
                dde_ = ddr_ - ddq
                
                uRL_interp = (uRL_old + duRL).clone()
                uILC_interp = (uILC_old + duILC).clone()
                uRL_old = uRL_interp.clone()
                uILC_old = uILC_interp.clone()
                
                G = robot.getGravity(q=q)
                uMB = G
                uFB = torch.matmul(torch.diag(torch.tensor([kp, kp], dtype=torch.float32)), e_) \
                    + torch.matmul(torch.diag(torch.tensor([kv, kv], dtype=torch.float32)), de_)
                
                uTot = uMB + uFB + uRL_interp + uILC_interp + d[:, i, 0].reshape(2, 1)
                
                data.ctrl[:] = uTot.flatten().numpy()
                mujoco.mj_step(model, data, nstep=frame_skip)
                mujoco.mj_rnePostConstraint(model, data)
                
                t += dt_rob
        
        rmse = torch.sqrt(torch.mean(torch.stack(e_tmp) ** 2))
        rmse_list.append(rmse.item())
        
        if verbose:
            print(f"  Episode {ep + 1}/{n_episodes}: RMSE = {rmse.item():.6f}")
    
    return rmse_list[-1], rmse_list


def objective_function(x: np.ndarray, taskT: float, n_episodes: int, scaling: int, f_robot: int, qi: torch.Tensor, qf: torch.Tensor) -> float:
    kp, kv, le, lde, ldde, lddde = x

    if kp < 0 or kp > 5 or kv < 0 or kv > 5 or le < 0 or le > 0.1 or lde < 0 or lde > 0.1 or ldde < 0 or ldde > 0.1 or lddde < 0 or lddde > 0.1:
        return 10.0

    try:
        rmse, _ = run_simulation(
            kp=kp, kv=kv, le=le, lde=lde, ldde=ldde, lddde=lddde,
            taskT=taskT, n_episodes=n_episodes,
            scaling=scaling, f_robot=f_robot,
            qi_init=qi, qf=qf,
            verbose=False
        )
        return rmse
    except Exception as e:
        print(f"  Error: {e}")
        return 10.0


def grid_search_optimization(
    le_range: tuple[float, float, int],
    lde_range: tuple[float, float, int],
    ldde_range: tuple[float, float, int],
    taskT: float, n_episodes: int, scaling: int, f_robot: int,
    qi: torch.Tensor, qf: torch.Tensor
) -> dict:
    le_vals = np.linspace(*le_range)
    lde_vals = np.linspace(*lde_range)
    ldde_vals = np.linspace(*ldde_range)
    
    total_combos = len(le_vals) * len(lde_vals) * len(ldde_vals)
    print(f"\nGrid Search: {total_combos} combinations")
    
    results = []
    best_rmse = float('inf')
    best_params = None
    count = 0
    
    start_time = time.time()

    for le in le_vals:
        for lde in lde_vals:
            for ldde in ldde_vals:
                count += 1
                rmse, _ = run_simulation(
                    kp=0.4, kv=0.25, le=le, lde=lde, ldde=ldde, lddde=1e-07,
                    taskT=taskT, n_episodes=n_episodes,
                    scaling=scaling, f_robot=f_robot,
                    qi_init=qi, qf=qf, verbose=False
                )

                results.append({'le': le, 'lde': lde, 'ldde': ldde, 'rmse': rmse})

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params = {'kp': 0.4, 'kv': 0.25, 'le': le, 'lde': lde, 'ldde': ldde}
                    print(f"  [{count}/{total_combos}] New best: RMSE={rmse:.6f}")

    elapsed = time.time() - start_time
    
    print(f"\nGrid Search completed in {elapsed:.1f}s")
    print(f"Best RMSE: {best_rmse:.6f}")
    print(f"Best params: le={best_params['le']:.6f}, lde={best_params['lde']:.6f}, ldde={best_params['ldde']:.6f}")
    
    return {'best_params': best_params, 'best_rmse': best_rmse, 'results': results, 'elapsed': elapsed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='both')
    parser.add_argument('--max-iter', type=int, default=50)
    parser.add_argument('--workers', type=int, default=-1)
    args = parser.parse_args()
    
    taskT = 1.0
    n_episodes = 15
    scaling = 2
    f_robot = 100
    le = 2e-05
    lde = 4e-05
    ldde = 1e-06
    lddde = 1e-07
    kp = 0.4
    kv = 0.25

    baseline_gains = {'kp': kp, 'kv': kv, 'le': le, 'lde': lde, 'ldde': ldde, 'lddde': lddde}
    
    print("="*60)
    print("RILC-SEA GAIN OPTIMIZATION")
    print("="*60)
    print(f"\nBaseline gains: kp={kp}, kv={kv}, le={le}, lde={lde}, ldde={ldde}")
    
    qi = torch.tensor([[0.0], [0.0]])
    qf = QF
    
    print("\n" + "="*60)
    print("Testing baseline first...")
    print("="*60)
    baseline_rmse, baseline_list = run_simulation(
        **baseline_gains, taskT=taskT, n_episodes=n_episodes,
        scaling=scaling, f_robot=f_robot,
        qi_init=qi, qf=qf, verbose=True
    )
    print(f"\nBaseline final RMSE: {baseline_rmse:.6f}")
    
    best_overall_rmse = baseline_rmse
    best_overall_params = baseline_gains.copy()
    
    if args.method in ['grid', 'both']:
        print("\n" + "="*60)
        print("Starting Grid Search...")
        print("="*60)
        
        grid_results = grid_search_optimization(
            le_range=(0.001, 0.01, 10),
            lde_range=(0.0001, 0.001, 10),
            ldde_range=(0.000001, 0.0001, 10),
            taskT=taskT, n_episodes=n_episodes,
            scaling=scaling, f_robot=f_robot,
            qi=qi, qf=qf
        )
        
        if grid_results['best_rmse'] < best_overall_rmse:
            best_overall_rmse = grid_results['best_rmse']
            best_overall_params = grid_results['best_params']
    
    if args.method in ['de', 'both']:
        print("\n" + "="*60)
        print("Starting Differential Evolution (optimizing kp/kv lower)...")
        print("="*60)
        
        bounds = [
            (0.0, 1.0),
            (0.0, 1.0),
            (1e-06, 5e-05),
            (1e-06, 5e-05),
            (1e-07, 5e-06),
            (1e-08, 5e-07),
        ]
        
        print("\nSearch bounds:")
        print(f"  kp: [{bounds[0][0]:.4f}, {bounds[0][1]:.4f}]")
        print(f"  kv: [{bounds[1][0]:.4f}, {bounds[1][1]:.4f}]")
        
        obj_func = partial(objective_function, taskT=taskT, n_episodes=n_episodes, scaling=scaling, f_robot=f_robot, qi=qi, qf=qf)
        
        start_time = time.time()
        
        result = differential_evolution(
            obj_func,
            bounds,
            maxiter=args.max_iter,
            popsize=5,
            tol=1e-6,
            mutation=(0.5, 1.0),
            recombination=0.7,
            disp=True,
            workers=args.workers,
            updating='deferred' if args.workers != 1 else 'immediate',
            polish=True
        )
        
        de_rmse = result.fun
        de_params = {
            'kp': result.x[0],
            'kv': result.x[1],
            'le': result.x[2],
            'lde': result.x[3],
            'ldde': result.x[4],
            'lddde': result.x[5]
        }
        
        print(f"\nDE Best RMSE: {de_rmse:.6f}")
        print(f"DE Best params: le={de_params['le']:.6f}, lde={de_params['lde']:.6f}, ldde={de_params['ldde']:.6f}")
        
        if de_rmse < best_overall_rmse:
            best_overall_rmse = de_rmse
            best_overall_params = de_params
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"\nBest RMSE: {best_overall_rmse:.6f}")
    print(f"Best params: {best_overall_params}")
    
    print("\n" + "="*60)
    print("Testing optimized gains...")
    print("="*60)
    optimized_rmse, optimized_list = run_simulation(
        **best_overall_params, taskT=taskT, n_episodes=n_episodes,
        scaling=scaling, f_robot=f_robot,
        qi_init=qi, qf=qf, verbose=True
    )
    print(f"\nOptimized final RMSE: {optimized_rmse:.6f}")
    print(f"Improvement: {((baseline_rmse - optimized_rmse) / baseline_rmse * 100):.2f}%")
    
    import json
    with open('optimized_gains.json', 'w') as f:
        json.dump({
            'best_gains': best_overall_params,
            'best_rmse': float(best_overall_rmse),
            'baseline_rmse': float(baseline_rmse),
            'improvement_pct': float((baseline_rmse - optimized_rmse) / baseline_rmse * 100)
        }, f, indent=2)
    print(f"\nSaved to optimized_gains.json")


if __name__ == '__main__':
    main()