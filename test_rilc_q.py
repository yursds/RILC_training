from __init__             import *
from stable_baselines3    import PPO
from matplotlib           import pyplot as plt

import torch
import mujoco
import numpy as np

from classes.controllers.ilc        import ILC_base
from classes.controllers.pd         import PD_base
from classes.robots.manipulator_RR  import Sim_RR

from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from classes.environments.env_rlilc_elastic import Env_RILC as ENV
import functools


_HERE = os.path.dirname(os.path.abspath(__file__))
abs_path = os.path.join(_HERE, 'classes')
URDF_PATH = os.path.join(abs_path,'robots/robot_models/softleg_urdf/urdf/leg_constrained.urdf')
MESH_DIR  = os.path.join(abs_path,'robots/robot_models/softleg_urdf/meshes')
MJC_PATH  = os.path.join(abs_path,'robots/robot_models/softleg_urdf/mjc/scene_elastic.xml')

parent_str = "model"
dat_str = "rilc_sea_05"
step_str = "best_model/best_model.zip"

print(dat_str)
model_str = parent_str + "/" + dat_str + "/" + step_str

QF = torch.tensor([[2.0], [-1.0]])
FL_ILC = True
FL_RL = True
OBS_ILC = True
PLOT = True

if FL_ILC: OBS_ILC = True
def minjerk(qi:torch.Tensor,qf:torch.Tensor,duration:float,t:float) -> list[torch.Tensor,torch.Tensor,torch.Tensor]:

    delta_q = qi-qf
    q_new   = qi + delta_q * (15*(t/duration)**4 - 6*(t/duration)**5 - 10*(t/duration)**3)
    dq_new  = delta_q * (60*(t**3)/(duration**4) - 30*((t**4)/(duration**5)) - 30*(t**2)/(duration**3))
    ddq_new = delta_q * (180*(t**2)/(duration**4) - 120*((t**3)/(duration**5)) - 60*(t/(duration**3)))
    
    return q_new, dq_new, ddq_new

def angle_normalize(x:torch.Tensor) -> torch.Tensor:
    """ angle in range [-pi; pi]"""
    
    sx = torch.sin(x)
    cx = torch.cos(x)
    x = torch.atan2(sx,cx)
    return x

def resample_u(u_old:torch.Tensor, u_new:torch.Tensor, num_step:int) -> torch.Tensor:
    """ Resample action with linear interpolation. 
    
    Args:
        u_old (torch.Tensor): last action.
        u_new (torch.Tensor): new action.
        num_step (int): number of interpolation points.
    
    Returns:
        torch.Tensor: delta action to add for each interpolation points.
    """
    du_step = (u_new-u_old)/num_step
    
    return du_step



if __name__ == '__main__':

    visual = True
    
    # ++++++++++++++++++++++++  init  +++++++++++++++++++++++++++++++++++ 
    yaml_str        = parent_str+ "/" +dat_str+ "/" +'config.yaml'
    config:dict     = load_config(yaml_str)
    taskT: float    = config['taskT']
    n_ep_reset: int = config['n_ep_reset']*3
    scaling: int    = config['scaling']
    f_robot: int    = config['f_robot']
    le: float       = config['le']
    lde: float      = config['lde']
    ldde: float     = config['ldde']
    kp: float       = config['kp']
    kv: float       = config['kv']
    env_id: str     = config['env_id']
    
    model = mujoco.MjModel.from_xml_path(MJC_PATH)
    __actual_dt = model.opt.timestep
    frame_skip  = int((1/f_robot)/__actual_dt)
    
    noise_q_dev  = 1e-6
    noise_dq_dev = 2.5e-4
    njoint = 2
    # kp = .0
    # kv = .2
    # le = 0.08
    # lde = 0.04
    
    # init_observation
    f_policy       = int(f_robot / scaling)
    samples        = int(taskT*f_policy) + 1
    
    env = ENV(
        taskT=taskT, 
        f_robot=f_robot, 
        scaling=scaling, 
        le=le, 
        lde=lde, 
        ldde=ldde, 
        kp=kp, 
        kv=kv, 
        n_ep_reset=n_ep_reset)
    
    # ++++++++++++++++ init env vars ++++++++++++++++++++++++++++++++++++
    
    dt_pol = 1/f_policy
    dt_rob = 1/f_robot
    uRL_old_ep_ts  = torch.zeros(njoint,1,samples)
    uILC_old_ep_ts = torch.zeros(njoint,1,samples)
    uFB_old_ep_ts = torch.zeros(njoint,1,samples)
    e_old_ep_ts = torch.zeros(njoint,1,samples)
    de_old_ep_ts = torch.zeros(njoint,1,samples)
    
    des_traj_at = functools.partial(minjerk, qi=torch.tensor([[0.0], [0.0]]), qf=QF, duration=taskT)
        
    # +++++++++++++++++++ load pin ++++++++++++++++++++++++++++
    
    robot = Sim_RR(urdf_path=URDF_PATH, mesh_dir=MESH_DIR, ee_name='LH_ANKLE')
    tmp_q = des_traj_at(t=0.0)[0].clone()
    tmp_dq = des_traj_at(t=0.0)[1].clone()
    robot.setState(q0=tmp_q, dq0=tmp_dq, q=tmp_q, dq=tmp_dq)
    robot.u0 = robot.getGravity(robot.q0).clone()
    qi = robot.q0.clone()
    dqi = robot.dq0.clone()
    
    # MuJoCo has 4 DOFs (theta_hip, q_hip, theta_knee, q_knee)
    # Order: theta_hip, q_hip, theta_knee, q_knee
    qi_np = robot.q0.flatten().numpy()
    qpos_init = np.array([qi_np[0], qi_np[0], qi_np[1], qi_np[1]])  # [theta_hip, q_hip, theta_knee, q_knee]
    dqi_np = robot.dq0.flatten().numpy()
    qvel_init = np.array([dqi_np[0], dqi_np[0], dqi_np[1], dqi_np[1]])
    
    # +++++++++++++++++ load ctrl ++++++++++++++++++++++++++++

    # scale gains for policy frequency (f_policy = 50Hz)
    # Use gains already scaled (from optimize_gains output)
    le = torch.tensor(le * f_policy)
    lde = torch.tensor(lde * f_policy)
    ldde = torch.tensor(ldde * f_policy)
    lddde = torch.tensor(0.0 * f_policy)
    
    conILC = ILC_base(
        dimU=njoint, 
        samples=samples, 
        Le = le, 
        Lde = lde, 
        Ldde = ldde,
        Lddde = lddde)
    conILC.newEp()
    
    PD = PD_base(
        dimU    = robot._dim_u,
        kp      = kp,
        kv      = kv,
    )

    # ++++++++++++++++++ load policy ++++++++++++++++++++++++++++

    model_rl = PPO.load(model_str)
    
    # ++++++++++++++++++++++ reset ++++++++++++++++++++++++++++++++++++++++++
    
    # model.dof_frictionloss = model.dof_frictionloss*(1.2)
    # model.dof_armature     = model.dof_armature*(1.2)
    # model.body_mass        = model.body_mass*(1.2)
    # model.body_ipos        = model.body_ipos*(1.2)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    data.qpos = qpos_init
    data.qvel = qvel_init
    mujoco.mj_inverse(model, data)
    joint_forces = torch.from_numpy(data.qfrc_inverse).clone()
    data.ctrl[:] = joint_forces[:2]  # only motor-side (theta) joints
    mujoco.mj_forward(model, data)
    
    conILC.resetAll()
    
    # ++++++++++++ renderer +++++++++++++++++++++++++++++
    mujoco_renderer = MujocoRenderer(model, data, None, 800, 600)
    
    # init vars
    q     = qi.clone().view(2,1)
    dq    = dqi.clone().view(2,1)
    ddq   = torch.zeros(2,1)
    dq_old  = dq.clone()
    theta   = torch.zeros(2,1)
    dtheta  = torch.zeros(2,1)
    
    d = torch.rand(2,samples,n_ep_reset)*0.5*0
        
    e_list   = []
    de_list  = []
    dde_list = []
    uRL_list = []
    uMB_list = []
    uILC_list= []
    uFB_list = []
    
    # q = link-side positions
    q_list   = []
    dq_list  = []
    ddq_list = []
    
    # theta = motor-side positions
    theta_list   = []
    dtheta_list = []
    
    r_list = torch.zeros(2,samples)
    dr_list = torch.zeros(2,samples)
    ddr_list = torch.zeros(2,samples)

    for i in range(samples):
        t = i*dt_pol
        r, dr, ddr = des_traj_at(t=t)

        r_list[:,i] = r.flatten()
        dr_list[:,i] = dr.flatten()
        ddr_list[:,i] = ddr.flatten()
    
    for ep in range(n_ep_reset):
        
        t = 0.0
        dde_old = torch.zeros(2, 1)
        
        if conILC.episodes == 0:
            conILC.newEp()
        else:
            conILC.stepILC()
        
        uFB      = torch.zeros(njoint,1)
        uRL      = torch.zeros(njoint,1)
        uRL_old  = torch.zeros(njoint,1)
        duRL     = torch.zeros(njoint,1)
        uILC     = torch.zeros(njoint,1)
        uILC_old = torch.zeros(njoint,1)
        duILC    = torch.zeros(njoint,1)
        
        mujoco.mj_resetData(model, data)
        data.qpos = qpos_init
        data.qvel = qvel_init
        mujoco.mj_inverse(model, data)
        joint_forces = torch.from_numpy(data.qfrc_inverse).clone()
        data.ctrl[:] = robot.getGravity(q=qi).flatten()
        mujoco.mj_forward(model, data)
        # dq_old = dq.clone()
        dq_old = dq.clone()
        
        e_tmp       = []
        de_tmp      = []
        dde_tmp     = []
        
        # link-side
        q_tmp       = []
        dq_tmp      = []
        ddq_tmp     = []
        
        # motor-side
        theta_tmp   = []
        dtheta_tmp = []
        
        uRL_tmp     = []
        uMB_tmp     = []
        uILC_tmp    = []
        uFB_tmp     = []
        
        for i in range(samples):
            r_, dr_, ddr_ = des_traj_at(t=t)
            
            # LINK-SIDE (q) - q_hip sensor measures deflection from equilibrium
            deflection = torch.zeros(2,1)
            dq_raw = torch.zeros(2,1)
            theta_sample = torch.zeros(2,1)
            theta_sample[0]  = torch.from_numpy(data.sensor("theta_hip").data)
            theta_sample[1]  = torch.from_numpy(data.sensor("theta_knee").data)
            deflection[0]  = torch.from_numpy(data.sensor("q_hip").data)
            deflection[1]  = torch.from_numpy(data.sensor("q_knee").data)
            dq_raw[0] = torch.from_numpy(data.sensor("dq_hip").data)
            dq_raw[1] = torch.from_numpy(data.sensor("dq_knee").data)
            q    = theta_sample + deflection
            dq   = torch.zeros(2,1)
            q    += noise_q_dev * torch.randn(2,1)
            dq   += noise_dq_dev * torch.randn(2,1)
            ddq   = (dq-dq_old)*f_robot
        
            e_    = r_ - q
            e_    = angle_normalize(e_)
            de_   = dr_ - dq
            dde_  = ddr_ - ddq
            # 3rd derivative of error
            if scaling > 1:
                dt = dt_pol
                ddde_ = (dde_ - dde_old) * f_robot
                dde_old = dde_.clone()
            else:
                ddde_ = torch.zeros(2, 1)
            
            e_tmp.append(e_.flatten().clone())
            de_tmp.append(de_.flatten().clone())
            dde_tmp.append(dde_.flatten().clone())

            # q_tmp.append(q.flatten().clone())
            # dq_tmp.append(dq.flatten().clone())
            # ddq_tmp.append(ddq.flatten().clone())
            
            q_old = robot.q.clone()
            robot.setState(q=q, dq=dq)
            
            u_delta = uFB+uILC
            conILC.updateMemError(e_=e_,de_=de_,dde_=dde_,ddde_=ddde_)
            conILC.updateMemInput(u_delta)
            
            if conILC.episodes != 0 and FL_ILC:
                uilc = conILC.getControl()
                uILC = uilc
            else:
                uILC = torch.zeros(2,1)
            
            duRL  = resample_u(u_old=uRL_old, u_new=uRL, num_step=scaling)
            duILC = resample_u(u_old=uILC_old, u_new=uILC,  num_step=scaling)
            uRL_interp  = uRL_old
            uILC_interp = uILC_old
            
            t_pol = t + dt_pol
            
            if t_pol <= taskT:
                r_f, dr_f, ddr_f = des_traj_at(t=t_pol)

                uRL_old_ep  = uRL_old_ep_ts[:,:,i]
                uILC_old_ep = uILC_old_ep_ts[:,:,i]
                uFB_old_ep = uFB_old_ep_ts[:,:,i]
                e_old_ep = e_old_ep_ts[:,:,i]
                de_old_ep = de_old_ep_ts[:,:,i]
                
                if OBS_ILC:
                    obs  = torch.cat([
                        q.flatten(), dq.flatten(), \
                        r_f.flatten(), dr_f.flatten(), \
                        uRL_old.flatten(), uILC.flatten(), uILC_old_ep.flatten(),
                        e_old_ep.flatten(), de_old_ep.flatten()], dim=0)
                else:
                    obs  = torch.cat([
                        q.flatten(), dq.flatten(), \
                        r_f.flatten(), dr_f.flatten(), \
                        uRL_old.flatten(), uILC.flatten()*0, uILC_old_ep.flatten()*0,
                        e_old_ep.flatten(), de_old_ep.flatten()], dim=0)
                
                obs_np = env.normalize_obs(obs)

                url, _ = model_rl.predict(obs_np, deterministic=True)
                uRL    = env.rescale_action(url).view(-1,1) if FL_RL else torch.zeros(2,1)
                uRL_old_ep_ts[:, :, i] = uRL.clone()
                uILC_old_ep_ts[:, :, i] = uILC.clone()
                uFB_old_ep_ts[:, :, i] = uFB.clone()
                e_old_ep_ts[:, :, i] = e_.clone()
                de_old_ep_ts[:, :, i] = de_.clone()
            
            for _ in range(scaling):
                
                r_, dr_, ddr_ = des_traj_at(t=t)

                # LINK-SIDE (q) - q_hip sensor measures deflection from equilibrium, not absolute position
                # Actual link position = theta - deflection
                deflection[0] = torch.from_numpy(data.sensor("q_hip").data)
                deflection[1] = torch.from_numpy(data.sensor("q_knee").data)
                dq_raw[0]     = torch.from_numpy(data.sensor("dq_hip").data)
                dq_raw[1]     = torch.from_numpy(data.sensor("dq_knee").data)
                
                # Compute actual link position from motor + deflection
                q    = theta + deflection
                dq   = dtheta + dq_raw
                q    += noise_q_dev * torch.randn(2,1)
                dq   += noise_dq_dev * torch.randn(2,1)
                ddq   = (dq-dq_old)*f_robot
                dq_old = dq.clone()
                
                # MOTOR-SIDE (theta) - NEW for SEA
                theta[0]  = torch.from_numpy(data.sensor("theta_hip").data)
                theta[1]  = torch.from_numpy(data.sensor("theta_knee").data)
                dtheta[0] = torch.from_numpy(data.sensor("dtheta_hip").data)
                dtheta[1] = torch.from_numpy(data.sensor("dtheta_knee").data)
                
                e_    = r_ - q
                e_    = angle_normalize(e_)
                de_   = dr_ - dq
                dde_  = ddr_ - ddq
                
                uRL_interp  = (uRL_old + duRL).clone()
                uILC_interp = (uILC_old + duILC).clone()
                uRL_old     = uRL_interp.clone()
                uILC_old    = uILC_interp.clone()
                
                G      = robot.getGravity(q=q)
                uMB = G
                uFB = torch.matmul(torch.diag(torch.tensor([kp, kp])),e_) \
                    + torch.matmul(torch.diag(torch.tensor([kv, kv])),de_)
                
                uTot = uMB + uFB + uRL_interp + uILC_interp + d[:,i,ep].reshape(2,1)
                
                data.ctrl[:2] = uTot.flatten().numpy()
                mujoco.mj_step(model, data, nstep=frame_skip)
                mujoco.mj_rnePostConstraint(model, data)
                
                # # append sensor data
                # theta_tmp.append(theta.flatten().clone())
                # dtheta_tmp.append(dtheta.flatten().clone())
                
                t += dt_rob

            if visual:
                mujoco_renderer.render("human")


            # append sensor data
            theta_tmp.append(theta.flatten().clone())
            dtheta_tmp.append(dtheta.flatten().clone())
            # update partial logging
            q_tmp.append(q.flatten().clone())
            dq_tmp.append(dq.flatten().clone())
            ddq_tmp.append(ddq.flatten().clone())
            uMB_tmp.append(uMB.flatten().clone())
            uILC_tmp.append(uILC_interp.flatten().clone())
            uFB_tmp.append(uFB.flatten().clone())
            uRL_tmp.append(uRL_interp.flatten().clone())

        e_list.append(e_tmp)
        de_list.append(de_tmp)
        dde_list.append(dde_tmp)
        q_list.append(q_tmp)
        dq_list.append(dq_tmp)
        ddq_list.append(ddq_tmp)
        theta_list.append(theta_tmp)
        dtheta_list.append(dtheta_tmp)
        uRL_list.append(uRL_tmp)
        uMB_list.append(uMB_tmp)
        uILC_list.append(uILC_tmp)
        uFB_list.append(uFB_tmp)
         
    for i in range(len(e_list)):
        rmse_list = torch.sqrt(torch.mean(torch.stack(e_list[i])**2))
        print(f"rilc MSE of episode: {i}", rmse_list)
    for i in range(len(e_list)):
        rmse_uFB = torch.sqrt(torch.mean(torch.stack(uFB_list[i])**2))
        print(f"rilc MS_uFB of episode: {i}", rmse_uFB)
    for i in range(len(e_list)):
        rmse_uILC = torch.sqrt(torch.mean(torch.stack(uILC_list[i])**2))
        print(f"rilc MS_uILC of episode: {i}", rmse_uILC)
    
    # +++++++++++++++++++++++++++ SAME PLOTS AS test_rilc.py +++++++++++++++++++++++++++
    
    for i in [0,n_ep_reset-1]:
        plt.figure(figsize=(8, 8))
        plt.subplot(2,2,1)
        plt.plot(torch.stack(e_list[i]).T[0,:], label="sim e1")
        plt.plot(torch.stack(e_list[i]).T[1,:], label="sim e2")
        plt.xlabel("Time steps")
        plt.ylabel("Error [$rad$]")
        plt.title(f"Error")
        plt.grid()
        plt.subplot(2,2,2)
        plt.plot(torch.stack(de_list[i]).T[0,:], label="sim de1")
        plt.plot(torch.stack(de_list[i]).T[1,:], label="sim de2")
        plt.xlabel("Time steps")
        plt.ylabel("Dot error [$rad/s$]")
        plt.title(f"Dot Error")
        plt.grid()
        plt.subplot(2,2,3)
        plt.plot(torch.stack(q_list[i]).T[0,:], label="sim q1")
        plt.plot(torch.stack(q_list[i]).T[1,:], label="sim q2")
        plt.plot(r_list[0,:], label="ref q1")
        plt.plot(r_list[1,:], label="ref q2")
        plt.xlabel("Time steps")
        plt.ylabel("Angle [$rad$]")
        plt.legend()
        plt.title(f"Link-side Joints' Angle in episode {i+1}")
        plt.grid()
        plt.subplot(2,2,4)
        plt.plot(torch.stack(dq_list[i]).T[0,:], label="sim dq1")
        plt.plot(torch.stack(dq_list[i]).T[1,:], label="sim dq2")
        plt.plot(dr_list[0,:], label="ref dq1")
        plt.plot(dr_list[1,:], label="ref dq2")
        plt.xlabel("Time steps")
        plt.ylabel("Dot Angle [$rad/s$]")
        plt.title(f"Link-side Joints' Dot Angle")
        plt.grid()
        plt.legend()

        plt.suptitle(f"ILC in episode {i+1}")
        plt.tight_layout()
    
    # +++++++++++++++++++++++++++ CONTROL SIGNAL PLOTS +++++++++++++++++++++++++++
    
    for i in [0,n_ep_reset-1]:
        plt.figure(figsize=(15, 3))
        
        plt.subplot(1, 6, 1)
        plt.plot(torch.stack(uMB_list[i])[:,0]+torch.stack(uRL_list[i])[:,0]+torch.stack(uILC_list[i])[:,0]+torch.stack(uFB_list[i])[:,0])
        plt.plot(torch.stack(uMB_list[i])[:,1]+torch.stack(uRL_list[i])[:,1]+torch.stack(uILC_list[i])[:,1]+torch.stack(uFB_list[i])[:,1])
        plt.title("uTOT")
        plt.xlabel("steps")
        plt.grid()
        
        plt.subplot(1, 6, 2)
        plt.plot(torch.stack(uRL_list[i])[:,0]+torch.stack(uILC_list[i])[:,0])
        plt.plot(torch.stack(uRL_list[i])[:,1]+torch.stack(uILC_list[i])[:,1])
        plt.title("uRL+uILC")
        plt.xlabel("steps")
        plt.grid()
        
        plt.subplot(1, 6, 3)
        plt.plot(torch.stack(uMB_list[i])[:,0])
        plt.plot(torch.stack(uMB_list[i])[:,1])
        plt.title("uMB")
        plt.xlabel("steps")
        plt.grid()
        
        plt.subplot(1, 6, 4)
        plt.plot(torch.stack(uILC_list[i])[:,0])
        plt.plot(torch.stack(uILC_list[i])[:,1])
        plt.title("uILC")
        plt.xlabel("steps")
        plt.grid()
        
        plt.subplot(1, 6, 5)
        plt.plot(torch.stack(uFB_list[i])[:,0])
        plt.plot(torch.stack(uFB_list[i])[:,1])
        plt.title("uFB")
        plt.xlabel("steps")
        plt.grid()
        
        plt.subplot(1, 6, 6)
        plt.plot(torch.stack(uRL_list[i])[:,0])
        plt.plot(torch.stack(uRL_list[i])[:,1])
        plt.title("uRL")
        plt.xlabel("steps")
        plt.grid()
        
        plt.suptitle(f"ILC Episode {i+1}")
        plt.tight_layout()
    
    # +++++++++++++++++++++++++++ SEA-SPECIFIC PLOTS +++++++++++++++++++++++++++
    
    if PLOT:
        # Plots with theta and q
        for i in [0,n_ep_reset-1]:
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2,3,1)
            plt.plot(torch.stack(e_list[i]).T[0,:], label="e1")
            plt.plot(torch.stack(e_list[i]).T[1,:], label="e2")
            plt.xlabel("Time steps")
            plt.ylabel("Error [rad]")
            plt.title(f"Error (link-side)")
            plt.grid()
            plt.legend()
            
            plt.subplot(2,3,2)
            plt.plot(torch.stack(q_list[i]).T[0,:], label="q1")
            plt.plot(torch.stack(q_list[i]).T[1,:], label="q2")
            plt.plot(r_list[0,:], label="ref1", linestyle='--')
            plt.plot(r_list[1,:], label="ref2", linestyle='--')
            plt.xlabel("Time steps")
            plt.ylabel("Angle [rad]")
            plt.title(f"Link-side positions")
            plt.grid()
            plt.legend()
            
            plt.subplot(2,3,3)
            plt.plot(torch.stack(theta_list[i]).T[0,:], label="theta1")
            plt.plot(torch.stack(theta_list[i]).T[1,:], label="theta2")
            plt.plot(r_list[0,:], label="ref1", linestyle='--')
            plt.plot(r_list[1,:], label="ref2", linestyle='--')
            plt.xlabel("Time steps")
            plt.ylabel("Angle [rad]")
            plt.title(f"Motor-side positions (theta)")
            plt.grid()
            plt.legend()
            
            plt.subplot(2,3,4)
            diff_theta_q1 = torch.stack(theta_list[i]).T[0,:] - torch.stack(q_list[i]).T[0,:]
            diff_theta_q2 = torch.stack(theta_list[i]).T[1,:] - torch.stack(q_list[i]).T[1,:]
            plt.plot(diff_theta_q1, label="theta1 - q1")
            plt.plot(diff_theta_q2, label="theta2 - q2")
            plt.xlabel("Time steps")
            plt.ylabel("Deflection [rad]")
            plt.title(f"SEA deflection (theta - q)")
            plt.grid()
            plt.legend()
            
            plt.subplot(2,3,5)
            plt.plot(torch.stack(dq_list[i]).T[0,:], label="dq1")
            plt.plot(torch.stack(dq_list[i]).T[1,:], label="dq2")
            plt.xlabel("Time steps")
            plt.ylabel("Velocity [rad/s]")
            plt.title(f"Link-side velocity")
            plt.grid()
            plt.legend()
            
            plt.subplot(2,3,6)
            plt.plot(torch.stack(dtheta_list[i]).T[0,:], label="dtheta1")
            plt.plot(torch.stack(dtheta_list[i]).T[1,:], label="dtheta2")
            plt.xlabel("Time steps")
            plt.ylabel("Velocity [rad/s]")
            plt.title(f"Motor-side velocity")
            plt.grid()
            plt.legend()
            
            plt.suptitle(f"SEA Model - Episode {i+1}")
            plt.tight_layout()
    
        plt.show()
    else:
        print("Plots generated but not shown.")
    
    if visual:
        mujoco_renderer.close()