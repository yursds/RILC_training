from __init__             import *
from stable_baselines3    import PPO
from matplotlib           import pyplot as plt

import torch
import mujoco

from classes.controllers.ilc        import ILC_base
from classes.controllers.pd         import PD_base
from classes.robots.manipulator_RR  import Sim_RR

from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from classes.environments.env_rlilc_mjc import Env_RILC as ENV
import functools



abs_path  = os.path.join(os.path.dirname((os.path.abspath(__file__))),'classes') # classes_folder
URDF_PATH = os.path.join(abs_path,'robots/robot_models/softleg_urdf/urdf/leg_constrained.urdf')
MJC_PATH  = os.path.join(abs_path,'robots/robot_models/softleg_urdf/mjc/scene_test.xml')

parent_str = "model"
dat_str = "rilc_constrained" # "rilc" "rl_classic" "rilc_constrained"
step_str = "best_model/best_model.zip"

print(dat_str)
model_str = parent_str + "/" + dat_str + "/" + step_str

QF = torch.tensor([[torch.pi/3], [torch.pi/3]])
# QF = torch.tensor([[-2.232461929321289], [-3.069495677947998]])
QF = torch.tensor([[2.4], [-1.4]])
FL_ILC = True
FL_RL = True
OBS_ILC = False

if FL_ILC: OBS_ILC = True
TRAJ = "minjerk" # "minjerk" "lissajous"


def custom_lissajous_at(t:float, complete_traj: torch.Tensor, dt:float) -> list[torch.Tensor, torch.Tensor, torch.Tensor]:

    idx = int(t // dt)
    if idx >= complete_traj.shape[1]:
        idx = complete_traj.shape[1] - 1
        print("Warning: index out of bounds in custom_lissajous_at")    
    des_traj = complete_traj[:, idx]

    q_des   = des_traj[0:2].view(2,1)
    dq_des  = des_traj[2:4].view(2,1)
    ddq_des = des_traj[4:6].view(2,1)
    
    return q_des, dq_des, ddq_des

def load_trajectory(filename: str = 'complete_traj.pt') -> torch.Tensor:

    if os.path.exists(filename):
        complete_traj = torch.load(filename, weights_only=True)
        # print(f"Trajectory loaded from {filename}")
    return complete_traj

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
    n_ep_reset: int = config['n_ep_reset']
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
    
    # +++++++++++++++++ init traj ++++++++++++++++++++++++++++
    
    if TRAJ == "minjerk":
        des_traj_at = functools.partial(minjerk, qi = torch.tensor([[0.0], [0.0]]), qf = QF, duration = taskT)
    elif TRAJ == "lissajous":
        traj = load_trajectory(filename=os.path.join(abs_path, "references", "traj.pt"))
        des_traj_at = functools.partial(custom_lissajous_at, complete_traj=traj, dt=dt_rob)
    elif TRAJ == "circle":
        traj = load_trajectory(filename=os.path.join(abs_path, "references", "traj_circle.pt"))
        des_traj_at = functools.partial(custom_lissajous_at, complete_traj=traj, dt=dt_rob)
    else:
        assert False, "No valid trajectory selected"
        
    # +++++++++++++++++++ load pin ++++++++++++++++++++++++++++
    
    robot = Sim_RR(urdf_path=URDF_PATH, ee_name='LH_ANKLE')
    tmp_q = des_traj_at(t=0.0)[0].clone()
    tmp_dq = des_traj_at(t=0.0)[1].clone()
    robot.setState(q0=tmp_q, dq0=tmp_dq, q=tmp_q, dq=tmp_dq)
    robot.u0 = robot.getGravity(robot.q0).clone()
    qi = robot.q0.clone()
    dqi = robot.dq0.clone()
    qpos_init = robot.q0.flatten().numpy().copy()
    qvel_init = robot.dq0.flatten().numpy().copy()
    
    # +++++++++++++++++ load ctrl ++++++++++++++++++++++++++++
    
    ldde = torch.tensor(ldde*f_policy)
    lde  = torch.tensor(lde*f_policy)
    le   = torch.tensor(le*f_policy)
    
    conILC = ILC_base(
        dimU=njoint, 
        samples=samples, 
        Le = le, 
        Lde = lde, 
        Ldde = ldde)
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

    # qpos_init = data.qpos # !!!!
    # qvel_init = data.qvel 
    data.qpos = qpos_init
    data.qvel = qvel_init
    mujoco.mj_inverse(model, data)
    joint_forces = torch.from_numpy(data.qfrc_inverse).clone()
    data.ctrl[:] = joint_forces #robot.getGravity(q=qi).flatten()
    mujoco.mj_forward(model, data)
    
    conILC.resetAll()
    
    # ++++++++++++ renderer +++++++++++++++++++++++++++++
    mujoco_renderer = MujocoRenderer(model, data, None, 800, 600)
    
    # init vars
    q     = qi.clone().view(2,1)
    dq    = dqi.clone().view(2,1)
    ddq    = torch.zeros(2,1)
    dq_old  = dq.clone()
    
    d = torch.rand(2,samples,n_ep_reset)*0.5*0
        
    e_list   = []
    de_list  = []
    dde_list = []
    uRL_list = []
    uMB_list = []
    uILC_list= []
    uFB_list = []
    q_list   = []
    dq_list  = []
    ddq_list = []
    
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
        dq_old = torch.as_tensor(data.qvel).view(2,1).clone()
        
        e_tmp       = []
        de_tmp      = []
        dde_tmp     = []
        q_tmp       = []
        dq_tmp      = []
        ddq_tmp     = []
        uRL_tmp     = []
        uMB_tmp     = []
        uILC_tmp    = []
        uFB_tmp     = []
        
        for i in range(samples):
            # r_, dr_, ddr_ = minjerk(qi = qi, qf = qf, duration = taskT, t = t)
            r_, dr_, ddr_ = des_traj_at(t=t)
            
            q[0]   = torch.from_numpy(data.sensor("q_hip").data)
            q[1]   = torch.from_numpy(data.sensor("q_knee").data)
            dq[0]  = torch.from_numpy(data.sensor("dq_hip").data)
            dq[1]  = torch.from_numpy(data.sensor("dq_knee").data)
            q    += noise_q_dev * torch.randn(2,1)
            dq   += noise_dq_dev * torch.randn(2,1)
            ddq   = (dq-dq_old)*f_robot
            
            # print(f'q:{q.flatten()}')
            # print(f'dq:{dq.flatten()}')
            # print(f'ddq:{ddq.flatten()}')
            
            e_    = r_ - q
            e_    = angle_normalize(e_)
            de_   = dr_ - dq
            dde_  = ddr_ - ddq
            
            e_tmp.append(e_.flatten().clone())
            de_tmp.append(de_.flatten().clone())
            dde_tmp.append(dde_.flatten().clone())
            
            q_old = robot.q.clone()
            robot.setState(q=q, dq=dq)
            
            # Update useful memory of ILC
            # iM       = robot.getInvMass(q=q_old)
            # u_delta  = torch.matmul(iM, uFB+uILC)
            u_delta = uFB+uILC
            # update ERROR memory of ILC
            conILC.updateMemError(e_=e_,de_=de_,dde_=dde_)
            # update INPUT memory of ILC
            conILC.updateMemInput(u_delta)
            
            # get new control of ILC
            if conILC.episodes != 0 and FL_ILC:
                uilc = conILC.getControl()
                # M    = robot.getMass(q=q)
                # uILC = torch.matmul(M,uilc)
                uILC = uilc
            else:
                uILC = torch.zeros(2,1)
            
            duRL  = resample_u(u_old=uRL_old, u_new=uRL, num_step=scaling)
            duILC = resample_u(u_old=uILC_old, u_new=uILC,  num_step=scaling)
            uRL_interp  = uRL_old
            uILC_interp = uILC_old
            
            t_pol = t + dt_pol
            
            if t_pol <= taskT:
                # r_f, dr_f, ddr_f = minjerk(qi = qi, qf = qf, duration = taskT, t = t_pol)
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
                # uRL = (torch.rand(2,1)*2-1)*2
                uRL_old_ep_ts[:, :, i] = uRL.clone()
                uILC_old_ep_ts[:, :, i] = uILC.clone()
                uFB_old_ep_ts[:, :, i] = uFB.clone()
                e_old_ep_ts[:, :, i] = e_.clone()
                de_old_ep_ts[:, :, i] = de_.clone()
            
            for _ in range(scaling):
                
                # r_, dr_, ddr_ = minjerk(qi = qi, qf = qf, duration = taskT, t = t)
                r_, dr_, ddr_ = des_traj_at(t=t)

                # get state
                q[0]  = torch.from_numpy(data.sensor("q_hip").data)
                q[1]  = torch.from_numpy(data.sensor("q_knee").data)
                dq[0] = torch.from_numpy(data.sensor("dq_hip").data)
                dq[1] = torch.from_numpy(data.sensor("dq_knee").data)
                q    += noise_q_dev * torch.randn(2,1)
                dq   += noise_dq_dev * torch.randn(2,1)
                ddq   = (dq-dq_old)*f_robot
                dq_old = dq.clone()
                
                e_    = r_ - q
                e_    = angle_normalize(e_)
                de_   = dr_ - dq
                dde_  = ddr_ - ddq
                
                # update interpolation of law rate commands
                uRL_interp  = (uRL_old + duRL).clone()
                uILC_interp = (uILC_old + duILC).clone()
                uRL_old     = uRL_interp.clone()
                uILC_old    = uILC_interp.clone()
                
                # ---------------- MB control - compensate g and simplify dynamics ----------------------#
                G      = robot.getGravity(q=q)
                uMB = G
                # ------------------------------ PD control ---------------------------------------------#
                uFB = torch.matmul(torch.diag(torch.tensor([kp, kp])),e_) \
                    + torch.matmul(torch.diag(torch.tensor([kv, kv])),de_)
                
                uTot = uMB + uFB + uRL_interp + uILC_interp + d[:,i,ep].reshape(2,1)
                
                data.ctrl[:] = uTot.flatten().numpy()
                mujoco.mj_step(model, data, nstep=frame_skip)
                mujoco.mj_rnePostConstraint(model, data)
                
                t += dt_rob
                
            if visual:
                mujoco_renderer.render("human")
            
            # update partial logging
            q_tmp.append(q.flatten().clone())
            dq_tmp.append(dq.flatten().clone())
            ddq_tmp.append(ddq.flatten().clone())
            uMB_tmp.append(uMB.flatten().clone())
            uILC_tmp.append(uILC_interp.flatten().clone())
            uFB_tmp.append(uFB.flatten().clone())
            uRL_tmp.append(uRL_interp.flatten().clone())
        
        # update complete logging
        e_list.append(e_tmp)
        de_list.append(de_tmp)
        dde_list.append(dde_tmp)
        q_list.append(q_tmp)
        dq_list.append(dq_tmp)
        ddq_list.append(ddq_tmp)
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
    
    for i in [0,n_ep_reset-1]:
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
        plt.suptitle(f"ILC in  episode {i+1}")
        plt.tight_layout()
    
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
        
    for i in [0, n_ep_reset-1]:
        plt.figure(figsize=(6,6))
        plt.title("Check Trajectory")
        plt.plot([robot.getForwKinEE(r)[0][0] for r in r_list.T], [robot.getForwKinEE(r)[0][1] for r in r_list.T], label="ref traj")
        plt.plot([robot.getForwKinEE(q)[0][0] for q in q_list[i]], [robot.getForwKinEE(q)[0][1] for q in q_list[i]], label=f"traj_sim ep {i+1}")
        plt.legend()
        plt.xlabel("q1 [rad]")
        plt.ylabel("q2 [rad]")
        plt.axis('equal')
        plt.grid()
    plt.show()
    
    if visual:
        mujoco_renderer.close()
    