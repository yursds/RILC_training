from __init__             import *
from stable_baselines3    import PPO
from matplotlib           import pyplot as plt

import torch
import mujoco

from classes.controllers.ilc        import ILC_base
from classes.robots.manipulator_RR  import Sim_RR
from classes.references.classic_ref import RefInvKin as GenRef

from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from classes.environments.env_rlilc_mjc import Env_RILC as ENV

abs_path  = os.path.join(os.path.dirname((os.path.abspath(__file__))),'classes') # classes_folder
URDF_PATH = os.path.join(abs_path,'robots/robot_models/softleg_urdf/urdf/leg_constrained.urdf')
MJC_PATH  = os.path.join(abs_path,'robots/robot_models/softleg_urdf/mjc/scene_test.xml')

parent_str = "model"
dat_str = "rilc" # "rilc" "rl_classic"
step_str = "best_model.zip"

print(dat_str)

model_str = parent_str + "/" + dat_str + "/" + step_str


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
    fl_noILC: bool  = config['fl_noILC']
    env_id: str     = config['env_id']
    
    njoint = 2
    
    # init_observation
    f_policy       = int(f_robot / scaling)
    samples        = int(taskT*f_policy) + 1
    
    dt_pol = 1/f_policy
    dt_rob = 1/f_robot
    
    uRL_old_ep_ts  = torch.zeros(njoint,1,samples)
    uILC_old_ep_ts = torch.zeros(njoint,1,samples)
    uFB_old_ep_ts = torch.zeros(njoint,1,samples)
    e_old_ep_ts = torch.zeros(njoint,1,samples)
    de_old_ep_ts = torch.zeros(njoint,1,samples)
    
    ldde = torch.tensor(ldde*f_policy)
    lde  = torch.tensor(lde*f_policy)
    le   = torch.tensor(le*f_policy)
    
    model_rl = PPO.load(model_str)
    env     = ENV()
    
    robot = Sim_RR(urdf_path=URDF_PATH, ee_name='LH_ANKLE')
    model = mujoco.MjModel.from_xml_path(MJC_PATH)
    data = mujoco.MjData(model)
    qpos_init = data.qpos 
    qvel_init = data.qvel 
    __actual_dt = model.opt.timestep
    frame_skip  = int((1/f_robot)/__actual_dt)
    mujoco_renderer = MujocoRenderer(model, data, None, 800, 600)
    
    conILC = ILC_base(
        dimU=njoint, 
        samples=samples, 
        Le = le, 
        Lde = lde, 
        Ldde = ldde)
    
    conILC.newEp()
    qi = torch.zeros(njoint,1)
    qf = torch.tensor([[-torch.pi/2, -torch.pi/2]]).T
    #qf = torch.tensor([[torch.pi/6, -torch.pi/6]]).T

    ref_gen = GenRef(
        robot    = robot,
        duration = taskT,
        pf       = qf,
        stayT    = 0.0,
        dt       = dt_pol,
        )
    ref_env      = ref_gen.getRef()
    
    q_init = torch.zeros(2,1)
    robot.setState(q=q_init)
    
    # init vars
    q     = torch.zeros(2,1)
    dq    = torch.zeros(2,1)
    ddq    = torch.zeros(2,1)
    dq_old  = torch.zeros(2,1)
    
    noise_q_dev  = 1e-6
    noise_dq_dev = 2.5e-4
    
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
        dq_old = torch.zeros(2,1)
        
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
            
            r_, dr_, ddr_ = minjerk(qi = qi, qf = qf, duration = taskT, t = t)
            
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
            
            q_old = robot.q.clone()
            robot.setState(q=q)
            
            # Update useful memory of ILC
            iM       = robot.getInvMass(q=q_old)
            u_delta  = torch.matmul(iM, uRL+uFB+uILC)
            # update ERROR memory of ILC
            conILC.updateMemError(e_=e_,de_=de_,dde_=dde_)
            # update INPUT memory of ILC
            conILC.updateMemInput(u_delta)
            
            # get new control of ILC
            if conILC.episodes != 0:
                M    = robot.getMass(q=q)
                uilc = conILC.getControl()
                uILC = torch.matmul(M,uilc)
            else:
                uILC = torch.zeros(2,1)
            
            duRL  = env.resample_u(u_old=uRL_old, u_new=uRL, num_step=scaling)
            duILC = env.resample_u(u_old=uILC_old, u_new=uILC,  num_step=scaling)
            uRL_interp  = uRL_old
            uILC_interp = uILC_old
            
            t_pol = t + dt_pol
            
            if t_pol <= taskT:
                r_f, dr_f, ddr_f = minjerk(qi = qi, qf = qf, duration = taskT, t = t_pol)
                
                uRL_old_ep  = uRL_old_ep_ts[:,:,i]
                uILC_old_ep = uILC_old_ep_ts[:,:,i]
                uFB_old_ep = uFB_old_ep_ts[:,:,i]
                e_old_ep = e_old_ep_ts[:,:,i]
                de_old_ep = de_old_ep_ts[:,:,i]
                
                if env_id != "Env_RILC":
                    obs  = torch.cat([
                        q.flatten(), dq.flatten(), \
                        r_f.flatten(), dr_f.flatten(), \
                        uRL_old.flatten(), uILC.flatten()*0, uILC_old_ep.flatten()*0,
                        e_old_ep.flatten(), de_old_ep.flatten()], dim=0)
                else:
                    obs  = torch.cat([
                        q.flatten(), dq.flatten(), \
                        r_f.flatten(), dr_f.flatten(), \
                        uRL_old.flatten(), uILC.flatten(), uILC_old_ep.flatten(),
                        e_old_ep.flatten(), de_old_ep.flatten()], dim=0)
                
                obs_np = env.normalize_obs(obs)

                url, _ = model_rl.predict(obs_np, deterministic=True)
                uRL    = env.rescale_action(url).view(-1,1)
                # uRL = (torch.rand(2,1)*2-1)*2
                uRL_old_ep_ts[:, :, i] = uRL.clone()
                uILC_old_ep_ts[:, :, i] = uILC.clone()
                uFB_old_ep_ts[:, :, i] = uFB.clone()
                e_old_ep_ts[:, :, i] = e_.clone()
                de_old_ep_ts[:, :, i] = de_.clone()
            
            d = torch.rand(2,1)*0.5
            
            for _ in range(scaling):
                
                r_, dr_, ddr_ = minjerk(qi = qi, qf = qf, duration = taskT, t = t)
                
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
                
                # total control to rob! - update dynamic
                if env_id != "Env_RILC":
                    uTot = uMB + uFB + uRL_interp +d*0.5
                    uILC_interp *= 0
                else:
                    uTot = uMB + uFB + uRL_interp + uILC_interp + d*0.5
                
                data.ctrl[:] = uTot.flatten().numpy()
                mujoco.mj_step(model, data, nstep=frame_skip)
                mujoco.mj_rnePostConstraint(model, data)
                
                t += dt_rob
                
            if visual:
                mujoco_renderer.render("human")
            
            # update partial logging
            e_tmp.append(e_.flatten().clone())
            de_tmp.append(de_.flatten().clone())
            dde_tmp.append(dde_.flatten().clone())
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
    
    # for i in [0,n_ep_reset-1]:
    #     plt.figure(figsize=(8, 8))
    #     plt.subplot(2,3,1)
    #     plt.plot(torch.stack(e_list[i]).T[0,:], label="sim e1")
    #     plt.plot(torch.stack(e_list[i]).T[1,:], label="sim e2")
    #     plt.xlabel("Time steps")
    #     plt.ylabel("Error [$rad$]")
    #     plt.title(f"Error")
    #     plt.grid()
    #     plt.subplot(2,3,2)
    #     plt.plot(torch.stack(de_list[i]).T[0,:], label="sim de1")
    #     plt.plot(torch.stack(de_list[i]).T[1,:], label="sim de2")
    #     plt.xlabel("Time steps")
    #     plt.ylabel("Dot error [$rad/s$]")
    #     plt.title(f"Dot Error")
    #     plt.grid()    
    #     plt.subplot(2,3,3)
    #     plt.plot(torch.stack(dde_list[i]).T[0,:], label="sim dde1")
    #     plt.plot(torch.stack(dde_list[i]).T[1,:], label="sim dde2")
    #     plt.xlabel("Time steps")
    #     plt.ylabel("DDot error [$rad/s^2$]")
    #     plt.title(f"DDot Error  ")
    #     plt.grid()    
    #     plt.subplot(2,3,4)
    #     plt.plot(torch.stack(q_list[i]).T[0,:], label="sim q1")
    #     plt.plot(torch.stack(q_list[i]).T[1,:], label="sim q2")
    #     plt.plot(ref_env[0,0,:], label="ref q1")
    #     plt.plot(ref_env[1,0,:], label="ref q2")
    #     plt.xlabel("Time steps")
    #     plt.ylabel("Angle [$rad$]")
    #     plt.legend()
    #     plt.title(f"Joints' Angle in episode  {i+1}")
    #     plt.grid()
    #     plt.subplot(2,3,5)
    #     plt.plot(torch.stack(dq_list[i]).T[0,:], label="sim dq1")
    #     plt.plot(torch.stack(dq_list[i]).T[1,:], label="sim dq2")
    #     plt.plot(ref_env[0,1,:], label="ref dq1")
    #     plt.plot(ref_env[1,1,:], label="ref dq2")
    #     plt.xlabel("Time steps")
    #     plt.ylabel("Dot Angle [$rad/s$]")
    #     plt.title(f"Joints' Dot Angle")
    #     plt.grid()
    #     plt.legend()
    #     plt.subplot(2,3,6)
    #     plt.plot(torch.stack(ddq_list[i]).T[0,:], label="sim ddq1")
    #     plt.plot(torch.stack(ddq_list[i]).T[1,:], label="sim ddq2")
    #     plt.plot(ref_env[0,2,:], label="ref ddq1")
    #     plt.plot(ref_env[1,2,:], label="ref ddq2")
    #     plt.xlabel("Time steps")
    #     plt.ylabel("DDot Angle [$rad/s^2$]")
    #     plt.title(f"Joints' DDot Angle")
    #     plt.legend()
    #     plt.grid()
    #     plt.suptitle(f"ILC in  episode {i+1}")
    #     plt.tight_layout()
    
    # for i in [0,n_ep_reset-1]:
    #     plt.figure(figsize=(15, 3))
        
    #     plt.subplot(1, 6, 1)
    #     plt.plot(torch.stack(uMB_list[i])[:,0]+torch.stack(uRL_list[i])[:,0]+torch.stack(uILC_list[i])[:,0]+torch.stack(uFB_list[i])[:,0])
    #     plt.plot(torch.stack(uMB_list[i])[:,1]+torch.stack(uRL_list[i])[:,1]+torch.stack(uILC_list[i])[:,1]+torch.stack(uFB_list[i])[:,1])
    #     plt.title("uTOT")
    #     plt.xlabel("steps")
    #     plt.grid()
        
    #     plt.subplot(1, 6, 2)
    #     plt.plot(torch.stack(uRL_list[i])[:,0]+torch.stack(uILC_list[i])[:,0])
    #     plt.plot(torch.stack(uRL_list[i])[:,1]+torch.stack(uILC_list[i])[:,1])
    #     plt.title("uRL+uILC")
    #     plt.xlabel("steps")
    #     plt.grid()
        
    #     plt.subplot(1, 6, 3)
    #     plt.plot(torch.stack(uMB_list[i])[:,0])
    #     plt.plot(torch.stack(uMB_list[i])[:,1])
    #     plt.title("uMB")
    #     plt.xlabel("steps")
    #     plt.grid()
        
    #     plt.subplot(1, 6, 4)
    #     plt.plot(torch.stack(uILC_list[i])[:,0])
    #     plt.plot(torch.stack(uILC_list[i])[:,1])
    #     plt.title("uILC")
    #     plt.xlabel("steps")
    #     plt.grid()
        
    #     plt.subplot(1, 6, 5)
    #     plt.plot(torch.stack(uFB_list[i])[:,0])
    #     plt.plot(torch.stack(uFB_list[i])[:,1])
    #     plt.title("uFB")
    #     plt.xlabel("steps")
    #     plt.grid()
        
    #     plt.subplot(1, 6, 6)
    #     plt.plot(torch.stack(uRL_list[i])[:,0])
    #     plt.plot(torch.stack(uRL_list[i])[:,1])
    #     plt.title("uRL")
    #     plt.xlabel("steps")
    #     plt.grid()
        
    #     plt.suptitle(f"ILC Episode {i+1}")
    #     plt.tight_layout()
    # plt.show()
    
    if visual:
        mujoco_renderer.close()
    