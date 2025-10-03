from __init__             import *
from stable_baselines3    import PPO
from matplotlib           import pyplot as plt

import torch
import mujoco
import pickle

from classes.controllers.ilc        import ILC_base
from classes.controllers.pd         import PD_base
from classes.robots.manipulator_RR  import Sim_RR

from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from classes.environments.env_rlilc_mjc import Env_RILC as ENV
import functools


abs_path  = os.path.join(os.path.dirname((os.path.abspath(__file__))),'classes') # classes_folder
URDF_PATH = os.path.join(abs_path,'robots/robot_models/softleg_urdf/urdf/leg_constrained.urdf')
MJC_PATH  = os.path.join(abs_path,'robots/robot_models/softleg_urdf/mjc/scene_test.xml')
EP_OVERRIDE = 5

parent_str = "model"
dat_str = "rilc_constrained" # "rilc" "rl_classic"
step_str = "best_model/best_model.zip"
last_filename = dat_str + f"_{EP_OVERRIDE}ep_random_mismatch_mass_5"

print(dat_str)
model_str = parent_str + "/" + dat_str + "/" + step_str

FL_ILC = True
FL_RL = True
OBS_ILC = False

if FL_ILC: OBS_ILC = True
# TRAJ = "minjerk" # "minjerk" "lissajous"

RANDOM_QF = True
NUM_SAMPLES = 3000
SEED = 42
torch.manual_seed(SEED)

if RANDOM_QF:
    flat_qf_envs = (torch.rand(2, NUM_SAMPLES)*2-1)*torch.pi
else:
    filepath = os.path.join(os.path.dirname((os.path.abspath(__file__))), 'statistic_data/data_mem/qf_visit_training_rilc.pkl')
    with open(filepath, 'rb') as f:
        list_ts = pickle.load(f)
    flat_qf_envs = torch.cat([torch.cat(env_qf,1) for env_qf in list_ts],1)

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

    visual = False
    
    # ++++++++++++++++++++++++  init  +++++++++++++++++++++++++++++++++++ 
    yaml_str        = parent_str+ "/" +dat_str+ "/" +'config.yaml'
    config:dict     = load_config(yaml_str)
    taskT: float    = config['taskT']
    n_ep_reset: int = EP_OVERRIDE # config['n_ep_reset']
    scaling: int    = config['scaling']
    f_robot: int    = config['f_robot']
    le: float       = config['le']
    lde: float      = config['lde']
    ldde: float     = config['ldde']
    kp: float       = config['kp']
    kv: float       = config['kv']
    env_id: str     = config['env_id']
    
    Ye_list   = []
    Yde_list  = []
    Ydde_list = []
    
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
    
    # +++++++++++++++++ init traj ++++++++++++++++++++++++++++
    
    des_traj_at = functools.partial(minjerk, qi = torch.tensor([[0.0], [0.0]]), duration = taskT)
    
    # +++++++++++++++++++ load pin ++++++++++++++++++++++++++++
    
    robot = Sim_RR(urdf_path=URDF_PATH, ee_name='LH_ANKLE')
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
    model.body_mass        = model.body_mass*(1.2)
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
    
    for n_qf in range(flat_qf_envs.shape[1]):
        
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
        
        uRL_old_ep_ts  = torch.zeros(njoint,1,samples)
        uILC_old_ep_ts = torch.zeros(njoint,1,samples)
        uFB_old_ep_ts = torch.zeros(njoint,1,samples)
        e_old_ep_ts = torch.zeros(njoint,1,samples)
        de_old_ep_ts = torch.zeros(njoint,1,samples)
        
        if n_qf%100 == 0:
            print(f"complete at {n_qf/flat_qf_envs.shape[1]*100}%")
        
        conILC.resetAll()
        
        qf = flat_qf_envs[:,n_qf].view(2,1).clone()
        
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
                r_, dr_, ddr_ = des_traj_at(t=t, qf=qf)
                
                q[0]   = torch.from_numpy(data.sensor("q_hip").data)
                q[1]   = torch.from_numpy(data.sensor("q_knee").data)
                dq[0]  = torch.from_numpy(data.sensor("dq_hip").data)
                dq[1]  = torch.from_numpy(data.sensor("dq_knee").data)
                
                q    += noise_q_dev * torch.randn(2,1)
                dq   += noise_dq_dev * torch.randn(2,1)
                ddq   = (dq-dq_old)*f_robot
                
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
                u_delta = uFB+uILC
                # update ERROR memory of ILC
                conILC.updateMemError(e_=e_,de_=de_,dde_=dde_)
                # update INPUT memory of ILC
                conILC.updateMemInput(u_delta)
                
                # get new control of ILC
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
                    r_f, dr_f, ddr_f = des_traj_at(t=t_pol, qf= qf)
                    
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
                    r_, dr_, ddr_ = des_traj_at(t=t, qf= qf)
                    
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
        
        Ye_list.append(e_list)
        Yde_list.append(de_list)
        Ydde_list.append(dde_list)
        
        if visual:
            mujoco_renderer.close()
    
    with open(f'qf_{last_filename}.pkl', 'wb') as f:
        pickle.dump(flat_qf_envs, f)

    with open(f'e_{last_filename}.pkl', 'wb') as f:
        pickle.dump(Ye_list, f)

    # with open(f'de_{last_filename}.pkl', 'wb') as f:
    #     pickle.dump(Yde_list, f)

    # with open(f'dde_{last_filename}.pkl', 'wb') as f:
    #     pickle.dump(Ydde_list, f)