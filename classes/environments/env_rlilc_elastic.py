__credits__ = ["Yuri De Santis"]

import __init__

# NUMPY & TORCH & GYMNASIUM
import torch
import copy
import mujoco
import os
import numpy                as np

import gymnasium.spaces     as spaces
from gymnasium              import utils
from gymnasium.envs.mujoco  import MujocoEnv

from typing import Optional

# MY CLASSES: ROBOT
from classes.robots.manipulator_RR  import Sim_RR
from classes.controllers.ilc        import ILC_base
from classes.controllers.pd         import PD_base

abs_path  = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # classes_folder
URDF_PATH = os.path.join(abs_path,'robots/robot_models/softleg_urdf/urdf/leg_constrained.urdf')
MESH_DIR  = os.path.join(abs_path,'robots/robot_models/softleg_urdf/meshes')
MJC_PATH  = os.path.join(abs_path,'robots/robot_models/softleg_urdf/mjc/leg_constrained_elastic.xml')


def load_trajectory(filename: str = 'complete_traj.pt') -> torch.Tensor:

    if os.path.exists(filename):
        complete_traj = torch.load(filename, weights_only=True)
        # print(f"Trajectory loaded from {filename}")
    else:
        raise FileNotFoundError(f"Trajectory file not found: {filename}")
    return complete_traj

TRAJ_LISS = load_trajectory(filename=os.path.join(abs_path, "references", "traj_test.pt"))




class Env_RILC(MujocoEnv, utils.EzPickle):
    """ Environment for RL+ILC.
    
        Args:
            taskT (float, optional): task duration. Defaults to 1.0.
            f_robot (float, optional): frequency of update of robot dynamics (action dynamic). Defaults to 500. (Different from computation timestep of dynamic (used for computation stability): Mujoco set default timestep to 0.002, you can change in xml option or in self.model.opt.timestep)
            scaling (int, optional): scaling factor of frequency of update of policy w.r.t. robot frequency. f_robot = f_policy*scaling. Defaults to 10. (choose a sclaling_factor > 1 for smoother behaviour)            
            stayT (float, optional): time to mantain in last position. Trajectory duration = taskT-stayT. Defaults to 0.0.
            n_ep_reset (int, optional): consecutive episode with same trajectory. Defaults to 5.
            le (float, optional): gain of learning in ILC law for error. Defaults to 0.0002.
            lde (float, optional): gain of learning in ILC law for dot error. Defaults to 0.0004.
            ldde (float, optional): gain of learning in ILC law for ddot error. Defaults to 0.0008.
            kp (float, optional): gain of PD for error. Defaults to 0.04.
            kv (float, optional): gain of PD for dot error. Defaults to 0.025.
            relative_pos (bool, optional): flag to consider the trajectory position respect global position. Defaults to False.
            dtype (torch.dtype, optional): type of variables. Defaults to torch.float32.
            seed (int, optional): seed use to define trajectories. Defaults to None.
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ]
    }
    
    def __init__(
        self,
        taskT : float       = 1.0,
        f_robot : float     = 500,
        scaling : int       = 10,
        stayT : float       = 0.0,
        n_ep_reset : int    = 5,  
        le : float          = 0.02,
        lde : float         = 0.04,
        ldde : float        = 0.08,
        kp : float          = 0.4,
        kv : float          = 0.25,
        threshold : float   = 1e-3,
        seed : int          = None,
        dtype : torch.dtype = torch.float32,
        **kwargs,
        ):
        
        """ Args:
            taskT (float, optional): task duration. Defaults to 1.0.
            f_robot (float, optional): frequency of update of robot dynamics (action dynamic). Defaults to 500. (Different from computation  timestep of dynamic: Mujoco set default timestep to 0.002, you can change in xml option or in self.model.opt.timestep)
            scaling (int, optional): scaling factor of frequency of update of policy w.r.t. robot frequency. f_robot = f_policy*scaling. Defaults to 10. (choose a sclaling_factor > 1 for smoother behaviour)            
            stayT (float, optional): time to mantain in last position. Trajectory duration = taskT-stayT. Defaults to 0.0.
            n_ep_reset (int, optional): consecutive episode with same trajectory. Defaults to 5.
            le (float, optional): gain of learning in ILC law for error. Defaults to 0.0002.
            lde (float, optional): gain of learning in ILC law for dot error. Defaults to 0.0004.
            ldde (float, optional): gain of learning in ILC law for ddot error. Defaults to 0.0008.
            kp (float, optional): gain of PD for error. Defaults to 0.04.
            kv (float, optional): gain of PD for dot error. Defaults to 0.025.
            threshold (float, optional): rmse threshold to stop updating ILC. Defaults to 1e-3.
            relative_pos (bool, optional): flag to consider the trajectory position respect global position. Defaults to False.
            dtype (torch.dtype, optional): type of variables. Defaults to torch.float32.
            seed (int, optional): seed use to define trajectories. Defaults to None.
        """
        
        # set seeding
        if seed is not None:
            torch.manual_seed(seed)
        
        # it is necessary? it seems that is used for mujoco wrappers
        utils.EzPickle.__init__(self, **kwargs)
        
        # init Mujoco model
        # NOTE -> ACTHUNG: not use same attributes' names for custom attributes of env!!!
        # NOTE: frame_skip is used to transform computation dynamic timestep into action dynamic timestep (there is a for loop!!!).
        # to extract some info of the model a MJC tmp_model is build at first and, then, build MJC env
        self._init_model = mujoco.MjModel.from_xml_path(MJC_PATH)
        __actual_dt = self._init_model.opt.timestep
        frame_skip  = int((1/f_robot)/__actual_dt)
        if frame_skip == 0:
            raise('AAAAAAAAAAAAH frame_skip is set to zeros, not possible, check timestep of the xml model')
        
        # mujoco env compute internally observation_space and action_space and is a mandatory set, but those are not normalize and use infos from xml.
        # to custom env spec you can override self.observation_space and self.action_space after the istance of MujocoEnv.
        # set an observation_space
        tmp_observation_space = spaces.Box(low=-torch.ones(1,).numpy(), high=torch.ones(1,).numpy())
        MujocoEnv.__init__(
            self,
            model_path        = MJC_PATH,
            frame_skip        = frame_skip,
            observation_space = tmp_observation_space,
            **kwargs
        )
        
        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }
    
        # set class attributes
        self.f_policy     = int(f_robot/scaling)
        self.f_robot      = f_robot
        self.scaling      = scaling
        self.taskT        = taskT
        self.stayT        = stayT
        self.dtype        = dtype
        self.n_ep_reset   = n_ep_reset
        self.le           = le
        self.lde          = lde
        self.ldde         = ldde
        self.kp           = kp
        self.kv           = kv
        self.threshold    = threshold
        
        self.samples = int(self.taskT*self.f_policy) + 1
        self.noise_q_dev  = 1e-6
        self.noise_dq_dev = 2.5e-4
        self.friction_pert = 0.2 # 20%
        self.des_traj_at = self.minjerk
        
        self.q_range_task = torch.tensor(torch.pi)
        self.delta_q_task = torch.zeros(2,1)
        self.qf_list = []
        
        # -------------------------------------- PIN_ROBOT -------------------------------------- #
        self._load_pin_robot(URDF_PATH)
        # ------------------------------------- CONTROLLERS ------------------------------------- #
        self._load_controllers()
        # -------------------------------------- DEFINE SPEC ------------------------------------ #
        self._build_spec()
        # --------------------------------------- INIT ENV -------------------------------------- #
        self._init_env_vars()
        # ----------------------------------------- RESET --------------------------------------- #
        self._reset_vars()
        # -------------------------------------- USEFUL DICT ------------------------------------ #
        self._build_dicts()
        
    def minjerk(self, t:float) -> list[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Compute position, velocity, acceleration of joints of trajectory_des trajectory w.r.t. time t.
        
        Args:
            t (float): reference time.
        
        Returns:
            list[torch.Tensor, torch.Tensor, torch.Tensor]: position, velocity, acceleration of joints. """
        
        duration = self.taskT
        delta_q = self.qi - self.qf
        q_new   = self.qi + delta_q * (15*(t/duration)**4 - 6*(t/duration)**5 - 10*(t/duration)**3)
        dq_new  = delta_q * (60*(t**3)/(duration**4) - 30*((t**4)/(duration**5)) - 30*(t**2)/(duration**3))
        ddq_new = delta_q * (180*(t**2)/(duration**4) - 120*((t**3)/(duration**5)) - 60*(t/(duration**3)))
        
        return q_new, dq_new, ddq_new
    
    def _load_pin_robot(self, urdf_path):
        """ define robot
        """
        # ---------------------------------------- ROBOT ---------------------------------------- #
        pin_robot = Sim_RR(urdf_path=urdf_path, mesh_dir=MESH_DIR, ee_name='LH_ANKLE')
        # attribute robot
        self.pin_rob = pin_robot
        self.qi = self.pin_rob.q0
    
    def _load_controllers(self):
        """ load controllers, ILC and PD """
        rob  = self.pin_rob
        le   = self.le
        lde  = self.lde
        ldde = self.ldde
        kp   = self.kp
        kv   = self.kv
        fs_p = self.f_policy
        threshold = self.threshold
        
        # ILC - NOTE: update with policy rate
        le_ts   = torch.tensor(le * fs_p)
        lde_ts  = torch.tensor(lde * fs_p)
        ldde_ts = torch.tensor(ldde * fs_p)
        
        self.ILC  = ILC_base(
            dimU = rob._dim_q,
            samples = self.samples,
            Le = le_ts,
            Lde = lde_ts,
            Ldde = ldde_ts,
            threshold = threshold,
            dtype = self.dtype,)
        
        # PD - NOTE: update with robot rate
        self.PD = PD_base(
            dimU    = rob._dim_u,
            kp      = kp,
            kv      = kv,
        )
    
    def _build_spec(self):
        """ define specs of environment """
        rob = self.pin_rob
        # get position, velocity and action limits
        q_max    = rob._q_M
        q_min    = rob._q_m
        dq_max   = torch.tensor([[50, 50]]).T
        dq_min   = -dq_max
        u_max    = torch.tensor([[5, 5]]).T
        u_min    = -u_max
        
        # ------------------- define limits - NOT SCALED (TORCH) -------------------------- #
        # action
        self.lowLimA    = -torch.tensor([[2, 2]]).flatten()
        self.highLimA   = torch.tensor([[2, 2]]).flatten()
        # self.lowLimA    = -torch.tensor([[5, 5]]).flatten()
        # self.highLimA   = torch.tensor([[5, 5]]).flatten()
        # observation -> [q dq]  [ref dref] [uRLold] [uILC uILColdep] [eoldep deoldep]
        self.lowLimO    = torch.cat([q_min, dq_min,
                                    q_min, dq_min,
                                    u_min, u_min, u_min,
                                    q_min, dq_min,], dim=0).flatten()
        self.highLimO   = torch.cat([q_max, dq_max,
                                    q_max, dq_max,
                                    u_max, u_max, u_max,
                                    q_max, dq_max,], dim=0).flatten()
        
        # ---------- action.space and observation.space - SCALED to [-1,1] (NUMPY) -------- #
        self.action_space = spaces.Box(
            low=-torch.ones_like(self.lowLimA).numpy(),
            high=torch.ones_like(self.highLimA).numpy())
        self.observation_space = spaces.Box(
            low=-torch.ones_like(self.lowLimO).numpy(),
            high=torch.ones_like(self.highLimO).numpy())
    
    def _init_env_vars(self):
        """ define variables of initialised environment """
        
        rob = self.pin_rob
        
        self.dt_pol  = 1/self.f_policy
        self.dt_rob  = 1/self.f_robot
        self.t       = 0.0
        
        self.env_PD0   = torch.zeros_like(rob.u0)
        self.env_RL0   = torch.zeros_like(rob.u0)
        self.env_ILC0  = torch.zeros_like(rob.u0)
        self.env_MB0   = rob.u0.clone()
        self.env_utot0 = self.env_PD0 + self.env_RL0 + self.env_ILC0 + self.env_MB0
        
        self.env_step  = 0
        self.env_epis  = 0
        
        self.epEold0    = torch.zeros(self.pin_rob._dim_q, 1, self.samples)
        self.epDEold0   = torch.zeros(self.pin_rob._dim_q, 1, self.samples)
        self.epUilcold0 = torch.zeros(self.pin_rob._dim_u, 1, self.samples) # [uRLold, uILCold]
        self.epEold     = self.epEold0.clone()
        self.epDEold    = self.epDEold0.clone()
        self.epUilcold  = self.epUilcold0.clone()
        self.epE        = self.epEold.clone()
        self.epDE       = self.epDEold.clone()
        self.epUilc     = self.epUilcold.clone()
        
        # save past step control
        self._uRLold  = self.env_RL0.clone()
        self._uILCold = self.env_ILC0.clone()
        self._uFBold  = self.env_PD0.clone()
        
        rand_number = torch.rand(2,1)*2-1
        self.qf = self.q_range_task*rand_number + self.delta_q_task

    def _build_dicts(self):
        """ define useful dict to analyse environment
        """
        scaling = self.scaling
        rob = self.pin_rob
        
        self.rmse0     = torch.sqrt(torch.mean(self.epEold0**2))
        self.rmse      = self.rmse0.clone()
        
        # default dict
        self.dict0 = {
            "rw_dict" : { 
                "rw_e"     : torch.zeros(1,)[0],
                "rw_de"    : torch.zeros(1,)[0],
                "rw_law"   : torch.zeros(1,)[0],
                #"rw_url"   : torch.zeros(1,),
                #"rw_ufb"   : torch.zeros(1,),
                "rw_du"    : torch.zeros(1,)[0],
                "rw_done"  : torch.zeros(1,)[0],
                "rw_tot"   : torch.zeros(1,)[0],
            },
            "env" : {  
                "uRL"  : torch.zeros_like(self.pin_rob.u0).flatten(),
                "uILC" : torch.zeros_like(self.pin_rob.u0).flatten(),
            },
            "robot" : {
                "e"    : torch.zeros(self.pin_rob.q0.size(0), scaling),
                "de"   : torch.zeros(self.pin_rob.q0.size(0), scaling),
                "dde"  : torch.zeros(self.pin_rob.q0.size(0), scaling),
                "uMB"  : rob.u0.clone().expand(-1,scaling),
                "uFB"  : torch.zeros(self.pin_rob.u0.size(0), scaling),
                "uRL"  : torch.zeros(self.pin_rob.u0.size(0), scaling),
                "uILC" : torch.zeros(self.pin_rob.u0.size(0), scaling),
            },
            "additional" : {
                "rmse" : self.rmse0.item(),
                #"consecutive_ep" : 0,
            },
        }
        # dict to update
        self.dict = copy.deepcopy(self.dict0)
    
    def _reset_vars(self):
        """ define reset paramenters. """
        # termination constrains
        self.max_steps  = self.samples
    
    @staticmethod
    def resample_u(u_old:torch.Tensor, u_new:torch.Tensor, num_step:int) -> torch.Tensor:
        """ resample action with linear interpolation. 

        Args:
            u_old (torch.Tensor): last action.
            u_new (torch.Tensor): new action.
            num_step (int): number of interpolation points.

        Returns:
            torch.Tensor: delta action to add for each interpolation points.
        """
        du_step = (u_new-u_old)/num_step
        
        return du_step

    def _updateRobot(self, uRL:torch.Tensor, uILC:torch.Tensor) -> list[list[torch.Tensor],list[torch.Tensor], bool]:
        """ Compute dynamic of robot.

        Args:
            ref_i (torch.ref): reference [q,dq]
            uRL (torch.Tensor): control RL.
            uILC (torch.Tensor): control ILc.
        
        Returns:
            list[list[torch.Tensor],list[torch.Tensor], bool]: [uMB, uFB], [e_,de_,dde_], flag_trunc
        """
        
        # compute reference
        r_, dr_, ddr_ = self.des_traj_at(t = self.t)

        # get state
        q, dq = self._get_state()
        ddq   = (dq-self.dq_old)*self.f_robot
        # update dq_old
        self.dq_old = dq.clone()
        
        # compute error
        e_  = r_ - q
        e_  = self.pin_rob.angle_normalize(e_)
        de_ = dr_ - dq
        dde_= ddr_ - ddq
        
        # ---------------- MB control - compensate g and simplify dynamics ----------------------#
        # get useful matrix of model-based control
        G = self.pin_rob.getGravity(q=q)
        uMB = G
        # ------------------------------ PD control ---------------------------------------------#
        uFB = self.PD.getControl(e_,de_)
        
        # total control to rob! - update dynamic
        uTot = uMB + uFB + uRL + uILC
        self.do_simulation(uTot.flatten().numpy(), self.frame_skip)
        
        self.t += self.dt_rob
        
        return [uMB, uFB], [e_,de_,dde_]
    
    def step(self, u:np.ndarray):
        """ Update Environment.

        Args:
            u (np.ndarray): normalized decoupled action.

        Returns:
            observation -> [q dq ddq] [ref dref ddref] [uRL] [uILCold uRLold]
            reward (float): reward
            done (bool): True if episode reach max steps
            trucated (bool): True the episode is truncated
            info: dict of useful info usable in callback
        """
        
        # rescale action
        u_rl  = self.rescale_action(u)
        # tranform in column vector
        uRL   = u_rl.view(self.pin_rob._dim_u,-1)
        
        uILC = self._ilc_routine()
        
        # save dict variables. NOTE: PREDICTION!!!
        self.dict["env"]["uRL"]  = uRL.flatten()
        self.dict["env"]["uILC"] = uILC.flatten()
        
        # interpolation of uRL and uILC
        duRL        = self.resample_u(u_old=self._uRLold, u_new=uRL, num_step=self.scaling)
        duILC       = self.resample_u(u_old=self._uILCold, u_new=uILC, num_step=self.scaling)
        uRL_interp  = self._uRLold
        uILC_interp = self._uILCold
        
        for ii in range(self.scaling):
            
            [uMB, uFB], [e_,de_,dde_] = self._updateRobot(uRL=uRL_interp, uILC=uILC_interp)
            
            self.dict["robot"]["uMB"][:,ii:ii+1] = uMB.clone()
            self.dict["robot"]["uFB"][:,ii:ii+1] = uFB.clone()
            self.dict["robot"]["uRL"][:,ii:ii+1] = uRL_interp.clone()
            self.dict["robot"]["uILC"][:,ii:ii+1]= uILC_interp.clone()
            self.dict["robot"]["e"][:,ii:ii+1]   = e_.clone()
            self.dict["robot"]["de"][:,ii:ii+1]  = de_.clone()
            self.dict["robot"]["dde"][:,ii:ii+1] = dde_.clone()
            
            uRL_interp  = uRL_interp + duRL
            uILC_interp = uILC_interp + duILC
        
        # use real commands used for robot
        # compute outputs - obs, reward, done
        obs     = self._get_obs(ctrlNew=[uMB, uFB, uRL_interp, uILC_interp]).numpy()
        reward  = self._compute_reward(ctrlNew=[uMB, uFB, uRL_interp, uILC_interp], errorNew=[e_,de_,dde_])
        done    = bool((self.t + 2*self.dt_pol) >= self.taskT)
        
        # store last useful inputs
        self._uFBold    = uFB.clone()
        self._uRLold    = uRL_interp.clone()
        self._uILCold   = uILC_interp.clone()
        
        self.env_step += 1
        
        if done:
            # compute rmse
            rmse = torch.sqrt(torch.mean(self.epE**2))
            
            self.dict["additional"]["rmse"]  = rmse.item()
            # additional reward
            self.rmse = rmse
            self.env_epis += 1
            rew_done       = 1/(10*rmse+1) 
            
            # additional reward for last error
            rew_final_error = -10*(torch.sum(e_**2))
            rew_done       += rew_final_error
        else:
            rew_done       = torch.tensor(0.0)
        
        reward = reward + rew_done.item()
        
        self.dict["rw_dict"]["rw_done"]  = rew_done
        self.dict["rw_dict"]["rw_tot"]   = torch.tensor(reward)
        info = self.dict
        
        #print("prima di uscire", obs.flatten())
        return obs, reward, done, False, info
    
    def _ilc_routine(self) -> torch.Tensor:
        
        # compute reference for ILC
        r_, dr_, ddr_ = self.des_traj_at(t = self.t)

        # get state
        # self.data.joint("HIP").qpos
        # self.data.sensordata
        q, dq = self._get_state()
        ddq   = (dq-self.dq_old)*self.f_robot
        
        # compute env error
        e_    = r_ - q
        e_    = self.pin_rob.angle_normalize(e_)
        de_   = dr_ - dq
        dde_  = ddr_ - ddq
        # Update useful memory of ILC
        # iM       = self.pin_rob.getInvMass(q=q)
        # u_delta = torch.matmul(iM, self._uILCold+self._uFBold)
        u_delta = self._uILCold+self._uFBold
        # update ERROR memory of ILC
        self.ILC.updateMemError(e_=e_,de_=de_,dde_=dde_)
        # update INPUT memory of ILC
        self.ILC.updateMemInput(u_delta)
        
        # env variables
        count = self.env_step
        self.epE[:,:,count] = e_.clone()
        self.epDE[:,:,count] = de_.clone()
        
        # get new control of ILC
        if self.env_epis != 0:
            # M = self.pin_rob.getMass(q=q)
            uilc = self.ILC.getControl()
            # uILC = torch.matmul(M,uilc)
            uILC = uilc
        else:
            uILC = torch.zeros_like(self.pin_rob.u0)
        
        return uILC
        
    def reset(self, seed: Optional[int] = None, force_reset_ILC = False, options=None):
        
        # reset ONLY DATA OF MJC
        mujoco.mj_resetData(self.model, self.data)
        self.set_state(self.init_qpos , self.init_qvel)
        
        self.env_step = 0
        self.t = 0.0
        
        pd0     = self.env_PD0
        rl0     = self.env_RL0
        mb0     = self.env_MB0
        ilc0    = self.env_ILC0
        
        self._uRLold    = rl0
        self._uILCold   = ilc0
        self._uFBold    = pd0
        
        self.pin_rob._uold = self.pin_rob.u0.clone()
        self.dq_old = torch.zeros(self.pin_rob._dim_q,1)
        
        # # reset episode under 10% probability
        # reset_mem = torch.rand(1)
        # prob = reset_mem**(self.n_ep_reset-self.env_epis)
        # if prob <= 0.1:
        #     self.env_epis = 0
        
        if self.env_epis == 0 or self.env_epis % self.n_ep_reset == 0 :
            
            self.ILC.resetAll()
            self.ILC.newEp()
            self.env_epis = 0 
            self.epUilcold   = self.epUilcold0.clone()
            self.epEold   = self.epEold0.clone()
            self.epDEold  = self.epDEold0.clone()
            self.rmse     = self.rmse0.clone()
            
            self.model = copy.deepcopy(self._init_model)
            self.model.dof_frictionloss = self.model.dof_frictionloss*(1+(torch.rand_like(torch.from_numpy(self.model.dof_frictionloss))*2-1)*self.friction_pert).numpy()
            
            # generate new reference
            rand_number = torch.rand(2,1)*2-1
            self.qf = self.q_range_task*rand_number + self.delta_q_task
            self.qf_list.append(self.qf.clone())
            #self.qf = torch.tensor([[torch.pi/2,torch.pi/2]]).T
            q, dq = self._get_state()
            self.qi = q.clone()
            
            # perturb Kp and Kd
            Kp = torch.diag(self.kp*(torch.rand(self.pin_rob._dim_u,)+1))
            Kd = torch.diag(self.kv*(torch.rand(self.pin_rob._dim_u,)+1))
            self.PD.setParams(Kp=Kp, Kv=Kd)
        
        obs = self._get_obs(ctrlNew=[mb0, pd0, rl0, ilc0]).numpy()
        info = self.dict0
        
        if self.env_epis >= 1:
            self.ILC.stepILC()
        
        #self.env_step += 1
        return obs, info

    def _get_obs(self, ctrlNew:list[torch.Tensor]) -> torch.Tensor:
        """ Get observation.

        Args:
            ctrlNew (list[torch.Tensor], optional): [uMB, uFB, uRL, uILC]

        Returns:
            torch.Tensor: observation = [q dq ddq] [ref dref ddref] [uRL] [uRLold, uILCold]
        """
        k     = self.env_step
        q, dq = self._get_state()
        ddq   = (dq-self.dq_old)*self.f_robot
        [uMB, uFB, uRL, uILC] = ctrlNew
        
        t_pol = self.t + self.dt_pol
        
        r_f, dr_f, ddr_f = self.des_traj_at(t = t_pol)
        
        # save useful data
        self.epUilc[:,:,k] = uILC

        # load useful data
        e_old_ep    = self.epEold[:,:,k]
        de_old_ep   = self.epDEold[:,:,k]
        uilc_old_ep    = self.epUilcold[:,:,k]
        
        obs = torch.cat([
            q.flatten(), dq.flatten(),
            r_f.flatten(), dr_f.flatten(), 
            self._uRLold.flatten(), uILC.flatten(), uilc_old_ep.flatten(),
            e_old_ep.flatten(), de_old_ep.flatten(),], dim=0)
        
        obs_scaled = self.normalize_obs(obs)
        return obs_scaled

    def _get_state(self) -> list[torch.Tensor, torch.Tensor]:
        
        # self.data.joint("HIP").qpos
        # self.data.sensordata
        q     = torch.zeros(self.pin_rob._dim_q,1)
        dq    = torch.zeros(self.pin_rob._dim_q,1)
        q[0]  = torch.from_numpy(self.data.sensor("q_hip").data)
        q[1]  = torch.from_numpy(self.data.sensor("q_knee").data)
        dq[0] = torch.from_numpy(self.data.sensor("dq_hip").data)
        dq[1] = torch.from_numpy(self.data.sensor("dq_knee").data)
        q    += self.noise_q_dev * torch.randn(2,1)
        dq   += self.noise_dq_dev * torch.randn(2,1)
        return q, dq
    
    def rescale_action(self, u_np:np.ndarray) -> torch.Tensor:
        """
        Convert normalized action to real one.

        Args:
            u (np.ndarray): action normalized.

        Returns:
            torch.Tensor: action rescaled to real limits.
        """
        u = torch.from_numpy(u_np)
        range_u = self.highLimA-self.lowLimA
        u_scale = torch.multiply(u + torch.ones_like(u), range_u)/2 + self.lowLimA
        
        return u_scale.view(self.pin_rob._dim_u,-1)
    
    def normalize_obs(self, obs:torch.Tensor) -> torch.Tensor:
        """
        Convert real observations to normalized ones.

        Args:
            obs (np.ndarray): real observations.

        Returns:
            np.ndarray: normalized observations.
        """
        
        range_obs = self.highLimO-self.lowLimO
        obs_scaled = 2.0*torch.multiply(obs - self.lowLimO, 1/range_obs) - torch.ones_like(obs)
        obs_scaled = torch.clip(obs_scaled, min=-torch.ones_like(obs_scaled), max=torch.ones_like(obs_scaled))

        return obs_scaled
    
    def rescale_obs(self, observation:np.ndarray) -> np.ndarray:
        """ Convert normalized obs to real one.

        Args:
            obs (np.ndarray): obs normalized.

        Returns:
            torch.Tensor: obs rescaled to real limits.
        """
        obs = torch.from_numpy(observation)
        range_obs = self.highLimO-self.lowLimO
        obs_scale = torch.multiply(obs + torch.ones_like(obs), range_obs)/2.0 + self.lowLimO
        
        return obs_scale.numpy()
    
    def normalize_action(self, u_vec:torch.Tensor) -> np.ndarray:
        """
        Convert real observations to normalized ones in numpy and flatten() shape.

        Args:
            obs (np.ndarray): real observations.

        Returns:
            np.ndarray: normalized observations.
        """
        u = u_vec.flatten()
        range_a = self.highLimA-self.lowLimA
        a_scaled = 2*torch.multiply(u - self.lowLimA, 1/range_a) - torch.ones_like(u) 
        
        return a_scaled.numpy()
    
    def _compute_reward(self, ctrlNew:list[torch.Tensor], errorNew:list[torch.Tensor]) -> float:
        """ Compute reward.

        Args:
            ctrlNew (list[torch.Tensor], optional): [uMB, uFB, uRL, uILC]
            errorNew (list[torch.Tensor], optional): [e de dde]

        Returns:
            torch.Tensor: observation = [q dq ddq] [r dr ddr] [uRL] [uRLold uILCold]
        """
        
        #count       = self.env_step
        
        k     = self.env_step
        uILC_old_ep = self.epUilcold[:,:,k]
        
        uMB, uFB, uRL, uILC = ctrlNew
        e, de, dde = errorNew
        
        rw_du  = - 0.1 * (torch.sum((uRL-self._uRLold)**2))          # reduce variation uRL between consecitive steps
        
        rw_e   = - 6.0 * (torch.sum(e**2))              # reduce error
        #rw_e   = 3.0 * (torch.exp(-18*torch.sum(e**2))-1)              # reduce error
        rw_de  = - 0.1 * (torch.sum(de**2))             # reduce dot error
        if self.env_epis > 0:
            rw_law = - 0.1 * (torch.sum(uILC-uILC_old_ep)**2)       # reduce ddot error
            # rw_law = - 0.1 * (torch.sum(uFB)**2)
        else:
            rw_law = - 0.1 * (torch.sum(uFB)**2)
        
        rw_tot  = rw_e + rw_de + rw_law + rw_du
        
        self.dict["rw_dict"]["rw_e"]     = rw_e
        self.dict["rw_dict"]["rw_de"]    = rw_de
        self.dict["rw_dict"]["rw_law"]   = rw_law
        self.dict["rw_dict"]["rw_du"]    = rw_du
        
        # check first step
        # additional reward for first action
        if self.env_step == 0:
            rw_tot += - (torch.sum(uRL**2))
        
        return rw_tot.item()


class Env_RL(Env_RILC):
    """ Environment for RL.
    
        Args:
            taskT (float, optional): task duration. Defaults to 1.0.
            f_robot (float, optional): frequency of update of robot dynamics (action dynamic). Defaults to 500. (Different from computation timestep of dynamic (used for computation stability): Mujoco set default timestep to 0.002, you can change in xml option or in self.model.opt.timestep)
            scaling (int, optional): scaling factor of frequency of update of policy w.r.t. robot frequency. f_robot = f_policy*scaling. Defaults to 10. (choose a sclaling_factor > 1 for smoother behaviour)            
            stayT (float, optional): time to mantain in last position. Trajectory duration = taskT-stayT. Defaults to 0.0.
            n_ep_reset (int, optional): consecutive episode with same trajectory. Defaults to 5.
            le (float, optional): gain of learning in ILC law for error. Defaults to 0.0002.
            lde (float, optional): gain of learning in ILC law for dot error. Defaults to 0.0004.
            ldde (float, optional): gain of learning in ILC law for ddot error. Defaults to 0.0008.
            kp (float, optional): gain of PD for error. Defaults to 0.04.
            kv (float, optional): gain of PD for dot error. Defaults to 0.025.
            relative_pos (bool, optional): flag to consider the trajectory position respect global position. Defaults to False.
            dtype (torch.dtype, optional): type of variables. Defaults to torch.float32.
            seed (int, optional): seed use to define trajectories. Defaults to None.
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ]
    }

    def __init__(
        self,
        taskT : float       = 1.0,
        f_robot : float     = 500,
        scaling : int       = 10,
        stayT : float       = 0.0,
        n_ep_reset : int    = 5,  
        le : float          = 0.0002,
        lde : float         = 0.0004,
        ldde : float        = 0.0008,
        kp : float          = 0.04,
        kv : float          = 0.025,
        threshold : float   = 1e-3,
        seed : int          = None,
        dtype : torch.dtype = torch.float32,
        **kwargs,
        ):
        
        """ Args:
            taskT (float, optional): task duration. Defaults to 1.0.
            f_robot (float, optional): frequency of update of robot dynamics (action dynamic). Defaults to 500. (Different from computation  timestep of dynamic: Mujoco set default timestep to 0.002, you can change in xml option or in self.model.opt.timestep)
            scaling (int, optional): scaling factor of frequency of update of policy w.r.t. robot frequency. f_robot = f_policy*scaling. Defaults to 10. (choose a sclaling_factor > 1 for smoother behaviour)            
            stayT (float, optional): time to mantain in last position. Trajectory duration = taskT-stayT. Defaults to 0.0.
            n_ep_reset (int, optional): consecutive episode with same trajectory. Defaults to 5.
            le (float, optional): gain of learning in ILC law for error. Defaults to 0.0002.
            lde (float, optional): gain of learning in ILC law for dot error. Defaults to 0.0004.
            ldde (float, optional): gain of learning in ILC law for ddot error. Defaults to 0.0008.
            kp (float, optional): gain of PD for error. Defaults to 0.04.
            kv (float, optional): gain of PD for dot error. Defaults to 0.025.
            threshold (float, optional): rmse threshold to stop updating ILC. Defaults to 1e-3.
            relative_pos (bool, optional): flag to consider the trajectory position respect global position. Defaults to False.
            dtype (torch.dtype, optional): type of variables. Defaults to torch.float32.
            seed (int, optional): seed use to define trajectories. Defaults to None.
        """
        
        super().__init__(
            taskT      = taskT,
            f_robot    = f_robot,
            scaling    = scaling,
            stayT      = stayT,
            n_ep_reset = n_ep_reset,
            le         = le,
            lde        = lde,
            ldde       = ldde,
            kp         = kp,
            kv         = kv,
            seed       = seed,
            dtype      = dtype,
            **kwargs,)
    
    def _ilc_routine(self) -> torch.Tensor:
        
        # compute reference for ILC
        r_, dr_, ddr_ = self.des_traj_at(t = self.t)

        # get state
        # self.data.joint("HIP").qpos
        # self.data.sensordata
        q, dq = self._get_state()
        ddq   = (dq-self.dq_old)*self.f_robot
        
        # compute env error
        e_    = r_ - q
        e_    = self.pin_rob.angle_normalize(e_)
        de_   = dr_ - dq
        dde_  = ddr_ - ddq
        # Update useful memory of ILC
        # iM       = self.pin_rob.getInvMass(q=q)
        # u_delta = torch.matmul(iM, self._uILCold+self._uFBold)
        u_delta = self._uILCold+self._uFBold
        # update ERROR memory of ILC
        self.ILC.updateMemError(e_=e_,de_=de_,dde_=dde_)
        # update INPUT memory of ILC
        self.ILC.updateMemInput(u_delta)
        
        # env variables
        count = self.env_step
        self.epE[:,:,count] = e_.clone()
        self.epDE[:,:,count] = de_.clone()
        
        # # get new control of ILC
        # if self.env_epis != 0 and not self.forced_noILC:
        #     # M = self.pin_rob.getMass(q=q)
        #     uilc = self.ILC.getControl()
        #     # uILC = torch.matmul(M,uilc)
        #     uILC = uilc
        # else:
        uILC = torch.zeros_like(self.pin_rob.u0)
        
        return uILC


class Env_RILC_LISS(Env_RILC):
    """ Environment for RL+ILC with a Lissajous trajectory.
    
        Args:
            taskT (float, optional): task duration. Defaults to 1.0.
            f_robot (float, optional): frequency of update of robot dynamics (action dynamic). Defaults to 500. (Different from computation timestep of dynamic (used for computation stability): Mujoco set default timestep to 0.002, you can change in xml option or in self.model.opt.timestep)
            scaling (int, optional): scaling factor of frequency of update of policy w.r.t. robot frequency. f_robot = f_policy*scaling. Defaults to 10. (choose a sclaling_factor > 1 for smoother behaviour)            
            stayT (float, optional): time to mantain in last position. Trajectory duration = taskT-stayT. Defaults to 0.0.
            n_ep_reset (int, optional): consecutive episode with same trajectory. Defaults to 5.
            le (float, optional): gain of learning in ILC law for error. Defaults to 0.0002.
            lde (float, optional): gain of learning in ILC law for dot error. Defaults to 0.0004.
            ldde (float, optional): gain of learning in ILC law for ddot error. Defaults to 0.0008.
            kp (float, optional): gain of PD for error. Defaults to 0.04.
            kv (float, optional): gain of PD for dot error. Defaults to 0.025.
            relative_pos (bool, optional): flag to consider the trajectory position respect global position. Defaults to False.
            dtype (torch.dtype, optional): type of variables. Defaults to torch.float32.
            seed (int, optional): seed use to define trajectories. Defaults to None.
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ]
    }
    
    def __init__(
        self,
        taskT : float       = 1.0,
        f_robot : float     = 500,
        scaling : int       = 10,
        stayT : float       = 0.0,
        n_ep_reset : int    = 5,  
        le : float          = 0.02,
        lde : float         = 0.04,
        ldde : float        = 0.08,
        kp : float          = 0.4,
        kv : float          = 0.25,
        seed : int          = None,
        dtype : torch.dtype = torch.float32,
        **kwargs,
        ):
        
        """ Args:
                taskT (float, optional): task duration. Defaults to 1.0.
                f_robot (float, optional): frequency of update of robot dynamics (action dynamic). Defaults to 500. (Different from computation  timestep of dynamic: Mujoco set default timestep to 0.002, you can change in xml option or in self.model.opt.timestep)
                scaling (int, optional): scaling factor of frequency of update of policy w.r.t. robot frequency. f_robot = f_policy*scaling. Defaults to 10. (choose a sclaling_factor > 1 for smoother behaviour)            
                stayT (float, optional): time to mantain in last position. Trajectory duration = taskT-stayT. Defaults to 0.0.
                n_ep_reset (int, optional): consecutive episode with same trajectory. Defaults to 5.
                le (float, optional): gain of learning in ILC law for error. Defaults to 0.0002.
                lde (float, optional): gain of learning in ILC law for dot error. Defaults to 0.0004.
                ldde (float, optional): gain of learning in ILC law for ddot error. Defaults to 0.0008.
                kp (float, optional): gain of PD for error. Defaults to 0.04.
                kv (float, optional): gain of PD for dot error. Defaults to 0.025.
                relative_pos (bool, optional): flag to consider the trajectory position respect global position. Defaults to False.
                dtype (torch.dtype, optional): type of variables. Defaults to torch.float32.
                seed (int, optional): seed use to define trajectories. Defaults to None.
        """
        
        super().__init__(
            taskT      = taskT,
            f_robot    = f_robot,
            scaling    = scaling,
            stayT      = stayT,
            n_ep_reset = n_ep_reset,
            le         = le,
            lde        = lde,
            ldde       = ldde,
            kp         = kp,
            kv         = kv,
            seed       = seed,
            dtype      = dtype,
            **kwargs,)
        
        self.complete_traj = TRAJ_LISS
        self.des_traj_at = self.custom_lissajous_at
        
    def _load_pin_robot(self, urdf_path):
        """ define robot
        """
        # ---------------------------------------- ROBOT ---------------------------------------- #
        pin_robot = Sim_RR(urdf_path=urdf_path, mesh_dir=MESH_DIR, ee_name='LH_ANKLE')
        # attribute robot
        pin_robot.setState(q0=TRAJ_LISS[:2,0].clone(), dq0=TRAJ_LISS[2:4,0].clone())
        self.u0 = pin_robot.getGravity(pin_robot.q0).clone()
        
        self.pin_rob = pin_robot
        self.qi = self.pin_rob.q0.clone()
        self.init_qpos = self.pin_rob.q0.flatten().numpy().copy()
        self.init_qvel = self.pin_rob.dq0.flatten().numpy().copy()
    
    def custom_lissajous_at(self, t: float) -> list[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        idx = int(t // self.dt)
        des_traj = self.complete_traj[:, idx]

        q_des   = des_traj[0:2].view(2,1)
        dq_des  = des_traj[2:4].view(2,1)
        ddq_des = des_traj[4:6].view(2,1)
        
        return q_des, dq_des, ddq_des


class Env_RL_LISS(Env_RILC_LISS):
    """ Environment for RL+ILC with a Lissajous trajectory.
    
        Args:
            taskT (float, optional): task duration. Defaults to 1.0.
            f_robot (float, optional): frequency of update of robot dynamics (action dynamic). Defaults to 500. (Different from computation timestep of dynamic (used for computation stability): Mujoco set default timestep to 0.002, you can change in xml option or in self.model.opt.timestep)
            scaling (int, optional): scaling factor of frequency of update of policy w.r.t. robot frequency. f_robot = f_policy*scaling. Defaults to 10. (choose a sclaling_factor > 1 for smoother behaviour)            
            stayT (float, optional): time to mantain in last position. Trajectory duration = taskT-stayT. Defaults to 0.0.
            n_ep_reset (int, optional): consecutive episode with same trajectory. Defaults to 5.
            le (float, optional): gain of learning in ILC law for error. Defaults to 0.0002.
            lde (float, optional): gain of learning in ILC law for dot error. Defaults to 0.0004.
            ldde (float, optional): gain of learning in ILC law for ddot error. Defaults to 0.0008.
            kp (float, optional): gain of PD for error. Defaults to 0.04.
            kv (float, optional): gain of PD for dot error. Defaults to 0.025.
            relative_pos (bool, optional): flag to consider the trajectory position respect global position. Defaults to False.
            dtype (torch.dtype, optional): type of variables. Defaults to torch.float32.
            seed (int, optional): seed use to define trajectories. Defaults to None.
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ]
    }
    
    def __init__(
        self,
        taskT : float       = 1.0,
        f_robot : float     = 500,
        scaling : int       = 10,
        stayT : float       = 0.0,
        n_ep_reset : int    = 5,  
        le : float          = 0.02,
        lde : float         = 0.04,
        ldde : float        = 0.08,
        kp : float          = 0.4,
        kv : float          = 0.25,
        seed : int          = None,
        dtype : torch.dtype = torch.float32,
        **kwargs,
        ):
        
        """ Args:
                taskT (float, optional): task duration. Defaults to 1.0.
                f_robot (float, optional): frequency of update of robot dynamics (action dynamic). Defaults to 500. (Different from computation  timestep of dynamic: Mujoco set default timestep to 0.002, you can change in xml option or in self.model.opt.timestep)
                scaling (int, optional): scaling factor of frequency of update of policy w.r.t. robot frequency. f_robot = f_policy*scaling. Defaults to 10. (choose a sclaling_factor > 1 for smoother behaviour)            
                stayT (float, optional): time to mantain in last position. Trajectory duration = taskT-stayT. Defaults to 0.0.
                n_ep_reset (int, optional): consecutive episode with same trajectory. Defaults to 5.
                le (float, optional): gain of learning in ILC law for error. Defaults to 0.0002.
                lde (float, optional): gain of learning in ILC law for dot error. Defaults to 0.0004.
                ldde (float, optional): gain of learning in ILC law for ddot error. Defaults to 0.0008.
                kp (float, optional): gain of PD for error. Defaults to 0.04.
                kv (float, optional): gain of PD for dot error. Defaults to 0.025.
                relative_pos (bool, optional): flag to consider the trajectory position respect global position. Defaults to False.
                dtype (torch.dtype, optional): type of variables. Defaults to torch.float32.
                seed (int, optional): seed use to define trajectories. Defaults to None.
        """
        
        super().__init__(
            taskT      = taskT,
            f_robot    = f_robot,
            scaling    = scaling,
            stayT      = stayT,
            n_ep_reset = n_ep_reset,
            le         = le,
            lde        = lde,
            ldde       = ldde,
            kp         = kp,
            kv         = kv,
            seed       = seed,
            dtype      = dtype,
            **kwargs,)
        
        self.complete_traj = TRAJ_LISS
        self.des_traj_at = self.custom_lissajous_at
    
    def _ilc_routine(self) -> torch.Tensor:
        
        # compute reference for ILC
        r_, dr_, ddr_ = self.des_traj_at(t = self.t)

        # get state
        # self.data.joint("HIP").qpos
        # self.data.sensordata
        q, dq = self._get_state()
        ddq   = (dq-self.dq_old)*self.f_robot
        
        # compute env error
        e_    = r_ - q
        e_    = self.pin_rob.angle_normalize(e_)
        de_   = dr_ - dq
        dde_  = ddr_ - ddq
        # Update useful memory of ILC
        # iM       = self.pin_rob.getInvMass(q=q)
        # u_delta = torch.matmul(iM, self._uILCold+self._uFBold)
        u_delta = self._uILCold+self._uFBold
        # update ERROR memory of ILC
        self.ILC.updateMemError(e_=e_,de_=de_,dde_=dde_)
        # update INPUT memory of ILC
        self.ILC.updateMemInput(u_delta)
        
        # env variables
        count = self.env_step
        self.epE[:,:,count] = e_.clone()
        self.epDE[:,:,count] = de_.clone()
        
        # # get new control of ILC
        # if self.env_epis != 0 and not self.forced_noILC:
        #     # M = self.pin_rob.getMass(q=q)
        #     uilc = self.ILC.getControl()
        #     # uILC = torch.matmul(M,uilc)
        #     uILC = uilc
        # else:
        uILC = torch.zeros_like(self.pin_rob.u0)
        
        return uILC


class Env_RILC_RANGE(Env_RILC):
    """ Environment for RL+ILC with a Lissajous trajectory.
    
        Args:
            taskT (float, optional): task duration. Defaults to 1.0.
            f_robot (float, optional): frequency of update of robot dynamics (action dynamic). Defaults to 500. (Different from computation timestep of dynamic (used for computation stability): Mujoco set default timestep to 0.002, you can change in xml option or in self.model.opt.timestep)
            scaling (int, optional): scaling factor of frequency of update of policy w.r.t. robot frequency. f_robot = f_policy*scaling. Defaults to 10. (choose a sclaling_factor > 1 for smoother behaviour)            
            stayT (float, optional): time to mantain in last position. Trajectory duration = taskT-stayT. Defaults to 0.0.
            n_ep_reset (int, optional): consecutive episode with same trajectory. Defaults to 5.
            le (float, optional): gain of learning in ILC law for error. Defaults to 0.0002.
            lde (float, optional): gain of learning in ILC law for dot error. Defaults to 0.0004.
            ldde (float, optional): gain of learning in ILC law for ddot error. Defaults to 0.0008.
            kp (float, optional): gain of PD for error. Defaults to 0.04.
            kv (float, optional): gain of PD for dot error. Defaults to 0.025.
            relative_pos (bool, optional): flag to consider the trajectory position respect global position. Defaults to False.
            dtype (torch.dtype, optional): type of variables. Defaults to torch.float32.
            seed (int, optional): seed use to define trajectories. Defaults to None.
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ]
    }
    
    def __init__(
        self,
        taskT : float       = 1.0,
        f_robot : float     = 500,
        scaling : int       = 10,
        stayT : float       = 0.0,
        n_ep_reset : int    = 5,  
        le : float          = 0.02,
        lde : float         = 0.04,
        ldde : float        = 0.08,
        kp : float          = 0.4,
        kv : float          = 0.25,
        seed : int          = None,
        dtype : torch.dtype = torch.float32,
        **kwargs,
        ):
        
        """ Args:
                taskT (float, optional): task duration. Defaults to 1.0.
                f_robot (float, optional): frequency of update of robot dynamics (action dynamic). Defaults to 500. (Different from computation  timestep of dynamic: Mujoco set default timestep to 0.002, you can change in xml option or in self.model.opt.timestep)
                scaling (int, optional): scaling factor of frequency of update of policy w.r.t. robot frequency. f_robot = f_policy*scaling. Defaults to 10. (choose a sclaling_factor > 1 for smoother behaviour)            
                stayT (float, optional): time to mantain in last position. Trajectory duration = taskT-stayT. Defaults to 0.0.
                n_ep_reset (int, optional): consecutive episode with same trajectory. Defaults to 5.
                le (float, optional): gain of learning in ILC law for error. Defaults to 0.0002.
                lde (float, optional): gain of learning in ILC law for dot error. Defaults to 0.0004.
                ldde (float, optional): gain of learning in ILC law for ddot error. Defaults to 0.0008.
                kp (float, optional): gain of PD for error. Defaults to 0.04.
                kv (float, optional): gain of PD for dot error. Defaults to 0.025.
                relative_pos (bool, optional): flag to consider the trajectory position respect global position. Defaults to False.
                dtype (torch.dtype, optional): type of variables. Defaults to torch.float32.
                seed (int, optional): seed use to define trajectories. Defaults to None.
        """
        
        super().__init__(
            taskT      = taskT,
            f_robot    = f_robot,
            scaling    = scaling,
            stayT      = stayT,
            n_ep_reset = n_ep_reset,
            le         = le,
            lde        = lde,
            ldde       = ldde,
            kp         = kp,
            kv         = kv,
            seed       = seed,
            dtype      = dtype,
            **kwargs,)
        
        self.q_range_task = torch.tensor(torch.pi/3)
        rand_number = torch.rand(2,1)*2-1
        self.delta_q_task = torch.tensor([[torch.pi/2],[0]])
        self.qf = self.q_range_task*rand_number + self.delta_q_task


class Env_RL_RANGE(Env_RILC_RANGE):
    """ Environment for RL+ILC with a Lissajous trajectory.
    
        Args:
            taskT (float, optional): task duration. Defaults to 1.0.
            f_robot (float, optional): frequency of update of robot dynamics (action dynamic). Defaults to 500. (Different from computation timestep of dynamic (used for computation stability): Mujoco set default timestep to 0.002, you can change in xml option or in self.model.opt.timestep)
            scaling (int, optional): scaling factor of frequency of update of policy w.r.t. robot frequency. f_robot = f_policy*scaling. Defaults to 10. (choose a sclaling_factor > 1 for smoother behaviour)            
            stayT (float, optional): time to mantain in last position. Trajectory duration = taskT-stayT. Defaults to 0.0.
            n_ep_reset (int, optional): consecutive episode with same trajectory. Defaults to 5.
            le (float, optional): gain of learning in ILC law for error. Defaults to 0.0002.
            lde (float, optional): gain of learning in ILC law for dot error. Defaults to 0.0004.
            ldde (float, optional): gain of learning in ILC law for ddot error. Defaults to 0.0008.
            kp (float, optional): gain of PD for error. Defaults to 0.04.
            kv (float, optional): gain of PD for dot error. Defaults to 0.025.
            relative_pos (bool, optional): flag to consider the trajectory position respect global position. Defaults to False.
            dtype (torch.dtype, optional): type of variables. Defaults to torch.float32.
            seed (int, optional): seed use to define trajectories. Defaults to None.
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ]
    }
    
    def __init__(
        self,
        taskT : float       = 1.0,
        f_robot : float     = 500,
        scaling : int       = 10,
        stayT : float       = 0.0,
        n_ep_reset : int    = 5,  
        le : float          = 0.02,
        lde : float         = 0.04,
        ldde : float        = 0.08,
        kp : float          = 0.4,
        kv : float          = 0.25,
        seed : int          = None,
        dtype : torch.dtype = torch.float32,
        **kwargs,
        ):
        
        """ Args:
                taskT (float, optional): task duration. Defaults to 1.0.
                f_robot (float, optional): frequency of update of robot dynamics (action dynamic). Defaults to 500. (Different from computation  timestep of dynamic: Mujoco set default timestep to 0.002, you can change in xml option or in self.model.opt.timestep)
                scaling (int, optional): scaling factor of frequency of update of policy w.r.t. robot frequency. f_robot = f_policy*scaling. Defaults to 10. (choose a sclaling_factor > 1 for smoother behaviour)            
                stayT (float, optional): time to mantain in last position. Trajectory duration = taskT-stayT. Defaults to 0.0.
                n_ep_reset (int, optional): consecutive episode with same trajectory. Defaults to 5.
                le (float, optional): gain of learning in ILC law for error. Defaults to 0.0002.
                lde (float, optional): gain of learning in ILC law for dot error. Defaults to 0.0004.
                ldde (float, optional): gain of learning in ILC law for ddot error. Defaults to 0.0008.
                kp (float, optional): gain of PD for error. Defaults to 0.04.
                kv (float, optional): gain of PD for dot error. Defaults to 0.025.
                relative_pos (bool, optional): flag to consider the trajectory position respect global position. Defaults to False.
                dtype (torch.dtype, optional): type of variables. Defaults to torch.float32.
                seed (int, optional): seed use to define trajectories. Defaults to None.
        """
        
        super().__init__(
            taskT      = taskT,
            f_robot    = f_robot,
            scaling    = scaling,
            stayT      = stayT,
            n_ep_reset = n_ep_reset,
            le         = le,
            lde        = lde,
            ldde       = ldde,
            kp         = kp,
            kv         = kv,
            seed       = seed,
            dtype      = dtype,
            **kwargs,)
        
    def _ilc_routine(self) -> torch.Tensor:
        
        # compute reference for ILC
        r_, dr_, ddr_ = self.des_traj_at(t = self.t)

        # get state
        # self.data.joint("HIP").qpos
        # self.data.sensordata
        q, dq = self._get_state()
        ddq   = (dq-self.dq_old)*self.f_robot
        
        # compute env error
        e_    = r_ - q
        e_    = self.pin_rob.angle_normalize(e_)
        de_   = dr_ - dq
        dde_  = ddr_ - ddq
        # Update useful memory of ILC
        # iM       = self.pin_rob.getInvMass(q=q)
        # u_delta = torch.matmul(iM, self._uILCold+self._uFBold)
        u_delta = self._uILCold+self._uFBold
        # update ERROR memory of ILC
        self.ILC.updateMemError(e_=e_,de_=de_,dde_=dde_)
        # update INPUT memory of ILC
        self.ILC.updateMemInput(u_delta)
        
        # env variables
        count = self.env_step
        self.epE[:,:,count] = e_.clone()
        self.epDE[:,:,count] = de_.clone()
        
        # # get new control of ILC
        # if self.env_epis != 0 and not self.forced_noILC:
        #     # M = self.pin_rob.getMass(q=q)
        #     uilc = self.ILC.getControl()
        #     # uILC = torch.matmul(M,uilc)
        #     uILC = uilc
        # else:
        uILC = torch.zeros_like(self.pin_rob.u0)
        
        return uILC




if __name__ == '__main__':
    
    env = Env_RILC_LISS(f_robot=100, scaling=2)
    env.reset()
    
    for _ in range(10):
        print(f'episode:{env.env_epis}')
        
        for _ in range(env.samples):
            
            # Key section
            action = env.action_space.sample()
            observation, reward, done, info, _ = env.step(action)
            
            if done:
                print(env.env_step)
                print(env.t)
                break
        env.reset()

    from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
    
    model = mujoco.MjModel.from_xml_path(MJC_PATH)    
    __actual_dt = model.opt.timestep
    frame_skip  = int((1/500)/__actual_dt)

    model.dof_frictionloss = model.dof_frictionloss*(1.2)
    data = mujoco.MjData(model)
    qpos_init = data.qpos 
    qvel_init = data.qvel 
    
    mujoco_renderer = MujocoRenderer(model, data, None, 800, 600)
    
    for i in range(env.complete_traj.shape[1]):
        qpos_init[0] = env.complete_traj[0,i]
        qpos_init[1] = env.complete_traj[1,i]
        qvel_init[0] = env.complete_traj[2,i]
        qvel_init[1] = env.complete_traj[3,i]
        mujoco.mj_resetData(model, data)
        data.qpos = qpos_init
        data.qvel = qvel_init
        mujoco.mj_step(model, data, nstep=frame_skip)
        mujoco.mj_rnePostConstraint(model, data)
        
        mujoco_renderer.render("human")
    
    mujoco_renderer.close()
    