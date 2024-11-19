import os
import subprocess
import meshcat_shapes
import torch
import pinocchio        as pin

from classes.robots.ABC_robot_  import RobURDF_ABC, SimRobURDF_ABC
from pinocchio.visualize        import MeshcatVisualizer


# Main methods for robot implementation
class RobURDF(RobURDF_ABC):
    """ Class that parsering urdf with pinocchio library and wrapping with torch library. """

    def __init__(self,urdf_path:str,mesh_dir:str=None,dt=0.01,visual=False,dtype=torch.float32,) -> None:
        """ Class that parsering urdf with pinocchio library and wrapping with torch library.
   
        Args:
            urdf_path (str): relative path to urdf file.
            mesh_dir (str): relative path to mesh folder.
            dtype (torch.dtype, optional): type of variables in class. Defaults to torch.float32.
            visual (bool, optional): to visualize robot with meshcat. Defaults to False.
            dt (float, optional): time sample [s] for integration and visualization. Defaults to 0.01 [s].
        
        NOTE: For particular method and some examples see: 
            https://docs.ros.org/en/melodic/api/pinocchio/html/namespacepinocchio.html
            https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/topic/doc-v2/doxygen-html/index.html

        """
        
        # 3 casi da implementare
        # il path è già quello giusto
        # il path è relativo a questo file
        # il path è relativo alla directory da dove è stato lanciato il codice
        super().__init__()
        current_folder = os.getcwd() # absolute path of the folder where the code is running.
        #current_file = os.path.abspath(__file__) # absolute path of this file.
        
        urdf_path = os.path.join(current_folder, urdf_path)
        
        if mesh_dir is None:
            model, _, _ = pin.buildModelsFromUrdf(urdf_path) 
        else:
            mesh_dir = os.path.join(current_folder, mesh_dir)
            model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_path, mesh_dir) 

        self.robModel   = model # urdf info
        self.robData    = model.createData()  # useful functions obtainable for manipulation of urdf data
        
        self._dt        = dt
        self._dtype     = dtype
        
        self.q0         = torch.from_numpy(pin.neutral(self.robModel)).expand(1,-1).T.type(dtype).clone()
        self.dq0        = torch.zeros(self.q0.shape).type(dtype).clone()
        self.ddq0       = torch.zeros(self.q0.shape).type(dtype).clone()

        self.q          = self.q0.clone()
        self.dq         = self.dq0.clone()
        self.ddq        = self.ddq0.clone()
        self.u0         = self.getGravity(self.q0).clone()
        
        self._dim_q:int = self.robModel.nq
        self._dim_u:int = self.robModel.nq      # full-actuated
        self._open      = True
        
        if mesh_dir is not None:
            if visual:
                self.robColl    = collision_model
                self.robVis     = visual_model
                self.viz        = MeshcatVisualizer(self.robModel, self.robColl, self.robVis)
                self._open      = False
                self.viz.initViewer(open=True)            
                self.viz.loadViewerModel("pinocchio")
                self.viz.display(self.q.flatten().detach().numpy())
    
    
    # METHODS FOR ROBOT'S KINEMATICS

    def _computeForwKin(self, q:torch.Tensor=None) -> None:
        """ Compute forward kinematic of frames.

        Args:
            q (torch.Tensor, optional): joint position variables. Defaults to None, it is set to self.q.
        """
        if q is None:
            q = self.q
        q = q.flatten().detach().numpy()
        pin.framesForwardKinematics(self.robModel, self.robData, q)

    def _computeJacobians(self, q:torch.Tensor=None) -> None:
        """Compute jacobian of all joints.

        Args:
            q (torch.Tensor, optional): joint position variables. Defaults to None, it is set to self.q.
        """
        if q is None:
            q = self.q
            
        self._computeForwKin(q)
        q = q.flatten().detach().numpy()
        pin.computeJointJacobians(self.robModel, self.robData, q)

    def _computeDotJacobians(self, q:torch.Tensor=None, dq:torch.Tensor=None,) -> None:
        """Compute dot jacobian of all joints.

        Args:
            q (torch.Tensor, optional): joint position variables. Defaults to None, it is set to self.q.
            dq (torch.Tensor, optional): joint velocity variables. Defaults to None, it is set to self.dq.
        """
        if q is None:
            q = self.q
        if dq is None:
            dq = self.dq
            
        self._computeForwKin(q)
        q = q.flatten().detach().numpy()
        dq = dq.flatten().detach().numpy()
        pin.computeJointJacobiansTimeVariation(self.robModel, self.robData, q, dq)
    
    
    # METHODS FOR ROBOT'S DYNAMICS

    def _updateState(self, dt:float = None, u0:torch.Tensor=None, damp_fl=True) -> None:
        """ Update state [q,dq]^T variables wrt robot dynamics.

        Args:
            u0 (torch.Tensor, optional): input to system. If None it is set to zero.
        """
        
        if dt==None:
            dt = self._dt
            
        x       = self.getState()
        x_new   = self._integrateForwDyn(x=x, u=u0, dt=dt, damp_fl=damp_fl)
        
        self.ddq    = self._getForwDynDDq(x[:self._dim_q], x[-self._dim_q:], u0, damp_fl)
        self.q      = x_new[:self._dim_q]
        self.dq     = x_new[-self._dim_q:]

    def _getForwDynDDq(self, q:torch.Tensor, dq:torch.Tensor, u:torch.Tensor, damp_fl:bool=True) -> torch.Tensor:
        
        iM  = self.getInvMass(q)
        C   = self.getCoriolis(q,dq)
        G   = self.getGravity(q)
        
        if damp_fl:
            D   = self.getDamping()
            C   = C+D

        ddq = torch.matmul(iM, torch.matmul(-C,dq) - G + u)
        
        return ddq

    def _integrateForwDyn(self, x:torch.Tensor, u:torch.Tensor, dt:float = None, damp_fl=True) -> torch.Tensor:
        
        x_new = self._rk4Dyn( x=x, u=u, dt=dt, damp_fl=damp_fl)
        return x_new

    def _rk4Dyn(self, x:torch.Tensor, u:torch.Tensor, dt:float = None, damp_fl=True) -> torch.Tensor:
        
        """Runge Kutta 4th order.

        Args:
            x (torch.Tensor): state column vector [q,dq]
            u (torch.Tensor): u0 column vector
            dt (float): sample time [s]

        Returns:
            _type_: _description_
        """
        fun=self.getForwDyn
        if dt==None:
            dt = self._dt
        
        k1 = fun(x, u, damp_fl)
        k2 = fun(x + k1*dt/2, u, damp_fl) 
        k3 = fun(x + k2*dt/2, u, damp_fl)
        k4 = fun(x + k3*dt, u, damp_fl) 
        
        x_new = x + (k1 + (k2 + k3)*2 + k4)* dt / 6
        
        return x_new

    def getMass(self, q:torch.Tensor=None) -> torch.Tensor:
        """Return Mass Matrix

        Args:
            q (torch.Tensor, optional): joint position variables. Defaults to None, it is set to self.q.
        """
        
        if q is None:
            q = self.q
        q = q.flatten().detach().numpy()
        M = torch.from_numpy(pin.crba(self.robModel, self.robData, q)).type(self._dtype)
        return M

    def getInvMass(self, q:torch.Tensor=None) -> torch.Tensor:
        """ Return Inverse of Mass Matrix  

        Args:
            q (torch.Tensor, optional): joint position variables. Defaults to None, it is set to self.q.
        """    
        if q is None:
            q = self.q
        q = q.flatten().detach().numpy()
        iM = torch.from_numpy(pin.computeMinverse(self.robModel,self.robData,q)).type(self._dtype)
        return iM

    def getCoriolis(self, q:torch.Tensor=None, dq:torch.Tensor=None) -> torch.Tensor:
        """ Return Coriolis Matrix 

        Args:
            q (torch.Tensor, optional): joint position variables. Defaults to None, it is set to self.q.
            dq (torch.Tensor, optional): joint velocity variables. Defaults to None, it is set to self.dq.
        """
        if q is None:
            q = self.q
        if dq is None:
            dq = self.dq
        
        q = q.flatten().detach().numpy()
        dq = dq.flatten().detach().numpy()
        C = torch.from_numpy(pin.computeCoriolisMatrix(self.robModel,self.robData,q,dq)).type(self._dtype)
        return C

    def getCoriolisVec(self, q:torch.Tensor=None, dq:torch.Tensor=None) -> torch.Tensor:
        """ Return Coriolis Vector Matrix 

        Args:
            q (torch.Tensor, optional): joint position variables. Defaults to None, it is set to self.q.
            dq (torch.Tensor, optional): joint velocity variables. Defaults to None, it is set to self.dq.
        """
        if q is None:
            q = self.q
        if dq is None:
            dq = self.dq
            
        C = self.getCoriolis(q, dq)
        c_vec = torch.matmul(C, dq)
        
        return c_vec

    def getGravity(self, q:torch.Tensor=None) -> torch.Tensor:
        """ Return Generalized Gravity Matrix 
         
        Args:
             q (torch.Tensor, optional): joint position variables. Defaults to None, it is set to self.q.
        """ 
        if q is None:
            q = self.q
        q = q.flatten().detach().numpy()
        G = torch.from_numpy(pin.computeGeneralizedGravity(self.robModel,self.robData,q)).type(self._dtype).view(-1,1)
        return G

    def getDamping(self):
        """ Return Damping Matrix """
        d_vec = self.robModel.damping
        D = torch.diag(torch.from_numpy(d_vec)).type(self._dtype)
        
        return D
    
    def getFriction(self):
        """ Return Damping Matrix """
        fr_vec = self.robModel.friction
        Fr = torch.diag(torch.from_numpy(fr_vec)).type(self._dtype)
        
        return Fr
    
    def getForwDyn(self, state:torch.Tensor, u0:torch.Tensor=None, damp_fl:bool=True) -> torch.Tensor:
        """ Get forward dynamic (dot state) of the system given current state and u0.
        Compute all necessary matrices from state.
        
        Args:
            state (torch.Tensor): [q,dq] column vector
            u0 (torch.Tensor, optional): u0 to system. Defaults to None, u0 is set to zero.
            damp_fl (bool, optional): boolean to use damping in dynamics. Defaults to False.

        Returns:
            torch.Tensor: dot state as vector column [dq,ddq]^T
        """
        if u0 is None:
            u0 = torch.zeros_like(self.dq).type(self._dtype)
        else:
            u0 = u0.type(self._dtype)
            
        q   = state[:self._dim_q]
        dq  = state[-self._dim_q:]
        
        ddq = self._getForwDynDDq(q=q, dq=dq, u=u0, damp_fl=damp_fl)
        
        return torch.cat([dq, ddq],dim=0)

    def getInvDyn(self, q:torch.Tensor,dq:torch.Tensor,ddq:torch.Tensor,damp_fl:bool=True) -> torch.Tensor:
        """ 
        Get inverse dynamic (tau) of the system given current state and u0.
        Compute all necessary matrices from state.
        
        Args:
            q (torch.Tensor): joint position variables.
            dq (torch.Tensor): joint velocity variables.
            ddq (torch.Tensor): joint acceleration variables.
            damp_fl (bool, optional): boolean to use damping in dynamics. Defaults to False.

        Returns:
            torch.Tensor: tau as vector column
        """
        q   = q.type(self._dtype)
        dq  = dq.type(self._dtype)
        ddq = ddq.type(self._dtype)
        M   = self.getMass(q)
        C   = self.getCoriolis(q,dq)
        G   = self.getGravity(q)
        
        if damp_fl:
            D   = self.getDamping()
            C = C+D
        
        tau = torch.matmul(M,ddq) + torch.matmul(C,dq) + G
        
        return tau

    def getNewState(self, dt:float = None, u0:torch.Tensor=None, damp_fl=True) -> list[torch.Tensor, torch.Tensor]:
        """
        Update state [q,dq]^T variables wrt robot dynamics.

        Args:
            u0 (torch.Tensor, optional): input to system. If None it is set to zero.
        Returns:
            list[torch.Tensor, torch.Tensor]: [self.q, self.dq]
        """
        
        self._updateState(dt=dt, u0=u0, damp_fl=damp_fl)
        
        return [self.q, self.dq]


    # USEFUL METHODS

    def _getFramesName(self) -> list[str]:
        
        str_list = []
        
        for i in range(self.robModel.nframes):
            str_list.append(self.robModel.frames[i].name)
        return str_list

    def _getFramesDist(self, q:torch.Tensor=None, only_jointDist = False) -> dict[torch.Tensor]:

        self._computeForwardKin(q)
        dist = {}
        for i in range(self.nframe):
            fname:str = self.robModel.frames[i].name
            fnameID:int = self.robModel.getFrameId(fname)
            # compute transformation Ti of frame i respect origin (world) frame
            Ti = self.robData.oMf[fnameID]
            if i == 0:
                d = Ti.translation
            else:
                d = Ti.translation
            
            dist[fname+" "+f"ID{fnameID}"] = (torch.from_numpy(d).type(self._dtype))
        
        # if only_jointDist:
        #     dist = dict(filter(lambda item: 'joint' in item[0], dist.items()))

        return dist

    def _render_axes(self) -> None:
        """ Update frames visualization. \\
            As per the de-facto standard (Blender, OpenRAVE, RViz, ...), the x-axis is red, the y-axis is green and the z-axis is blue."""
        
        self._computeForwKin()
        
        for i in range(self.robModel.nframes):
            # compute transformation Ti of frame i respect origin (world) frame
            frame_name = self.robModel.frames[i].name
            frame_id = self.robModel.getFrameId(frame_name)
            Ti = self.robData.oMf[frame_id]
            
            meshcat_shapes.frame(
                self.viz.viewer[f"frame_{frame_name}"],
                axis_length=0.15,
                axis_thickness=0.005,
                opacity=0.5,
                origin_radius=0.008,
            )
            
            self.viz.viewer[f"frame_{frame_name}"].set_transform(pin.SE3(Ti.rotation, Ti.translation).homogeneous)

    def _joints_type(self) -> None:
        """ Print Joint Types. """
        for joint in self.robModel.joints:
            joint_type = joint.shortname()
            if (joint_type == "JointModelPX") or (joint_type == "JointModelPY") or (joint_type == "JointModelPZ"):
                print("Joint", joint.id, "is prismatic")
            elif (joint_type == "JointModelRX") or (joint_type == "JointModelRY") or (joint_type == "JointModelRZ"):
                print("Joint", joint.id, "is revolute")
            else:
                print("Not implemented yet.")

    def _getLengthLink(self, only_jointDist=True) -> torch.Tensor:
        """ For serial manipulators the links' length are computed as difference of joints position.\
            Check URDF for the correct definition of joints. """
        dist = self._getFramesDist(only_jointDist)
        
        len_link = torch.zeros(len(dist)-1)    
        for idx, (_, value) in enumerate(dist.items()):
            # only (y,z) coordinate 
            val = value
            val[0] = 0.0
            if idx == 0:
                pass
            else:
                l = torch.sqrt(torch.sum((val-old)**2))
                len_link[idx-1] = l
            old = val
        return len_link

    def render(self, dt:float = None, frames_flag=False) -> None:
        """ Update visualization. """
        if dt==None:
            dt = self._dt
        if self._open == False:
            subprocess.run(['open', self.viz.viewer.url()], check=True)
            self._open   = True
        
        if frames_flag:
            self._render_axes()
        self.viz.display(self.q.flatten().detach().numpy())
        self.viz.sleep(dt)

    def getState(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: state as vector column [q,dq]
        """
        return torch.cat([self.q,self.dq],dim=0)

    def getDotState(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: dot state as vector column [dq,ddq]
        """
        
        return torch.cat([self.dq,self.ddq],dim=0)

    def setState(self,q:torch.Tensor=None, dq:torch.Tensor=None, ddq:torch.Tensor=None,
                 q0:torch.Tensor=None, dq0:torch.Tensor=None, ddq0:torch.Tensor=None) -> None:
        """
        Update only not None input of function.

        Args:
            q (torch.Tensor, optional): joint position variable. Defaults to None.
            dq (torch.Tensor, optional): joint velocity variable. Defaults to None.
            ddq (torch.Tensor, optional): joint acceleration variable. Defaults to None.
        """
        if q != None:
            self.q = q.type(self._dtype).clone()
        if dq != None:
            self.dq = dq.type(self._dtype).clone()
        if ddq != None:
            self.ddq = ddq.type(self._dtype).clone()
        if q0 != None:
            self.q0 = q0.type(self._dtype).clone()
        if dq0 != None:
            self.dq0 = dq0.type(self._dtype).clone()
        if ddq0 != None:
            self.ddq0 = ddq0.type(self._dtype).clone()




# Main methods for robot Simulator implementation
class SimRobURDF(RobURDF, SimRobURDF_ABC):
    """ Class that parsering urdf of "realistic" manipulator. """
        
    def __init__(self,urdf_path:str,mesh_dir:str=None,dt=0.01,visual=False,dtype=torch.float32,ee_name="end_effector",) -> None:
        """ Class that parsering urdf of RR "realistic" manipulator.
    
        Args:
            urdf_path (str): relative path to urdf file.
            mesh_dir (str): relative path to mesh folder.
            dtype (torch.dtype, optional): type of variables in class. Defaults to torch.float32.
            visual (bool, optional): to visualize robot with meshcat. Defaults to False.
            dt (float, optional): time sample [s] for integration and visualization. Defaults to 0.01 [s].
        
        NOTE: For particular method and some examples see: 
            https://docs.ros.org/en/melodic/api/pinocchio/html/namespacepinocchio.html
            https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/topic/doc-v2/doxygen-html/index.html

        """
        
        super().__init__(urdf_path=urdf_path, mesh_dir=mesh_dir, visual=visual, dt = dt, dtype=dtype,ee_name=ee_name,)
        self._extractLimits()
        self._uold  = self.u0.clone()
    
    def _extractLimits(self) -> None: 
        "Extraction of limits of q, dq, ddq, u, du. (dq, ddq, u, du have symmetric ranges)"
        robot = self.robModel
        dtype = self._dtype
        try:
            self._q_M   = torch.from_numpy(robot.upperPositionLimit).expand(1,-1).T.type(dtype)
            self._q_m   = torch.from_numpy(robot.lowerPositionLimit).expand(1,-1).T.type(dtype)
        except:
            self._q_M   = torch.tensor([[torch.inf]],dtype=dtype).expand(self._dim_q,-1)
            self._q_m   = -self._q_M
            
        try:    
            self._dq_M  = torch.from_numpy(robot.velocityLimit).expand(1,-1).T.type(dtype)
        except:
            self._dq_M  = torch.tensor([[torch.inf]],dtype=dtype).expand(self._dim_q,-1)
        
        try:
            self._u_M   = torch.from_numpy(robot.effortLimit).expand(1,-1).T.type(dtype)
        except:
            self._u_M   = torch.tensor([[torch.inf]],dtype=dtype).expand(self._dim_u,-1)
            
        self._ddq_M = torch.tensor([[torch.inf]],dtype=dtype).expand(self._dim_q,-1)
        self._du_M  = torch.tensor([[torch.inf]],dtype=dtype).expand(self._dim_u,-1)
        
        # # Always symmetric range
        # self._dq_m  = -self._dq_M
        # self._ddq_m = -self._ddq_M
        # self._u_m   = -self._u_M
        # self._du_m  = -self._du_M       
        
    def _saturatedq(self,dq)->torch.Tensor:
        for idx, value in enumerate(dq):
           if torch.abs(value) > self._dq_M[idx]:
               if value < 0:
                   dq[idx] = -self._dq_M[idx]
               else:
                   dq[idx] = self._dq_M[idx]
        return dq
    
    def _saturateddq(self,ddq)->torch.Tensor:
        
        for idx, value in enumerate(ddq):
           if torch.abs(value) > self._ddq_M[idx]:
               if value < 0:
                   ddq[idx] = -self._ddq_M[idx]
               else:
                   ddq[idx] = self._ddq_M[idx]
        return ddq
    
    def _saturateu(self, u:torch.Tensor)->torch.Tensor:
        
        # delta_u = u-self._uold
        # for idx, value in enumerate(delta_u):
        #    if torch.abs(value) > self._du_M[idx]:
        #        if value < 0:
        #            delta_u[idx] = -self._du_M[idx]
        #        else:
        #            delta_u[idx] = self._du_M[idx]
        # u = self._uold + delta_u
        
        for idx, value in enumerate(u):
            if torch.abs(value) > self._u_M[idx]:
                if value < 0:
                    u[idx] = -self._u_M[idx]
                else:
                    u[idx] = self._u_M[idx]
        #self._uold = u
        return u

    def _getForwDynDDq(self, q:torch.Tensor, dq:torch.Tensor, u:torch.Tensor, damp_fl:bool=True) -> torch.Tensor:
        
        u   = self._saturateu(u)
        ddq = super()._getForwDynDDq(q=q,dq=dq,u=u,damp_fl=damp_fl)
        #ddq = self._saturateddq(ddq)
        
        return ddq
    
    # def getInvDyn(self, q:torch.Tensor,dq:torch.Tensor,ddq:torch.Tensor,damp_fl=True) -> torch.Tensor:
    #     """ 
    #     Get inverse dynamic (tau) of the system given current state and action.
    #     Compute all necessary matrices from state.
        
    #     Args:
    #         q (torch.Tensor): joint position variables.
    #         dq (torch.Tensor): joint velocity variables.
    #         ddq (torch.Tensor): joint acceleration variables.
    #         damp_fl (bool, optional): boolean to use damping in dynamics. Defaults to False.

    #     Returns:
    #         torch.Tensor: tau as vector column
    #     """
    #     dq  = self._saturatedq(dq)
    #     ddq = self._saturateddq(ddq)
    #     tau = super().getInvDyn(q,dq,ddq,damp_fl)
    #     tau = self._saturateu(tau)
        
    #     return tau
    
    # def _rk4Dyn(self, x:torch.Tensor, u:torch.Tensor, dt:float=None, damp_fl=True) -> torch.Tensor:    
    #     """ Runge Kutta 4th order with saturation.

    #     Args:
    #         x (torch.Tensor): state column vector [q,dq]
    #         u (torch.Tensor): u0 column vector
    #         dt (float): sample time [s]

    #     Returns:
    #         _type_: _description_
    #     """
        
    #     if dt==None:
    #         dt = self._dt
        
    #     q   = x[:self._dim_q]
    #     dq  = x[-self._dim_q:]
        
    #     fun = self._getForwDynDDq
    #     s_v = self._saturatedq
        
    #     k1_a = fun(q, dq, u, damp_fl)
    #     k1_v = s_v(dq + k1_a*dt)
        
    #     k2_a = fun(q + k1_v*dt/2, k1_v/2, u, damp_fl)
    #     k2_v = s_v(dq + k2_a*dt)
        
    #     k3_a = fun(q + k2_v*dt/2, k2_v/2, u, damp_fl)
    #     k3_v = s_v(dq + k3_a*dt)
        
    #     k4_a = fun(q + k3_v*dt, k3_v, u, damp_fl)
    #     k4_v = s_v(dq + k4_a*dt)
        
    #     dq_new = dq + (k1_a + (k2_a + k3_a)*2 + k4_a)* dt / 6
    #     q_new = q + (k1_v + (k2_v + k3_v)*2 + k4_v)* dt / 6
        
    #     return torch.cat([q_new, dq_new],dim=0)

