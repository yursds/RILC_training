from classes.robots.builder_manipulator      import SimManURDF
import torch
import os

# Main methods for RR robot simulator implementation
class Sim_RR(SimManURDF):
    """ Class that parsering urdf of RR "realistic" manipulator. """
        
    def __init__(self,
                urdf_path:str,
                mesh_dir:str       = None,
                dt:float           = 0.01,
                visual:bool        = False,
                dtype:torch.dtype  = torch.float32,
                ee_name            = "end_effector",
        ) -> None:
        """ Class that parsering urdf of RR manipulator.
   
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
        self.base_pos, _ = self.getForwKinJoint(name_joint=ee_name)

     
    def getForwKinEE(self, q:torch.Tensor=None) -> list[torch.Tensor, torch.Tensor]:
        """ Compute forward kinematic of frame end effector only for x-y plane. [trasl, rot] """
        
        traslEE, rotEE = super().getForwKinEE(q)
        
        return traslEE[1:3], rotEE[1:3,1:3]

    def getForwKinJoint(self,  name_joint:str, q:torch.Tensor=None) -> list[torch.Tensor, torch.Tensor]:
        """ Compute forward kinematic of frame joint with name = name_joint. [trasl, rot] """
        
        trasl_j, rot_j = super().getForwKinJoint(q=q, name_joint=name_joint)
        
        return trasl_j[1:3], rot_j[1:3,1:3]
    
    def getJacPosEE(self, q:torch.Tensor=None) -> torch.Tensor:
        """ Return Jacobian for Position only for x-y plane.
         
        Args:
             q (torch.Tensor, optional): joint position variables. Defaults to None, it is set to self.q.
        """
        return super().getJacPosEE(q)[1:3,:]
    
    def getDotJacPosEE(self, q:torch.Tensor=None, dq:torch.Tensor=None) -> torch.Tensor:
        """ Return Jacobian for Position of end-effector frame only for x-y plane.
         
        Args:
             q (torch.Tensor, optional): joint position variables. Defaults to None, it is set to self.q.
             dq (torch.Tensor, optional): joint velocity variables. Defaults to None, it is set to self.dq.
        """ 
        return super().getDotJacPosEE(q,dq)[1:3,:]

    def _extractLimits(self) -> None:
        return super()._extractLimits()
    
    
# Double Pendulum robot simulator (example-robot-data modified)
class DoublePendulum(Sim_RR):
    """ Simulator of Double Pendulum PLANE Y-Z. """
    
    def __init__(self,dt=0.01,visual=False,dtype=torch.float32):
        """ Class that parsering Double Pendulum urdf. (example-robot-data modified)
   
        Args:
            dtype (torch.dtype, optional): type of variables in class. Defaults to torch.float32.
            visual (bool, optional): to visualize robot with meshcat. Defaults to False.
            dt (float, optional): time sample [s] for integration and visualization. Defaults to 0.01 [s].
        
        NOTE: For particular method and some examples see: 
            https://docs.ros.org/en/melodic/api/pinocchio/html/namespacepinocchio.html
            https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/topic/doc-v2/doxygen-html/index.html

        """
        abs_path = os.path.dirname(os.path.abspath(__file__))
        urdf_path   = abs_path+"/robot_models/double_pendulum/urdf/double_pendulum.urdf" 
        mesh_dir    = abs_path+"/robot_models/double_pendulum/meshes"  # Directory containing meshes associated with the model 
        ee_name     = "end_effector"
        super().__init__(urdf_path=urdf_path, mesh_dir=mesh_dir, visual=visual, dt = dt, dtype=dtype,ee_name=ee_name)
        
        self._extractLimits()
        
        # init past action
        self._uold  = torch.zeros(self._dim_u,1).type(self._dtype)
        
    def _extractLimits(self) -> None:
        """ Custom limits """
        # no spec on urdf -> spec of franka emika panda
        dqM         = 2.6       #[rad/s]
        ddqM        = 10        #[rad/s^2]
        uM          = 2        #[Nm]    #changed
        duM         = 100      #[Nm/s]
        # for method angle_normalization
        qM          = torch.pi
        
        # limits (symmetric)
        self._q_M   = torch.tensor([qM]).expand(self._dim_q).unsqueeze(0).T.type(self._dtype)
        self._dq_M  = torch.tensor([dqM]).expand(self._dim_q).unsqueeze(0).T.type(self._dtype)
        self._ddq_M = torch.tensor([ddqM]).expand(self._dim_q).unsqueeze(0).T.type(self._dtype)
        self._u_M   = torch.tensor([uM]).expand(self._dim_q).unsqueeze(0).T.type(self._dtype)
        self._du_M  = torch.tensor([duM*self._dt]).expand(self._dim_q).unsqueeze(0).T.type(self._dtype)


# Particular Double Pendulum --> Leg robot simulator
class LegDoublePend(DoublePendulum):
    """ Simulator of Leg robot from Double Pendulum robot. """
    
    def __init__(self,dt=0.01,visual=False,dtype=torch.float32):
        """ Simulator of Leg robot from Double Pendulum robot. (example-robot-data modified)
   
        Args:
            dtype (torch.dtype, optional): type of variables in class. Defaults to torch.float32.
            visual (bool, optional): to visualize robot with meshcat. Defaults to False.
            dt (float, optional): time sample [s] for integration and visualization. Defaults to 0.01 [s].
        
        NOTE: For particular method and some examples see: 
            https://docs.ros.org/en/melodic/api/pinocchio/html/namespacepinocchio.html
            https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/topic/doc-v2/doxygen-html/index.html

        """
    
        super().__init__(visual=visual, dt = dt, dtype=dtype)
        
        self._extractLimits()
        
        # init state
        self.q0     = torch.tensor([[2.0240, 1.5780]]).T
        self.q      = torch.tensor([[2.0240, 1.5780]]).T


