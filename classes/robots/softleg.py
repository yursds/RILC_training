import torch
from classes.robots.manipulator_RR      import Sim_RR, SimManURDF
import os 

class SoftLeg_RR(Sim_RR):
    """ Simulator of SoftLeg. """
    
    def __init__(self,visual=False,dtype=torch.float32,fs=500):
        """ Class that parsering urdf.
    
        Args:
            dtype (torch.dtype, optional): type of variables in class. Defaults to torch.float32.
            visual (bool, optional): to visualize robot with meshcat. Defaults to False.
            dt (float, optional): time sample [s] for integration and visualization. Defaults to 0.01 [s].
        
        NOTE: For particular method and some examples see: 
            https://docs.ros.org/en/melodic/api/pinocchio/html/namespacepinocchio.html
            https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/topic/doc-v2/doxygen-html/index.html

        """
        abs_path = os.path.dirname(os.path.abspath(__file__))
        urdf_path   = abs_path+"/robot_models/softleg_urdf/urdf/softleg-rlilc.urdf"
        mesh_dir    = abs_path+"/robot_models/softleg_urdf/meshes"  # Directory contenente le mesh associate al modello 
        ee_name     = "contact_joint"
        base_name   = "contact_joint"
        
        self._fs    = fs    # sensors frequency [Hz]
        dt          = 1/self._fs
        super().__init__(urdf_path=urdf_path,mesh_dir=mesh_dir,visual=visual,dt=dt,dtype=dtype,ee_name=ee_name,)
        
        #self.q0    = torch.tensor([[-2.0,1.05]]).T
        self.q0    = torch.tensor([[0.0,0.0]]).T
        self.q     = self.q0.clone()
        self.u0    = self.getGravity(self.q0).clone()
        self._uold = self.u0.clone()
        
        # limits (symmetric)
        ddqM        = 15      #[rad/s^2]
        duM         = 10      #[Nm/s]
        self._ddq_M = torch.tensor([[ddqM]]).expand(self._dim_q,1).type(self._dtype)
        self._du_M  = torch.tensor([[duM*self._dt]]).expand(self._dim_u,1).type(self._dtype) # scale dot u is not accessible
        self.base_pos, _ = self.getForwKinJoint(name_joint=base_name)
        if visual:
            self.viz.viewer['/Cameras/default/rotated/<object>'].set_property('zoom',2)

class SoftLeg_RR_3D(SimManURDF):
    """ Simulator of SoftLeg. """
    
    def __init__(self,visual=False,dtype=torch.float32,fs=500):
        """ Class that parsering urdf.
    
        Args:
            dtype (torch.dtype, optional): type of variables in class. Defaults to torch.float32.
            visual (bool, optional): to visualize robot with meshcat. Defaults to False.
            dt (float, optional): time sample [s] for integration and visualization. Defaults to 0.01 [s].
        
        NOTE: For particular method and some examples see: 
            https://docs.ros.org/en/melodic/api/pinocchio/html/namespacepinocchio.html
            https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/topic/doc-v2/doxygen-html/index.html

        """
        
        abs_path = os.path.dirname(os.path.abspath(__file__))
        urdf_path   = abs_path+"/robot_models/softleg_urdf/urdf/softleg-rlilc.urdf"
        mesh_dir    = abs_path+"/robot_models/softleg_urdf/meshes"  # Directory contenente le mesh associate al modello 
        ee_name     = "contact_joint"
        base_name   = "contact_joint"
        
        self._fs    = fs    # sensors frequency [Hz]
        dt          = 1/self._fs
        super().__init__(urdf_path=urdf_path,mesh_dir=mesh_dir,visual=visual,dt=dt,dtype=dtype,ee_name=ee_name,)
        
        #self.q0    = torch.tensor([[-2.0,1.05]]).T
        self.q0    = torch.tensor([[0.0,0.0]]).T
        self.q     = self.q0.clone()
        self.u0    = self.getGravity(self.q0).clone()
        self._uold = self.u0.clone()
        
        # limits (symmetric)
        ddqM        = 15      #[rad/s^2]
        duM         = 10      #[Nm/s]
        self._ddq_M = torch.tensor([[ddqM]]).expand(self._dim_q,1).type(self._dtype)
        self._du_M  = torch.tensor([[duM*self._dt]]).expand(self._dim_u,1).type(self._dtype) # scale dot u is not accessible
    
        self.base_pos, _ = self.getForwKinJoint(name_joint=base_name)
        

class SoftLeg_RR_disturb(Sim_RR):
    """ Simulator of SoftLeg. """
    
    def __init__(self,visual=False,dtype=torch.float32,fs=500):
        """ Class that parsering urdf.
    
        Args:
            dtype (torch.dtype, optional): type of variables in class. Defaults to torch.float32.
            visual (bool, optional): to visualize robot with meshcat. Defaults to False.
            dt (float, optional): time sample [s] for integration and visualization. Defaults to 0.01 [s].
        
        NOTE: For particular method and some examples see: 
            https://docs.ros.org/en/melodic/api/pinocchio/html/namespacepinocchio.html
            https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/topic/doc-v2/doxygen-html/index.html

        """
        urdf_path   = "./robot_models/softleg_urdf/urdf/softleg-rlilc_disturb.urdf" 
        mesh_dir    = "./robot_models/softleg_urdf/meshes"  # Directory contenente le mesh associate al modello 
        ee_name     = "contact_joint"
        base_name   = "contact_joint"
        
        self._fs    = fs    # sensors frequency [Hz]
        dt          = 1/self._fs
        super().__init__(urdf_path=urdf_path,mesh_dir=mesh_dir,visual=visual,dt=dt,dtype=dtype,ee_name=ee_name,)
        
        #self.q0    = torch.tensor([[-2.0,1.05]]).T
        self.q0    = torch.tensor([[0.0,0.0]]).T
        self.q     = self.q0.clone()
        self.u0    = self.getGravity(self.q0).clone()
        self._uold = self.u0.clone()
        
        # limits (symmetric)
        ddqM        = 15      #[rad/s^2]
        duM         = 10      #[Nm/s]
        self._ddq_M = torch.tensor([[ddqM]]).expand(self._dim_q,1).type(self._dtype)
        self._du_M  = torch.tensor([[duM*self._dt]]).expand(self._dim_u,1).type(self._dtype) # scale dot u is not accessible
    
        self.base_pos, _ = self.getForwKinJoint(name_joint=base_name)

