from classes.robots.ABC_robot_man   import ManURDF_ABC, SimManURDF_ABC
from classes.robots.builder_robot   import RobURDF, SimRobURDF

import pinocchio            as pin
import torch


# Main methods implementation
class ManURDF(RobURDF, ManURDF_ABC):
    """ Class that parsering urdf of manipulator with pinocchio library and wrapping with torch library. """
        
    def __init__(self,urdf_path:str,mesh_dir:str=None,dt=0.01,visual=False,dtype=torch.float32,ee_name="end_effector",) -> None:
        """ Class that parsering urdf of manipulator with pinocchio library and wrapping with torch library.
   
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
        
        super().__init__(urdf_path=urdf_path,mesh_dir=mesh_dir,dt=dt,visual=visual,dtype=dtype,)
        self.ee_name    = ee_name
        
    # METHODS FOR ROBOT'S KINEMATICS   
    def getAnalyticalLinearizedDynamics(self, q: torch.Tensor, dq: torch.Tensor, u: torch.Tensor, damp_fl: bool = True) -> list[torch.Tensor, torch.Tensor]:
        """
        Compute continuous-time linearized dynamics matrices A, B using Pinocchio analytical derivatives.
        dx = A x + B u
        where x = [q, dq]^T
        
        Args:
            q, dq, u: Current state and input (torch.Tensor)
            damp_fl: Whether to include damping D in the model
            
        Returns:
            A (2n x 2n), B (2n x n) matrices of the continuous system
        """
        q_np = q.flatten().detach().numpy()
        dq_np = dq.flatten().detach().numpy()
        u_np = u.flatten().detach().numpy()
        
        # Calculate tau_net applied to the rigid body (subtracting damping if needed)
        # The equation is: M ddq + C dq + G = u - D dq
        # So tau_net = u - D dq
        
        tau_net = u_np.copy()
        D = None
        if damp_fl:
            D = torch.diag(torch.from_numpy(self.robModel.damping)).numpy()
            tau_net -= D @ dq_np

        # Compute derivatives of Forward Dynamics (ddq)
        # dacc/dq, dacc/dv, dacc/dtau
        
        # Note: pinocchio.computeABADerivatives takes (model, data, q, v, tau)
        pin.computeABADerivatives(self.robModel, self.robData, q_np, dq_np, tau_net)
        
        dacc_dq = torch.from_numpy(self.robData.ddq_dq).type(self._dtype)
        dacc_dv = torch.from_numpy(self.robData.ddq_dv).type(self._dtype)
        dacc_dtau = torch.from_numpy(self.robData.Minv).type(self._dtype) 
        
        # Adjust dacc_dv for damping term:
        # ddq = f_pin(q, dq, u - D dq)
        # d(ddq)/d(dq) = d(f_pin)/d(v) + d(f_pin)/d(tau_in) * d(tau_in)/d(dq)
        # d(tau_in)/d(dq) = -D
        
        if damp_fl:
             dacc_dv_total = dacc_dv + torch.matmul(dacc_dtau, -torch.from_numpy(D).type(self._dtype))
        else:
             dacc_dv_total = dacc_dv
             
        # Construct A and B
        # A = [[0, I], [dacc_dq, dacc_dv_total]]
        # B = [[0], [dacc_dtau]]
        
        n = self._dim_q
        A = torch.zeros(2*n, 2*n).type(self._dtype)
        B = torch.zeros(2*n, n).type(self._dtype)
        
        A[:n, n:] = torch.eye(n).type(self._dtype)
        A[n:, :n] = dacc_dq
        A[n:, n:] = dacc_dv_total
        B[n:, :] = dacc_dtau
        
        return A, B

    def getForwKinEE(self, q:torch.Tensor=None,) -> list[torch.Tensor, torch.Tensor]:
        """ Compute forward kinematic of frame end effector. [trasl, rot] """
        
        frameName = self.ee_name 
        self._computeForwKin(q)
        
        fnameID:int = self.robModel.getFrameId(frameName)
        Tee = self.robData.oMf[fnameID]
        traslEE = torch.from_numpy(Tee.translation).view(-1,1).type(self._dtype)
        rotEE = torch.from_numpy(Tee.rotation).type(self._dtype)
        
        return traslEE, rotEE

    def getForwKinJoint(self,  name_joint:str, q:torch.Tensor=None) -> list[torch.Tensor, torch.Tensor]:
        """ Compute forward kinematic of frame joint with name = name_joint. [trasl, rot] """
        
        frameName = name_joint
        self._computeForwKin(q)
        
        fnameID:int = self.robModel.getFrameId(frameName)
        Tjoint      = self.robData.oMf[fnameID]
        trasl_j     = torch.from_numpy(Tjoint.translation).view(-1,1).type(self._dtype)
        rot_j       = torch.from_numpy(Tjoint.rotation).type(self._dtype)
        
        return trasl_j, rot_j

    def getJacEE(self, q:torch.Tensor=None,) -> torch.Tensor:
        """ Return Jacobian of end-effector frame.

        Args:
            q (torch.Tensor, optional): joint position variables. Defaults to None, it is set to self.q.
        """
        frameName = self.ee_name 
        self._computeJacobians(q)
        
        frame_id = self.robModel.getFrameId(frameName)
        J = torch.from_numpy(pin.getFrameJacobian(self.robModel, self.robData, frame_id, pin.LOCAL_WORLD_ALIGNED)).type(self._dtype)

        return J
  
    def getJacPosEE(self, q:torch.Tensor=None,) -> torch.Tensor:
        """ Return Jacobian for Position
         
        Args:
             q (torch.Tensor, optional): joint position variables. Defaults to None, it is set to self.q.
        """  
        J = self.getJacEE(q)
        Jpos = J[0:3,:]
        return Jpos
         
    def getPinvJacEE(self, q:torch.Tensor=None, mu = 1e-16) -> torch.Tensor:
        """ Return Pseudo Inverse of Jacobian 
         
        Args:
             q (torch.Tensor, optional): joint position variables. Defaults to None, it is set to self.q.
        """ 
        
        J = self.getJacEE(q)
        JT = J.transpose(0, 1)
        invJ = torch.inverse(torch.matmul(J,JT) + torch.eye(J.size(0))*mu).type(self._dtype)
        pinvJ = torch.matmul(JT,invJ)
        
        return pinvJ
    
    def getPinvJacPosEE(self, q:torch.Tensor=None, mu = 1e-16) -> torch.Tensor:
        """ Return Pseudo Inverse of Jacobian for Position
         
        Args:
             q (torch.Tensor, optional): joint position variables. Defaults to None, it is set to self.q.
        """   
        Jpos = self.getJacPosEE(q)
        JposT = Jpos.transpose(0, 1)
        invJpos = torch.inverse(torch.matmul(Jpos,JposT) + torch.eye(Jpos.size(0))*mu).type(self._dtype)
        pinvJpos = torch.matmul(JposT,invJpos)
        
        return pinvJpos
   
    def getDotJacEE(self, q:torch.Tensor=None, dq:torch.Tensor=None) -> torch.Tensor:
        """ Return Jacobian Derivative of end-effector frame.

        Args:
            q (torch.Tensor, optional): joint position variables. Defaults to None, it is set to self.q.
            dq (torch.Tensor, optional): joint velocity variables. Defaults to None, it is set to self.dq.
        """
        frameName = self.ee_name 
        self._computeDotJacobians(q,dq)
        
        frame_id = self.robModel.getFrameId(frameName)
        dJ = torch.from_numpy(pin.getFrameJacobianTimeVariation(self.robModel, self.robData, frame_id, pin.LOCAL_WORLD_ALIGNED)).type(self._dtype)

        return dJ
    
    def getDotJacPosEE(self, q:torch.Tensor=None, dq:torch.Tensor=None) -> torch.Tensor:
        """ Return Jacobian for Position of end-effector frame.
         
        Args:
             q (torch.Tensor, optional): joint position variables. Defaults to None, it is set to self.q.
             dq (torch.Tensor, optional): joint velocity variables. Defaults to None, it is set to self.dq.
        """ 
        
        dJ = self.getDotJacEE(q, dq)
        dJpos = dJ[0:3,:]
        return dJpos
     
    def getPinvDotJacEE(self, q:torch.Tensor=None, dq:torch.Tensor=None, mu = 1e-16) -> torch.Tensor:
        """ Return Pseudo Inverse of Jacobian Derivative of end-effector frame.
         
        Args:
             q (torch.Tensor, optional): joint position variables. Defaults to None, it is set to self.q.
             dq (torch.Tensor, optional): joint velocity variables. Defaults to None, it is set to self.dq.
        """ 
            
        dJ = self.getDotJacEE(q, dq)
        dJT = dJ.transpose(0, 1)
        invdJ = torch.inverse(torch.matmul(dJ,dJT) + torch.eye(dJ.size(0))*mu).type(self._dtype)
        pinvdJ = torch.matmul(dJT,invdJ)
        
        return pinvdJ
 
    def getPinvDotJacPosEE(self, q:torch.Tensor=None, dq:torch.Tensor=None, mu = 1e-16) -> torch.Tensor:
        """ Return Pseudo Inverse of Jacobian Derivative for Position of end-effector frame.
         
        Args:
             q (torch.Tensor, optional): joint position variables. Defaults to None, it is set to self.q.
             dq (torch.Tensor, optional): joint velocity variables. Defaults to None, it is set to self.dq.
        """ 
            
        dJpos = self.getDotJacPosEE(q, dq)
        dJposT = dJpos.transpose(0, 1)
        invdJpos = torch.inverse(torch.matmul(dJpos,dJposT) + torch.eye(dJpos.size(0))*mu).type(self._dtype)
        pinvdJpos = torch.matmul(dJposT,invdJpos)
        
        return pinvdJpos
    
    # USEFUL METHODS
    def angle_normalize(self, x:torch.Tensor)->torch.Tensor:
        """ angle in range [-pi; pi]"""
        
        sx = torch.sin(x)
        cx = torch.cos(x)
        x = torch.atan2(sx,cx)
        return x
 
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
            q = self.angle_normalize(q.type(self._dtype))
            self.q = q.clone()
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


# Main methods for robot simulator implementation
class SimManURDF(SimRobURDF, ManURDF):
    """ Class that parsering urdf of "realistic" manipulator. """
        
    def __init__(self,urdf_path:str,mesh_dir:str,dt=0.01,visual=False,dtype=torch.float32,ee_name="end_effector",) -> None:
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

    def getNewState(self, action:torch.Tensor, dt:float = None, damp_fl=True) -> list[torch.Tensor, torch.Tensor]:
            """
            Return state [q,dq]^T variables wrt robot dynamics.

            Args:
                action (torch.Tensor, optional): input to system. If None it is set to zero.
            Returns:
                list[torch.Tensor, torch.Tensor]: [self.q, self.dq]
            """
            
            self._updateState(u0=action, dt=dt, damp_fl=damp_fl)
            self.q = self.angle_normalize(self.q)
            return [self.q, self.dq]

