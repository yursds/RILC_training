from abc        import ABC, abstractmethod
import torch


# Abstract class: methods that all robot must to have
class Rob_ABC(ABC):
    """ Abstract class to build robot with main methods. """

    def __init__(self,) -> None: """ Abstract class to build robot with main methods. """

    # METHODS FOR ROBOT'S KINEMATICS
    @abstractmethod
    def _computeForwKin(self, q:torch.Tensor) -> torch.Tensor: """ Compute forward kinematic of frames. """
    @abstractmethod
    def _computeJacobians(self, q:torch.Tensor) -> torch.Tensor: """ Compute jacobian of all joints. """
    
    # METHODS FOR ROBOT'S DYNAMICS
    @abstractmethod
    def getMass(self, q:torch.Tensor) -> torch.Tensor: """ Return Mass Matrix. """
    @abstractmethod
    def getInvMass(self, q:torch.Tensor) -> torch.Tensor: """ Return Inverse Mass Matrix. """
    @abstractmethod
    def getCoriolis(self, q:torch.Tensor, dq:torch.Tensor) -> torch.Tensor: """ Return Coriolis Matrix. """
    @abstractmethod
    def getCoriolisVec(self, q:torch.Tensor, dq:torch.Tensor) -> torch.Tensor: """ Return Coriolis Vector Matrix. """
    @abstractmethod
    def getGravity(self, q:torch.Tensor) -> torch.Tensor: """ Return Generalized Gravity Matrix. """
    @abstractmethod
    def getDamping(self,) -> torch.Tensor: """ Return Damping Matrix. """
    @abstractmethod
    def getForwDyn(self, q:torch.Tensor,dq:torch.Tensor, u0:torch.Tensor) -> torch.Tensor: """ Get forward dynamic (dot state). """
    @abstractmethod
    def getInvDyn(self, q:torch.Tensor,dq:torch.Tensor,ddq:torch.Tensor) -> torch.Tensor: """Get inverse dynamic (tau). """
                   
    # USEFUL METHODS
    @abstractmethod
    def getState(self) -> torch.Tensor: """ Get state [q,dq] """
    @abstractmethod
    def getDotState(self) -> torch.Tensor: """ Get dot state [dq,ddq] """
    @abstractmethod
    def setState(self,q:torch.Tensor, dq:torch.Tensor, ddq:torch.Tensor) -> None: """ Set variables. """
    @abstractmethod
    def render(self,) -> None: """ Visualization. """


# Abstract Derived class: that built robot from URDF using Pinocchio
class RobURDF_ABC(Rob_ABC):
    """ Abstract class to build robot with pinocchio library (parsering URDF). """
    
    def __init__(self,) -> None:
        """ Abstract class to build robot with main methods. """
        super().__init__()

    # USEFUL METHODS
    @abstractmethod
    def _getFramesName(self) -> list[str]: """ List of frames' name used in URDF. """
    @abstractmethod
    def _getFramesDist(self, q:torch.Tensor) -> dict[torch.Tensor]: """ Dict of frames' name and their distance respect world frame. """
    @abstractmethod
    def _getLengthLink(self) -> list[torch.Tensor]: """ List of link length. """
    @abstractmethod
    def _render_axes(self) -> None: """ Frames visualization. """


# Abstract Derived class: that built robot from URDF and considering saturation.
class SimRobURDF_ABC(RobURDF_ABC):
    """ Abstract class to build simulator robot with main methods. """

    def __init__(self) -> None:
        """ Abstract class to build simulator robot with main methods. """
        super().__init__()
    
    @abstractmethod
    def _extractLimits(self) -> None: "Extraction of limits of q, dq, ddq, u, du. (dq, ddq, u, du have symmetric ranges)"       
    @abstractmethod
    def _saturatedq(self, dq:torch.Tensor)->torch.Tensor: """ Saturate dq. """
    @abstractmethod
    def _saturateddq(self, ddq:torch.Tensor)->torch.Tensor: """ Saturate ddq. """
    @abstractmethod
    def _saturateu(self, u:torch.Tensor)->torch.Tensor: """ Saturate u. """
 

