from classes.robots.ABC_robot_ import*


# Abstract Derived class: that built Manipulator robot (focus on end-effector methods)
class Manipulator_ABC(Rob_ABC):
    """ Abstract class to build Manipulator robot with main methods. """
    
    def __init__(self,) -> None:
        """ Abstract class to build Manipulator robot with main methods. """
        super().__init__()
    
    # METHODS FOR ROBOT'S KINEMATICS
    @abstractmethod
    def getForwKinEE(self, q:torch.Tensor) -> torch.Tensor: 
        """ Get forward kinematic of end effector frame. """
        pass

    @abstractmethod
    def getJacEE(self, q:torch.Tensor) -> torch.Tensor: 
        """ Return Jacobian of end-effector frame. """
        pass

    @abstractmethod
    def getJacPosEE(self, q:torch.Tensor) -> torch.Tensor: 
        """ Return Jacobian for Position of end-effector frame. """
        pass

    @abstractmethod
    def getDotJacEE(self, q:torch.Tensor, dq:torch.Tensor) -> torch.Tensor: 
        """ Return Jacobian Derivative of end-effector frame. """
        pass

    @abstractmethod
    def getDotJacPosEE(self, q:torch.Tensor, dq:torch.Tensor) -> torch.Tensor: 
        """ Return Jacobian Derivative for Position of end-effector frame. """
        pass

    @abstractmethod
    def getPinvJacEE(self, q:torch.Tensor, mu:float) -> torch.Tensor: 
        """ Return Pseudo Inverse of Jacobian """
        pass

    @abstractmethod
    def getPinvJacPosEE(self, q:torch.Tensor, mu:float) -> torch.Tensor: 
        """ Return Pseudo Inverse of Jacobian for Position of end-effector frame. """
        pass

    @abstractmethod
    def getPinvDotJacEE(self, q:torch.Tensor, dq:torch.Tensor, mu:float) -> torch.Tensor: 
        """ Return Pseudo Inverse of Jacobian Derivative of end-effector frame. """
        pass

    @abstractmethod
    def getPinvDotJacPosEE(self, q:torch.Tensor, dq:torch.Tensor, mu:float) -> torch.Tensor: 
        """ Return Pseudo Inverse of Jacobian Derivative for Position of end-effector frame. """ 
        pass 


# Abstract Derived class: that built Manipulator robot (focus on end-effector methods)
class ManURDF_ABC(RobURDF_ABC, Manipulator_ABC):
    """ Abstract class to build Manipulator robot with main methods. """
    
    def __init__(self,) -> None:
        """ Abstract class to build Manipulator robot with main methods. """
        super().__init__()


# Abstract Derived class: that built Manipulator robot from URDF
class SimManURDF_ABC(SimRobURDF_ABC, Manipulator_ABC):
    """ Abstract class to build Manipulator robot with main methods. """
    
    def __init__(self,) -> None:
        """ Abstract class to build Manipulator robot with main methods. """
        super().__init__()



