import torch

from abc        import ABC, abstractmethod


class CartesianRef_ABC(ABC):
    """ Abstract class to generate cartesian reference. """
    @abstractmethod
    def __init__(self,) -> None: 
        """ Abstract class to generate cartesian reference. """
        pass

    @abstractmethod
    def _get_x_value(self, t:torch.Tensor) -> torch.Tensor: 
        """ Get x reference at step t. """
        pass

    @abstractmethod
    def _get_y_value(self, t:torch.Tensor) -> torch.Tensor: 
        """ Get y reference at step t. """
        pass

    @abstractmethod
    def _get_z_value(self, t:torch.Tensor) -> torch.Tensor: 
        """ Get z reference at step t. """
        pass

    @abstractmethod
    def _computeRef(self,) -> None: 
        """ Compute reference as a torch.Tensor. """
        pass

    @abstractmethod
    def getRef(self,) -> torch.Tensor: 
        """ Get reference as a torch.Tensor. """
        pass

    @abstractmethod
    def getDictRef(self,) -> dict[torch.Tensor]: 
        """ Get a dict of torch.Tensor of reference. """
        pass

    @abstractmethod
    def getResampleRef(self, samples:int) -> torch.Tensor: 
        """ Resample reference. """
        pass

    @abstractmethod
    def setParams(self,) -> None: 
        """ Update reference parameters. """
        pass

    @abstractmethod
    def plotRef(self,) -> None: 
        """ Plot reference. """
        pass


class JointRef_ABC(ABC):
    """ Abstract class to generate cartesian reference. """
    @abstractmethod
    def __init__(self,) -> None: 
        """ Abstract class to generate cartesian reference. """
        pass

    @abstractmethod
    def _computeRef(self,) -> None: 
        """ Compute reference as a torch.Tensor. """
        pass

    @abstractmethod
    def getRef(self,) -> torch.Tensor: 
        """ Get reference as a torch.Tensor. """
        pass

    @abstractmethod
    def getDictRef(self,) -> dict[torch.Tensor]: 
        """ Get a dict of torch.Tensor of reference. """
        pass

    @abstractmethod
    def getResampleRef(self, samples:int) -> torch.Tensor: 
        """ Resample reference. """
        pass

    @abstractmethod
    def setParams(self,) -> None: 
        """ Update reference parameters. """
        pass

    @abstractmethod
    def plotRef(self,) -> None: 
        """ Plot reference. """
        pass


