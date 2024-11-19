import torch

from abc        import ABC, abstractmethod


class CartesianRef_ABC(ABC):
    """ Abstract class to generate cartesian reference. """
    @abstractmethod
    def __init__(self,) -> None: """ Abstract class to generate cartesian reference. """
    @abstractmethod
    def _get_x_value(self, t:torch.Tensor) -> torch.Tensor: """ Get x reference at step t. """
    @abstractmethod
    def _get_y_value(self, t:torch.Tensor) -> torch.Tensor: """ Get y reference at step t. """
    @abstractmethod
    def _get_z_value(self, t:torch.Tensor) -> torch.Tensor: """ Get z reference at step t. """
    @abstractmethod
    def _computeRef(self,) -> None: """ Compute reference as a torch.Tensor. """
    @abstractmethod
    def getRef(self,) -> torch.Tensor: """ Get reference as a torch.Tensor. """
    @abstractmethod
    def getDictRef(self,) -> dict[torch.Tensor]: """ Get a dict of torch.Tensor of reference. """
    @abstractmethod
    def getResampleRef(self, samples:int) -> torch.Tensor: """ Resample reference. """
    @abstractmethod
    def setParams(self,) -> None: """ Update reference parameters. """
    @abstractmethod
    def plotRef(self,) -> None: """ Plot reference. """


class JointRef_ABC(ABC):
    """ Abstract class to generate cartesian reference. """
    @abstractmethod
    def __init__(self,) -> None: """ Abstract class to generate cartesian reference. """
    @abstractmethod
    def _computeRef(self,) -> None: """ Compute reference as a torch.Tensor. """
    @abstractmethod
    def getRef(self,) -> torch.Tensor: """ Get reference as a torch.Tensor. """
    @abstractmethod
    def getDictRef(self,) -> dict[torch.Tensor]: """ Get a dict of torch.Tensor of reference. """
    @abstractmethod
    def getResampleRef(self, samples:int) -> torch.Tensor: """ Resample reference. """
    @abstractmethod
    def setParams(self,) -> None: """ Update reference parameters. """
    @abstractmethod
    def plotRef(self,) -> None: """ Plot reference. """


