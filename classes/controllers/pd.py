import torch


class PD_base(object):
    """
    This class implements a basic Proportional-Derivative (PD) controller.

    :param dimU: Dimension of the input vector.
    :type dimU: int
    :param kp: Proportional gain.
    :type kp: float
    :param kv: Derivative gain.
    :type kv: float
    :param dtype: Data type. Defaults to torch.float32.
    :type dtype: torch.dtype, optional
    """
    
    def __init__(self, dimU: int, kp: float, kv: float, dtype: torch.dtype = torch.float32):
        """
        Initialize the PD controller with given parameters.

        :param dimU: Dimension of the input vector.
        :type dimU: int
        :param kp: Proportional gain.
        :type kp: float
        :param kv: Derivative gain.
        :type kv: float
        :param dtype: Data type. Defaults to torch.float32.
        :type dtype: torch.dtype, optional
        """
        self.dtype = dtype
        self.dimU = dimU
        self.Kp = torch.diag(torch.tensor([kp for _ in range(dimU)], dtype=dtype))
        self.Kv = torch.diag(torch.tensor([kv for _ in range(dimU)], dtype=dtype))

    def getControl(self, e_: torch.Tensor, de_: torch.Tensor) -> torch.Tensor:
        """
        Compute the control input based on the error and derivative of error.

        :param e_: Error as a column vector.
        :type e_: torch.Tensor
        :param de_: Derivative of error as a column vector.
        :type de_: torch.Tensor
        :return: Control input.
        :rtype: torch.Tensor
        """
        e_ = e_.type(self.dtype)
        de_ = de_.type(self.dtype)
        
        u = torch.matmul(self.Kp, e_) + torch.matmul(self.Kv, de_)
        
        return u
    
    def setParams(self, Kp: torch.Tensor = None, Kd: torch.Tensor = None):
        """
        Set the parameters of the PD controller.

        :param Kp: Proportional gain matrix. If None, it is not changed.
        :type Kp: torch.Tensor, optional
        :param Kd: Derivative gain matrix. If None, it is not changed.
        :type Kd: torch.Tensor, optional
        """
        if Kp is not None:
            self.Kp = Kp
        
        if Kd is not None:
            self.Kd = Kd