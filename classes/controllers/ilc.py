import torch
from tensordict import TensorDict


class ILC_base(object):
    """
    This class implements a model free ILC action ONLY for square MIMO.
    NOTE the control law used is:
    u_{j+1}(k) = u_{j}(k) + Le * e_{j}(k+1) + Ledot * de_{j}(k+1) + Leddot * dde_{j}(k+1)

    :param dimU: dimension of input vector (single step).
    :type dimU: int
    :param samples: number of samples in a single episode.
    :type samples: int
    :param Le: learning gain scalar for error. Defaults to torch.tensor(0.01).
    :type Le: torch.Tensor, optional
    :param Lde: learning gain scalar for dot error. Defaults to torch.tensor(0.0).
    :type Lde: torch.Tensor, optional
    :param Ldde: learning gain scalar for ddot error. Defaults to torch.tensor(0.0).
    :type Ldde: torch.Tensor, optional
    :param threshold: rmse threshold to stop updating ILC. Defaults to 1e-3.
    :type threshold: float, optional
    :param dtype: data type. Defaults to torch.float32.
    :type dtype: torch.dtype, optional
    """
    
    def __init__(
        self,
        dimU: int,
        samples: int,
        Le: torch.Tensor = torch.tensor(0.01),
        Lde: torch.Tensor = torch.tensor(0.0),
        Ldde: torch.Tensor = torch.tensor(0.0),
        threshold: float = 1e-3,
        dtype: torch.dtype = torch.float32,
    ):
        """
        This class implements a model free ILC action ONLY for square MIMO.
        NOTE the control law used is:
        u_{j+1}(k) = u_{j}(k) + Le * e_{j}(k+1) + Ledot * de_{j}(k+1) + Leddot * dde_{j}(k+1)

        :param dimU: dimension of input vector (single step).
        :type dimU: int
        :param samples: number of samples in a single episode.
        :type samples: int
        :param Le: learning gain scalar for error. Defaults to torch.tensor(0.01).
        :type Le: torch.Tensor, optional
        :param Lde: learning gain scalar for dot error. Defaults to torch.tensor(0.0).
        :type Lde: torch.Tensor, optional
        :param Ldde: learning gain scalar for ddot error. Defaults to torch.tensor(0.0).
        :type Ldde: torch.Tensor, optional
        :param threshold: rmse threshold to stop updating ILC. Defaults to 1e-3.
        :type threshold: float, optional
        :param dtype: data type. Defaults to torch.float32.
        :type dtype: torch.dtype, optional
        """
        
        self.dtype = dtype
        self.dimU = dimU
        self.samples = samples
        self.threshold = threshold
        self.done = False
        self.rmse = None
        self.best_u = torch.zeros(dimU, samples)
        
        if Le.flatten().size(0) == 1:
            self.Le = Le.expand(self.dimU, 1)
        else:
            if Le.size(0) == self.dimU:
                self.Le = Le.type(self.dtype)
            else:
                ValueError("Size error")
        if Lde.flatten().size(0) == 1:
            self.Lde = Lde.expand(self.dimU, 1)
        else:
            if Lde.size(0) == self.dimU:
                self.Lde = Lde.type(self.dtype)
            else:
                ValueError("Size error")
        if Ldde.flatten().size(0) == 1:
            self.Ldde = Ldde.expand(self.dimU, 1)
        else:
            if Ldde.size(0) == self.dimU:
                self.Ldde = Ldde.type(self.dtype)
            else:
                ValueError("Size error")
    
        self.uEp = torch.zeros(dimU, samples).type(self.dtype)  # control inputs for current episode
        self.uk = torch.zeros(dimU, 1).type(self.dtype)  # current control input
        self.ek = torch.zeros(dimU, 1).type(self.dtype)  # current error
        self.idx = 0  # idx step
        self.mem = []  # list of episode's memory
        self.episodes = 0  # number of episodes
            
        # data memory template (of error and input) stacked as column
        self.__tmplMem = TensorDict({
            'error': torch.Tensor(),
            'dot_error': torch.Tensor(),
            'ddot_error': torch.Tensor(),
            'input': torch.Tensor(),
        }, batch_size=[])
    
    def __updateMem(self, data: torch.Tensor, dict: str) -> None:
        """
        Store new data in a tensordict in a list. Data stacked as column.

        :param data: data as column vector
        :type data: torch.Tensor
        :param dict: string to specify a field in the tensordict.
        :type dict: str
        """
        if not isinstance(data, torch.Tensor):
            raise TypeError("data must be a torch.Tensor")
        if data.size()[1] != 1:
            raise ValueError("data must be a column vector")
        
        # use newest tensordict
        tmp_mem: torch.Tensor = self.mem[-1][dict]
        tmp_mem = torch.cat([tmp_mem.clone(), data.type(self.dtype)], dim=1)  # stack as column
        
        # update mem
        self.mem[-1][dict] = tmp_mem
    
    def __computeRMSE(self) -> None:
        """
        Compute RMSE of last episode.
        """
        e_: torch.Tensor = self.mem[-1]["error"]
        self.rmse = torch.sqrt(torch.mean(e_ ** 2))
    
    def updateMemError(self, e_: torch.Tensor, de_: torch.Tensor = None, dde_: torch.Tensor = None) -> None:
        """
        Store new error, dot error, ddot error in a tensordict in a list, used in control law.
        
        :param e_: error as column vector.
        :type e_: torch.Tensor
        :param de_: dot error as column vector. If None is set to zeros.
        :type de_: torch.Tensor, optional
        :param dde_: ddot error as column vector. If None is set to zeros.
        :type dde_: torch.Tensor, optional
        """
        
        if de_ is None:
            de_ = torch.zeros((self.dimU, 1), dtype=self.dtype)
        if dde_ is None:
            dde_ = torch.zeros((self.dimU, 1), dtype=self.dtype)
        
        str_list = ['error', 'dot_error', 'ddot_error']
        self.__updateMem(e_, str_list[0])
        self.__updateMem(de_, str_list[1])
        self.__updateMem(dde_, str_list[2])
    
    def updateMemInput(self, u_: torch.Tensor) -> None:
        """
        Store new input in a tensordict in a list, used in control law.
        
        :param u_: input as column vector.
        :type u_: torch.Tensor
        """
        
        str_ = 'input'
        self.__updateMem(u_, str_)
    
    def newEp(self) -> None:
        """
        Create new tensordict to store new data of a new episode.
        Reset step index.
        """
        self.mem.append(self.__tmplMem.clone())
        self.episodes += 1
        self.idx = 0
    
    def updateLearningGain(self, Le: torch.Tensor = None, Lde: torch.Tensor = None, Ldde: torch.Tensor = None) -> None:
        """
        Update learning gain for error, dot_error, ddot_error.

        :param Le: learning gain for error. If None not change.
        :type Le: torch.Tensor, optional
        :param Lde: learning gain for dot error. If None not change.
        :type Lde: torch.Tensor, optional
        :param Ldde: learning gain for ddot error. If None not change.
        :type Ldde: torch.Tensor, optional
        """
        
        if Le is not None:
            if Le.flatten().size(0) == 1:
                self.Le = Le.expand(self.dimU, 1)
            else:
                if Le.size(0) == self.dimU:
                    self.Le = Le.type(self.dtype)
                else:
                    ValueError("Size error")
        if Lde is not None:
            if Lde.flatten().size(0) == 1:
                self.Lde = Lde.expand(self.dimU, 1)
            else:
                if Lde.size(0) == self.dimU:
                    self.Lde = Lde.type(self.dtype)
                else:
                    ValueError("Size error")
        if Ldde is not None:
            if Ldde.flatten().size(0) == 1:
                self.Ldde = Ldde.expand(self.dimU, 1)
            else:
                if Ldde.size(0) == self.dimU:
                    self.Ldde = Ldde.type(self.dtype)
                else:
                    ValueError("Size error")

    def stepILC(self) -> None:
        """
        Update control to use in this episode.
        Start new episode.
        NOTE the control law used is:
        u_{j+1} = u_{j} + Le * e_{j} + Ledot * de_{j} + Leddot * dde_{j}
        """
        
        if len(self.mem) == 0:
            raise ValueError("ILC first episode is not initialized")
        
        Le = self.Le
        Lde = self.Lde
        Ldde = self.Ldde
        
        u_old: torch.Tensor = self.mem[-1]["input"]
        e_old: torch.Tensor = self.mem[-1]["error"]
        de_old: torch.Tensor = self.mem[-1]["dot_error"]
        dde_old: torch.Tensor = self.mem[-1]["ddot_error"]
        
        self.__computeRMSE()
        
        # condition to continue updating ILC 
        if self.rmse.item() > self.threshold:
            self.best_u = u_old \
                + torch.einsum("ij,ik->ik", Le, e_old) \
                + torch.einsum("ij,ik->ik", Lde, de_old) \
                + torch.einsum("ij,ik->ik", Ldde, dde_old)
        else:
            if not self.done:
                self.done = True
        
        self.uEp = self.best_u
        self.newEp() 
    
    def resetAll(self) -> None:
        """
        Reset flag, index and memory.
        """
        
        self.idx = 0
        self.episodes = 0
        self.mem = []
        self.done = False
        self.rmse = 0.0
    
    def getMemory(self) -> list[dict[torch.Tensor]]:
        """
        Return list of dict of error and input stacked as column.
        """
        return self.mem

    def getControl(self) -> torch.Tensor:
        """
        Return input of current step.
        """
        k = self.idx
        self.uk = self.uEp[:, k:k + 1]
        self.idx += 1
         
        return self.uk