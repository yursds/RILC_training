import torch

from classes.references.ABC_reference   import *
from classes.robots.builder_manipulator import ManURDF

from matplotlib                 import pyplot       as plt
from torch.nn                   import functional   as F



class MinJerkRef(JointRef_ABC):
    """ Definition of joints reference using minjerk. """
    
    def __init__(self,q0:torch.Tensor,qf:torch.Tensor=None,duration=3.0,dt=0.01,stayT=0.5) -> None:
        """ Definition of joints reference using minjerk.
    
        Args:
            robot (ManipulatorURDF): manipulator robot class.
            gaitT (float, optional): Duration of gait. Defaults to 3.0.
            ratio (float, optional): Ratio between swing and gait duration. Defaults to 0.5.
            t_off (int, optional): Time offset respect default gait sequence. Defaults to 0.0.
            length (float, optional): Step length. Defaults to 0.15.
            height (float, optional): Max step height. Defaults to 0.05.
        """
        
        super().__init__()
        
        if stayT >= duration:
            stayT = 0.0
            Warning("StayT is bigger than duration of task.")
            
        if isinstance(duration, torch.Tensor):
            duration = duration.item()
        
        samplesStay     = torch.floor(torch.tensor(stayT/dt, dtype=torch.int32)).item()
        samplesMinj     = torch.floor(torch.tensor(duration/dt, dtype=torch.int32)).item() - samplesStay
        self._duration  = samplesMinj*dt
        self._dt        = dt
        self._samplesM  = samplesMinj
        self._samplesS  = samplesStay
        self.samples    = samplesMinj+samplesStay
        
        self._qi        = self.angle_normalize(q0)
        if qf is None:
            qf = q0
        self._qf        = self.angle_normalize(qf)
        dim_q           = q0.size(0)
        self._dim_q     = dim_q
        
        num_derivative  = 3
        self._refTensor = torch.zeros(dim_q,num_derivative,self.samples)
        self._refDict   = {}
        for i in range(dim_q):
            self._refDict[f"q{i+1}"] = torch.zeros(num_derivative,self.samples)
        
        self._computeRef()
        self.jointRef   = self._refTensor
    
    def _minjerk(self,qi:torch.Tensor,qf:torch.Tensor,duration:float,t:float) -> list[torch.Tensor,torch.Tensor,torch.Tensor]:

        delta_q = qi-qf
        q_new   = qi + delta_q * (15*(t/duration)**4 - 6*(t/duration)**5 - 10*(t/duration)**3)
        dq_new  = delta_q * (60*(t**3)/(duration**4) - 30*((t**4)/(duration**5)) - 30*(t**2)/(duration**3))
        ddq_new = delta_q * (180*(t**2)/(duration**4) - 120*((t**3)/(duration**5)) - 60*(t/(duration**3)))
        
        return q_new, dq_new, ddq_new
    
    def _computeRef(self) -> None:
        
        qi  = self._qi
        qf  = self._qf
        
        deltaT  = self._duration
        samples = self._samplesM
        
        tmp = torch.zeros(self._refTensor.size(0), self._refTensor.size(1), samples)  
        t_vec = torch.linspace(0,deltaT,samples)
        for i, t in enumerate(t_vec):
            q,dq,ddq = self._minjerk(qi,qf,deltaT,t)
            tmp[:,:,i] = torch.cat([q,dq,ddq], dim=1)
        
        stay_qf = torch.zeros(self._refTensor.size(0), self._refTensor.size(1), self._samplesS)        
        stay_qf[:,0,:] += qf
        
        self._refTensor = torch.cat([tmp, stay_qf],dim=2)
        self.jointRef   = self._refTensor
        
    def _resample(self, ref:torch.Tensor, numsamples) -> list[torch.Tensor,float]:
        """ 
        Get resample reference as a torch.Tensor and new dt. The index of tensor is [axis,derivative,time]. 
        
        Args:
            numsamples (int): number of new samples. Default to 100..
        """
        ref0 = ref
        ref1 = torch.zeros(ref0.size(0), ref0.size(1), numsamples)
        for i in range(ref0.size(0)):
            tens = ref0[i,:,:].unsqueeze(0)
            ref_:torch.Tensor = F.interpolate(tens, size=numsamples, mode='linear', align_corners=False)
            ref_ = ref_.squeeze(0)
            ref1[i,:,:] = ref_
        
        return ref1
    
    def getRef(self) -> torch.Tensor:
        """ [axis, derivative, time] """
        self._computeRef()
        return self.jointRef          
    
    def plotRef(self, plot_now=False):
        
        joint_tensor = self._refTensor
        dim = joint_tensor.size(2)
        labels = ["Position", "Velocity", "Acceleration"]
        units_j = ["[$rad$]", "[$rad/s$]", "[$rad/s^2$]"]
        
        for index in range(self._dim_q):
            plt.figure(figsize=(8, 6))
            for i, label in enumerate(labels):
                plt.subplot(len(labels), 1, i+1)
                plt.plot(torch.arange(dim)*self._dt, joint_tensor[index, i, :], label=f"q{index+1} - {label}")
                plt.xlabel("Tempo [s]")
                plt.ylabel(label+" "+units_j[i])
                plt.grid()
                plt.legend()
            plt.suptitle(f"Joint {index+1} Reference")
            plt.tight_layout()
            
        if plot_now:
            plt.show()    
    
    def getDictRef(self,) -> dict[torch.Tensor]: 
        """ Get reference as a dictionary of torch.Tensor. \
            The keys are "q_i", the index of tensor is [derivative,time]. """
        
        tensor = self._refTensor
        for i in range(self._dim_q):
            self._refDict[f"q{i+1}"] = tensor[i,:,:]    
        return self._refDict
    
    def getResampleRef(self, numsamples:int) -> torch.Tensor: 
        """ 
        Get resample reference as a torch.Tensor and new dt. The index of tensor is [axis,derivative,time]. 
        
        Args:
            numsamples (int): number of new samples. Default to 100..
        """
        
        ts0     = self._refTensor
        ts1     = self._resample(ts0,numsamples)
        
        return ts1
    
    def setParams(self,) -> None:
        """ Update parameters of gait reference that are not None. """
        pass
    
    def angle_normalize(self, x:torch.Tensor)->torch.Tensor:
        """ angle in range [-pi; pi]"""
        
        sx = torch.sin(x)
        cx = torch.cos(x)
        x = torch.atan2(sx,cx)
        return x


class MinSnapRef(MinJerkRef):
    """ Definition of joints reference using minjerk. """
        
    def __init__(self,q0:torch.Tensor,qf:torch.Tensor=None,duration=3.0,dt=0.01,stayT=0.5) -> None:
        
        super().__init__(q0=q0,qf=qf,duration=duration,dt=dt,stayT=stayT)
    
    def _minsnap(self,qi:torch.Tensor,qf:torch.Tensor,duration:float,t:float) -> list[torch.Tensor,torch.Tensor,torch.Tensor]:
        c1 = -70
        c2 = -20
        c3 = -84
        c4 = 35
        d2 = 7
        d1 = d2-1
        d3 = d2-2
        d4 = d2-3
        delta_q = qi-qf
        q_new   = qi + delta_q * (c1 * (t/duration)**d1 - c2 * (t/duration)**d2 - c3 * (t/duration)**d3 - c4 * (t/duration)**d4)
        dq_new  = delta_q * (c1*d1 * (t**(d1-1))/(duration**d1) - c2*d2 * (t**(d2-1))/(duration**d2) - c3*d3 * (t**(d3-1))/(duration**d3) - c4*d4 * (t**(d4-1))/(duration**d4))
        ddq_new = delta_q * (c1*d1*(d1-1) * (t**(d1-2))/(duration**d1) - c2*d2*(d2-1) * (t**(d2-2))/(duration**d2) - c3*d3*(d3-1) * (t**(d3-2))/(duration**d3) - c4*d4*(d4-1) * (t**(d4-2))/(duration**d4))
        
        return q_new, dq_new, ddq_new

    def _computeRef(self) -> None:
        
        qi  = self._qi
        qf  = self._qf
        
        deltaT  = self._duration
        samples = self._samplesM
        
        tmp = torch.zeros(self._refTensor.size(0), self._refTensor.size(1), samples)  
        t_vec = torch.linspace(0,deltaT,samples)
        for i, t in enumerate(t_vec):
            q,dq,ddq = self._minsnap(qi,qf,deltaT,t)
            tmp[:,:,i] = torch.cat([q,dq,ddq], dim=1)
            
        stay_qf = torch.zeros(self._refTensor.size(0), self._refTensor.size(1), self._samplesS)        
        stay_qf[:,0,:] += qf
        
        self._refTensor = torch.cat([tmp, stay_qf],dim=2)
        self.jointRef   = self._refTensor


class InvKin(ABC):
    
    def __init__(self,robot:ManURDF,pf:torch.Tensor=None,) -> None:
        """ Inverse Kinematic of Manipulator robot.
        Args:
            robot (ManipulatorURDF): manipulator robot class.
            pf (torch.Tensor, optional): Desired position of end-effector.
        """
        
        super().__init__()
        pi      = robot.getForwKinEE(robot.q0)[0]
        if pf is None:
            pf = pi
        self.qi         = robot.q0
        self.qf         = robot.q0
        self._pi        = pi
        self._pf        = pf
        self._dim_q     = robot._dim_q
        self.__robot    = robot
        self.eps        = 1e-5
        self.it_MAX     = 1000
        
        # swing phase
        self._computeInvKin()
    
    def _clik(self, q:torch.Tensor, dt = 0.01, kp=1.0) -> list[torch.Tensor,torch.Tensor]:

        pos_cur = self.__robot.getForwKinEE(q)[0]
        
        err     = self._pf - pos_cur
        tmp_v   = 0 + kp*err
        
        # RK4 integration
        k1v = torch.matmul(self.__robot.getPinvJacPosEE(q=q), tmp_v)
        k2v = torch.matmul(self.__robot.getPinvJacPosEE(q=q + k1v * dt / 2), tmp_v)
        k3v = torch.matmul(self.__robot.getPinvJacPosEE(q=q + k2v * dt / 2), tmp_v)
        k4v = torch.matmul(self.__robot.getPinvJacPosEE(q=q + k3v * dt), tmp_v)
        new_dq = (k1v + 2 * (k2v + k3v) + k4v) / 6
        
        new_q   = q + dt * new_dq
        
        return new_q, torch.sqrt((err**2).mean())
 
    def _computeInvKin(self, dt = 0.01, kp=1.0) -> None:
        
        q   = self.qi
        eps = self.eps
        iter_max = self.it_MAX
        
        for _ in range(iter_max):
            
            q, err = self._clik(q=q, dt=dt, kp=kp)
            if err <= eps:        
                break
            
        self.qf = q
    
    def get_q(self) -> torch.Tensor:
        """ [axis, derivative, time] """

        return self.qf          
    
    def setParams(self,qi:torch.Tensor=None, pf:torch.Tensor=None) -> None:
        """ Update parameters that are not None. """
        if qi is not None:
            self.qi = qi
        if pf is not None:
            self._pf = pf


class RefInvKin(MinSnapRef):

    def __init__(self,robot:ManURDF,pf:torch.Tensor=None,duration=3.0,stayT=0.1,dt:float=None):
        
        inv_kin = InvKin(robot=robot, pf=pf)
        q0 = self.angle_normalize(robot.q0)
        qf = self.angle_normalize(inv_kin.get_q())
        if dt == None:
            dt = robot._dt
        self.__robot = robot
        super().__init__(q0=q0,qf=qf,duration=duration,dt=dt,stayT=stayT)
        self.jointRef[:,0,:] = self.angle_normalize(self.jointRef[:,0,:])
    
    def setParams(self,qi:torch.Tensor=None, pf:torch.Tensor=None) -> None:
        """ Update parameters that are not None. """
        if qi is not None:
            self.qi = self.angle_normalize(qi)
        if pf is not None:
            self._pf = pf
            inv_kin = InvKin(robot=self.__robot, pf=pf)
            qf = inv_kin.get_q()
            self._qf = self.angle_normalize(qf)






# if __name__ == '__main__':
        
#     from sympy import symbols, Eq, solve

#     x, y, z, w = symbols('x y z w')
#     d2 = 7
#     d1 = d2-1
#     d3 = d2-2
#     d4 = d2-3
    
#     eq1 = Eq(x - y - z - w, -1)
#     eq2 = Eq(x*(d1) - y*(d2) - z*(d3) - w*(d4), 0)
#     eq3 = Eq(x*(d1)*(d1-1) - y*(d2)*(d2-1) - z*(d3)*(d3-1) - w*(d4)*(d4-1), 0)
#     eq4 = Eq(x*(d1)*(d1-1)*(d1-2) - y*(d2)*(d2-1)*(d2-2) - z*(d3)*(d3-1)*(d3-2) - w*(d4)*(d4-1)*(d4-2), 0)
    
#     sol = solve((eq1,eq2,eq3, eq4), (x, y, z, w))

#     print(sol)
    
#     from ..robots.softleg         import SoftLeg_RR
#     rob      = SoftLeg_RR(visual=True)
#     rob.render()
    
#     pf = torch.tensor([[0.5,0.0]]).T
#     a = InvKin(rob, pf)
    
#     qf = a.get_q()
    
#     rob.setState(q=qf)
#     rob.render()
    
#     rob = SoftLeg_RR(visual=True)
#     a = RefInvKin(rob, pf)
#     sampl = a.samples
#     ref = a.getRef()
#     q_ref = ref[:,0,:]
#     a.plotRef(plot_now=True)
#     for i in range(sampl):
        
#         rob.setState(q=q_ref[:,i:i+1])
#         rob.render()