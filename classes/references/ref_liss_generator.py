import __init__

# NUMPY & TORCH & GYMNASIUM
import torch
import numpy                as np
import os
import mujoco

# MY CLASSES: ROBOT
from classes.robots.manipulator_RR  import Sim_RR

abs_path  = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # classes_folder
URDF_PATH = os.path.join(abs_path,'robots/robot_models/softleg_urdf/urdf/leg_constrained.urdf')
MJC_PATH  = os.path.join(abs_path,'robots/robot_models/softleg_urdf/mjc/scene_test.xml')

class LISS_GEN(object):

    def __init__(
        self,
        taskT : float       = 1.0,
        f_robot : float     = 500,
        ):
        
        self._load_pin_robot(urdf_path=URDF_PATH)
        
        self.f_robot      = f_robot
        self.dt           = 1.0 / f_robot
        self.taskT        = taskT
        self.complete_traj = None
        self.complete_traj_op = None
        self.precompute_custom_lissajous()
        self.des_traj_at = self.custom_lissajous_at
    
    def _load_pin_robot(self, urdf_path):
        """ define robot
        """
        # ---------------------------------------- ROBOT ---------------------------------------- #
        pin_robot = Sim_RR(urdf_path=urdf_path, ee_name='LH_ANKLE')
        # attribute robot
        # pin_robot.setState(q0=torch.tensor([[-torch.pi/2+torch.pi/10],[torch.pi/2+torch.pi/6]]))
        self.u0 = pin_robot.getGravity(pin_robot.q0).clone()
        self.pin_rob = pin_robot
        self.qi = self.pin_rob.q0
        
    @staticmethod
    def _lissajous(t: float) -> np.ndarray:
        """
        Compute position, velocity, and acceleration in Cartesian coordinates.
        """
        ny = 3
        nz = 2
        f0 = 1/4
        ampY = 0.05
        ampZ = 0.05
        k = 6
        dy = 0
        
        dz = k * np.pi / 4 / ny
        a = 2 * np.pi * ny * f0
        b = 2 * np.pi * nz * f0

        y = ampY * np.cos(a * t + dy) + 0.1
        z = ampZ * np.cos(b * t + dz) + 0.2

        vy = - 2 * a * ampY * np.sin(a * t + dy)
        vz = - 2 * b * ampZ * np.sin(b * t + dz)

        ay = - 4 * a * a * ampY * np.cos(a * t + dy)
        az = - 4 * b * b * ampZ * np.cos(b * t + dz)

        des_traj = np.zeros(6)
        des_traj[:2] = [y, z]
        des_traj[2:4] = [vy, vz]
        des_traj[4:6] = [ay, az]

        return torch.as_tensor(des_traj)

    @staticmethod
    def _blender_mjt(t: float, tf: float, x0: float,  dx0: float, ddx0: float, xf: float, dxf: float,  ddxf: float) -> np.ndarray:
        """
        Compute Minimum Jerk Trajectory in 1D.
        """
        # if t < 0:
        #     t = 0.0
        # elif t > tf:
        #     t = tf

        t2 = t**2
        t3 = t**3
        t4 = t**4
        t5 = t**5

        tf2 = tf**2
        tf3 = tf**3
        tf4 = tf**4
        tf5 = tf**5

        C5 = (12*x0 - 12*xf + 6*dxf*tf + 6*dx0*tf - ddxf*tf2 + ddx0*tf2) / (2*tf5)
        C3 = (20*x0 - 20*xf + 8*dxf*tf + 12*dx0*tf - ddxf*tf2 + 3*ddx0*tf2) / (2*tf3)
        C4 = (30*x0 - 30*xf + 14*dxf*tf + 16*dx0*tf - 2*ddxf*tf2 + 3*ddx0*tf2) / (2*tf4)

        x = x0 + dx0*t + (ddx0*t2)/2 - C5*t5 - C3*t3 + C4*t4
        dx = dx0 + ddx0*t - 5*C5*t4 - 3*C3*t2 + 4*C4*t3
        ddx = ddx0 - 20*C5*t3 - 6*C3*t + 12*C4*t2

        return np.array([x, dx, ddx])
    
    def _blended_traj(self, t: float):
        """
        Compute blended Lissajous trajectory with Minimum Jerk Transition.
        """
        
        t_switch = 0.7
        t_delay  = 0.2
        des_traj = self._lissajous(t+t_delay)
        
        des_start = torch.zeros(6, 1).flatten().numpy()
        # des_switch = torch.cat([
        #     self.pin_rob.getForwKinEE(self.qi)[0],
        #     torch.zeros(4, 1)], dim=0
        # ).flatten().numpy()
        des_switch = self._lissajous(t_switch+t_delay)
        # des_start = torch.as_tensor(np.zeros(6))
        # des_switch = torch.as_tensor(np.zeros(6))

        if t < t_switch:
            for i in range(2):
                mjt_point = self._blender_mjt(
                    t=t,
                    tf=t_switch,
                    x0=des_start[i],
                    xf=des_switch[i],
                    dx0=des_start[i+2], 
                    dxf=des_switch[i+2], 
                    ddx0=des_start[i+4], 
                    ddxf=des_switch[i+4]
                )
                
                des_traj[i] = mjt_point[0]
                des_traj[i+2] = mjt_point[1]
                des_traj[i+4] = mjt_point[2]

        des_traj[:2] += self.pin_rob.getForwKinEE(self.qi)[0].flatten()
        return des_traj

    def _inv_kin_traj(self, op_traj: torch.Tensor) -> list[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert into joint space the trajectory.
        """
        from classes.references.classic_ref import InvKin

        x, y = op_traj[0], op_traj[1]
        vx, vy = op_traj[2], op_traj[3]
        ax, ay = op_traj[4], op_traj[5]

        inv_robot = InvKin(robot=self.pin_rob, pf=torch.as_tensor([[x], [y]], dtype=torch.float32))
        # inv_robot.it_MAX = 100
        inv_robot.computeInvKin()
        q_des = inv_robot.get_q()
        # q_des = torch.as_tensor([[0],[0]], dtype=torch.float32)
        
        J_inv = self.pin_rob.getPinvJacPosEE(torch.as_tensor(q_des, dtype=torch.float32), mu = 0.01)
        dq_des = J_inv @ torch.as_tensor([[vx], [vy]], dtype=torch.float32)

        dJ = self.pin_rob.getPinvDotJacPosEE(q=q_des, dq=dq_des, mu=0.01)
        dJ_dot_dq = dJ @ dq_des
        ddq_des = J_inv @ (torch.as_tensor([[ax], [ay]], dtype=torch.float32) - dJ_dot_dq)

        q_des_t = torch.as_tensor(q_des, dtype=torch.float32)
        dq_des_t = torch.as_tensor(dq_des, dtype=torch.float32)
        ddq_des_t = torch.as_tensor(ddq_des, dtype=torch.float32)

        return torch.cat((q_des_t, dq_des_t, ddq_des_t), dim=0).flatten()

    def precompute_custom_lissajous(self):

        time_dim = int(self.taskT * self.f_robot) + 1
        self.complete_traj_op = torch.zeros(6, time_dim)
        self.complete_traj    = torch.zeros(6, time_dim)

        for i in range(time_dim):
            self.complete_traj_op[:, i] = self._blended_traj(i * self.dt)
            self.complete_traj[:, i] = self._inv_kin_traj(self.complete_traj_op[:, i])

    def custom_lissajous_at(self, t: float) -> list[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # if t < 0:
        #     t = 0.0
        # elif t > self.taskT:
        #     t = self.taskT
        
        idx = int(t // self.dt)
        des_traj = self.complete_traj[:, idx]

        q_des   = des_traj[0:2].view(2,1)
        dq_des  = des_traj[2:4].view(2,1)
        ddq_des = des_traj[4:6].view(2,1)
        
        return q_des, dq_des, ddq_des

    def save_trajectory(self, filename: str = "complete_trajectory.pt"):

        if self.complete_traj is not None:
            torch.save(self.complete_traj, filename)
            print(f"Trajectory saved to {filename}")

    def load_trajectory(self, filename: str = "complete_trajectory.pt"):

        if os.path.exists(filename):
            complete_traj = torch.load(filename, weights_only=True)
            print(f"Trajectory loaded from {filename}")
        return complete_traj


if __name__ == "__main__":
    
    obj = LISS_GEN(taskT=5, f_robot=100)
    
    # Save the trajectory to a file named "traj.pt"
    obj.save_trajectory(filename=os.path.join(abs_path,"references/traj.pt"))
    
    complete_traj = obj.load_trajectory(filename=os.path.join(abs_path,"references/traj.pt"))
    
    # Verify that the trajectory was loaded correctly
    if complete_traj is not None:
        print("Loading test successful. Shape of loaded trajectory:", complete_traj.shape)
        
    
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10,8))
    plt.subplot(3,2,1)
    plt.title("Lissajous Curve")
    plt.plot(obj.complete_traj_op[0], obj.complete_traj_op[1])
    plt.grid()
    plt.axis('equal')
    plt.subplot(3,2,2)
    plt.title("Op velocity")
    plt.plot(obj.complete_traj_op[2], label="X")
    plt.plot(obj.complete_traj_op[3], label="Y")
    plt.legend()
    plt.grid()
    plt.subplot(3,2,3)
    plt.title("Joint Space Trajectory")
    plt.plot(obj.complete_traj[0], obj.complete_traj[1], label="Trajectory joint space")
    plt.grid()
    plt.subplot(3,2,4)
    plt.title("Joint Position")
    plt.plot(obj.complete_traj[0], label="Joint 1")
    plt.plot(obj.complete_traj[1], label="Joint 2")
    plt.legend()
    plt.grid()
    plt.subplot(3,2,5)
    plt.title("Joint Velocity")
    plt.plot(obj.complete_traj[2], label="Joint 1")
    plt.plot(obj.complete_traj[3], label="Joint 2")
    plt.legend()
    plt.grid()
    plt.subplot(3,2,6)
    plt.title("Joint Acceleration")
    plt.plot(obj.complete_traj[4], label="Joint 1")
    plt.plot(obj.complete_traj[5], label="Joint 2")
    plt.legend()
    plt.grid()
    # plt.show()

    check_traj = torch.zeros(2, obj.complete_traj.shape[1])
    for i in range(obj.complete_traj.shape[1]):
        pos = obj.pin_rob.getForwKinEE(obj.complete_traj[:2, i])[0]
        check_traj[:, i] = pos.flatten()
    
    plt.figure(figsize=(6,6))
    plt.title("Check Trajectory")
    plt.plot(check_traj[0], check_traj[1], label="Check Trajectory")
    plt.plot(obj.complete_traj_op[0], obj.complete_traj_op[1], label="Original Trajectory", linestyle='dashed')
    plt.legend()
    plt.axis('equal')
    plt.grid()
    plt.show()