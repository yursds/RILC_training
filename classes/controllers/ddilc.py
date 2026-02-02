import torch
import mujoco
import numpy as np
from .noilc import NOILC

class DDILC(NOILC):
    """
    Data-Driven Iterative Learning Control (DD-ILC).
    Extends Norm Optimal ILC by identifying the Lifted System Matrix G 
    directly from data (System Identification) via experiments on the plant.
    """
    def __init__(
        self, 
        dimU: int, 
        samples: int, 
        model_mj, 
        data_mj, 
        u_nom: torch.Tensor, 
        dt: float,
        frame_skip: int,
        q_init: np.ndarray,
        dq_init: np.ndarray,
        gravity_comp: np.ndarray,
        scaling: int = 1, # Added scaling
        Q: torch.Tensor = None, 
        R: torch.Tensor = None, 
        threshold: float = 1e-3,
        epsilon: float = 1e-2
    ):
        """
        Initialize DD-ILC. Performs System Identification to construct G.

        :param dimU: Input dimension.
        :param samples: Number of time steps.
        :param model_mj: MuJoCo model (for ID).
        :param data_mj: MuJoCo data (for ID).
        :param u_nom: Nominal control trajectory (around which to identify).
        :param dt: Time step.
        :param frame_skip: Simulation frame skip.
        :param q_init: Initial position.
        :param dq_init: Initial velocity.
        :param gravity_comp: Gravity compensation/Base control for reset.
        :param scaling: Control scaling factor (policy freq / robot freq).
        :param Q: State weight matrix.
        :param R: Input weight matrix.
        :param threshold: Convergence threshold.
        :param epsilon: Perturbation magnitude for ID.
        """
        # Identify G
        self.epsilon = epsilon
        self.scaling = scaling
        G = self.identify_system(model_mj, data_mj, u_nom, samples, dimU, frame_skip, q_init, dq_init, gravity_comp)
        
        # Initialize NOILC with identified G
        super().__init__(dimU, samples, G, Q, R, threshold=threshold)
        
    def identify_system(self, model_mj, data_mj, u_nom, samples, dimU, frame_skip, q_init, dq_init, gravity_comp):
        """
        Constructs G matrix by perturbing the actual MuJoCo plant (System Identification).
        """
        print(f"DDILC: Identifying dynamics from data with epsilon={self.epsilon}, scaling={self.scaling}...")
        
        G = torch.zeros(dimU * samples, dimU * samples)
        
        # Helper to run single rollout
        def run_rollout(u_seq):
            # Reset
            mujoco.mj_resetData(model_mj, data_mj)
            data_mj.qpos[:] = q_init
            data_mj.qvel[:] = dq_init
            mujoco.mj_inverse(model_mj, data_mj) 
            data_mj.ctrl[:] = gravity_comp 
            mujoco.mj_forward(model_mj, data_mj)
            
            y_traj = []
            u_curr_seq = u_seq.clone() 
            
            for k in range(samples):
                ctrl = u_curr_seq[:, k].numpy()
                data_mj.ctrl[:] = ctrl
                
                # HOLD Action for 'scaling' steps logic
                for _ in range(self.scaling):
                    mujoco.mj_step(model_mj, data_mj, nstep=frame_skip)
                
                # Read Output (q) - Assuming 2DOF RR robot structure
                try:
                    q_read = np.concatenate([data_mj.sensor("q_hip").data, data_mj.sensor("q_knee").data])
                except:
                    q_read = data_mj.qpos[:dimU].copy()
                    
                y_traj.append(torch.from_numpy(q_read).float())
                
            return torch.stack(y_traj) 

        # 1. Nominal Rollout
        y_nom = run_rollout(u_nom) 
        
        # 2. Perturbation Rollouts
        for t_impulse in range(samples):
            for i_input in range(dimU):
                # Perturb
                u_p = u_nom.clone()
                u_p[i_input, t_impulse] += self.epsilon
                
                y_p = run_rollout(u_p)
                
                # Difference
                dy = (y_p - y_nom) / self.epsilon
                
                # Fill G column
                col_idx = t_impulse * dimU + i_input
                G[:, col_idx] = dy.reshape(-1)
                
        return G
