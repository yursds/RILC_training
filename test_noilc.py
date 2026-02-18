
from __init__ import *
import torch
import mujoco
from matplotlib import pyplot as plt
import os
import functools

# Add local directory to path
sys.path.append(os.getcwd())

from classes.controllers.noilc import NOILC
from classes.controllers.pd import PD_base
from classes.robots.manipulator_RR import Sim_RR
from classes.environments.env_rlilc_mjc import Env_RILC as ENV

from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

# Configuration constants
abs_path = os.path.join(os.path.dirname((os.path.abspath(__file__))), 'classes')
URDF_PATH = os.path.join(abs_path, 'robots/robot_models/softleg_urdf/urdf/leg_constrained.urdf')
MJC_PATH = os.path.join(abs_path, 'robots/robot_models/softleg_urdf/mjc/scene_test.xml')

parent_str = "model"
dat_str = "rilc_constrained"

# Trajectory parameters
QF = torch.tensor([[2.4], [-1.4]])
TRAJ = "minjerk"

# Helper for MinJerk
def minjerk(qi:torch.Tensor,qf:torch.Tensor,duration:float,t:float) -> list[torch.Tensor,torch.Tensor,torch.Tensor]:
    delta_q = qi-qf
    q_new   = qi + delta_q * (15*(t/duration)**4 - 6*(t/duration)**5 - 10*(t/duration)**3)
    dq_new  = delta_q * (60*(t**3)/(duration**4) - 30*((t**4)/(duration**5)) - 30*(t**2)/(duration**3))
    ddq_new = delta_q * (180*(t**2)/(duration**4) - 120*((t**3)/(duration**5)) - 60*(t/(duration**3)))
    return q_new, dq_new, ddq_new

def angle_normalize(x:torch.Tensor) -> torch.Tensor:
    sx = torch.sin(x)
    cx = torch.cos(x)
    x = torch.atan2(sx,cx)
    return x


def get_linearized_matrices(robot, q, dq, u, dt, damping=True):
    """
    Computes discrete A, B matrices via finite differences around (q, dq, u).
    State x = [q, dq].
    x_{k+1} = f(x_k, u_k) approx x_k + dt * [dq; ddq(q, dq, u)]
    """
    nq = robot._dim_q
    nu = robot._dim_u
    nx = 2 * nq
    
    eps = 1e-4
    
    A = torch.zeros(nx, nx)
    B = torch.zeros(nx, nu)
    
    # Nominal next state
    # We use robot class methods but need to be careful not to mutate robot state permanently
    # robot.getNewState calls _updateState which modifies self.q, self.dq.
    # So we must backup.
    
    q_nom = q.clone()
    dq_nom = dq.clone()
    
    # Save original state
    q_bak = robot.q.clone()
    dq_bak = robot.dq.clone()
    
    # Helper to get next state x_{k+1} from x_k, u_k
    def get_next_state(q_in, dq_in, u_in):
        robot.setState(q=q_in, dq=dq_in)
        # Integration step (RK4 or Euler? Robot uses RK4)
        x_next = robot.getNewState(dt=dt, action=u_in, damp_fl=damping)
        # getNewState returns [q_new, dq_new] list
        return torch.cat([x_next[0], x_next[1]], dim=0) # [q_new; dq_new]
    
    x_nom_next = get_next_state(q, dq, u)
    
    # Compute A = df/dx
    # x = [q; dq]
    # Perturb q
    for i in range(nq):
        q_p = q.clone()
        q_p[i] += eps
        x_p_next = get_next_state(q_p, dq, u)
        A[:, i] = (x_p_next - x_nom_next).flatten() / eps
        
    # Perturb dq
    for i in range(nq):
        dq_p = dq.clone()
        dq_p[i] += eps
        x_p_next = get_next_state(q, dq_p, u)
        A[:, nq + i] = (x_p_next - x_nom_next).flatten() / eps
        
    # Compute B = df/du
    for i in range(nu):
        u_p = u.clone()
        u_p[i] += eps
        x_p_next = get_next_state(q, dq, u_p)
        B[:, i] = (x_p_next - x_nom_next).flatten() / eps
        
    # Restore robot state
    robot.setState(q=q_bak, dq=dq_bak)
    
    return A, B

def construct_lifted_model_nonlinear(robot, q_traj, dq_traj, u_traj, dt, samples, dimU):
    """
    Constructs the Time-Varying Lifted G Matrix by linearizing the full nonlinear robot dynamics
    along the reference trajectory.
    
    Includes M(q), C(q,dq), G(q) effects via the robot's forward dynamics.
    """
    nx = 2 * dimU
    nu = dimU
    
    # Storage for time-varying matrices
    As = [] # A_0, A_1, ...
    Bs = [] # B_0, B_1, ...
    C = torch.cat([torch.eye(dimU), torch.zeros(dimU, dimU)], dim=1) # Output is q (first dimU states)
    
    print("Linearizing dynamics along trajectory...")
    for k in range(samples):
        # Current operating point
        q_k = q_traj[:, k].view(-1,1)
        dq_k = dq_traj[:, k].view(-1,1)
        u_k = u_traj[:, k].view(-1,1)
        
        Ak, Bk = get_linearized_matrices(robot, q_k, dq_k, u_k, dt)
        As.append(Ak)
        Bs.append(Bk)
        
    # Construct Impulse Response (Markov Parameters) for Time-Varying System
    # y(k) = C * x(k)
    # x(k) = A_{k-1} x(k-1) + B_{k-1} u(k-1)
    # => x(k) = sum_{j=0}^{k-1} [Phi(k, j+1) B_j u(j)] + Phi(k,0)x(0)
    # where Phi(k, j) = A_{k-1} ... A_j, Phi(k,k) = I
    # G_{row=k, col=j} = C * Phi(k, j+1) * B_j  (for j < k)
    
    G = torch.zeros(dimU * samples, dimU * samples)
    
    print("Building Lifted G Matrix...")
    # Precompute State Transition Matrices Phi(k, j) could be expensive (samples^2).
    # Iterative build is better.
    
    # We want to map u_flat -> e_flat (or y_flat)
    # y_flat_dim = samples * dimU
    # G is block lower triangular
    
    # Only iterate causal elements
    for c_t in range(samples): # Input time j
        # Propagate impulse from u(j) forward
        # x_impulse(j+1) = B_j * I (since u(j)=Identity for impulse response columns)
        # Actually done column by column of u.
        
        Bj = Bs[c_t]
        
        # We have dimU inputs.
        for input_idx in range(dimU):
            # Impulse on input channel 'input_idx' at time c_t
            # Effect on State at time c_t + 1
            x_curr = Bj[:, input_idx].view(-1,1) 
            
            # Map this state to output at time c_t + 1
            y_curr = C @ x_curr
            
            # Fill G at row (c_t)*dimU creates effect at y(c_t)? No.
            # y(k) depends on u(0)..u(k-1).
            # y(c_t) does NOT depend on u(c_t) if strictly proper (D=0).
            # In update law we align indices.
            # Usually NOILC uses current cycle error.
            # Let's assume standard form: y[k] affected by u[0]...u[k-1].
            # If our samples array includes t=0 to t=T.
            # u[0] affects y[1]...y[T]. y[0] is fixed structure.
            # G matrix usually relates available correction.
            
            # Let's fill for t = c_t + 1 to samples-1
            # For t = c_t, effect is D (assumed 0)
            
            for r_t in range(c_t + 1, samples):
                 # y(r_t) = C * x(r_t)
                 # G entry
                 row_base = r_t * dimU
                 col_idx = c_t * dimU + input_idx
                 
                 G[row_base:row_base+dimU, col_idx] = y_curr.flatten()
                 
                 # Propagate state to next step
                 # x(r_t+1) = A(r_t) * x(r_t)
                 if r_t < samples - 1:
                     x_curr = As[r_t] @ x_curr
                     y_curr = C @ x_curr
                     
    return G

if __name__ == '__main__':
    visual = True 
    
    kp = 0.4
    kv = 0.25
    scaling = 2
    f_robot = 100
    taskT = 1.0
    n_ep_reset = 20 # More episodes to see convergence
    le = 0.0002 
    
    model = mujoco.MjModel.from_xml_path(MJC_PATH)
    data = mujoco.MjData(model)
    
    __actual_dt = model.opt.timestep
    frame_skip = int((1/f_robot)/__actual_dt)
    
    f_policy = int(f_robot / scaling)
    samples = int(taskT*f_policy) + 1
    
    noise_q_dev = 1e-6
    noise_dq_dev = 2.5e-4
    njoint = 2
    
    env = ENV(taskT=taskT, f_robot=f_robot, scaling=scaling, le=le, lde=0, ldde=0, kp=kp, kv=kv, n_ep_reset=n_ep_reset)
    
    dt_pol = 1/f_policy
    dt_rob = 1/f_robot
    
    # Robot
    robot = Sim_RR(urdf_path=URDF_PATH, ee_name='LH_ANKLE')
    
    if TRAJ == "minjerk":
        des_traj_at = functools.partial(minjerk, qi = torch.tensor([[0.0], [0.0]]), qf = QF, duration = taskT)
    
    # Logging Lists
    e_list = []
    de_list = []
    dde_list = []
    q_list = []
    dq_list = []
    ddq_list = []
    uILC_list = []
    uFB_list = []
    uMB_list = []
    uRL_list = [] # Just zeros
    
    r_list = torch.zeros(2, samples)
    dr_list = torch.zeros(2, samples)
    ddr_list = torch.zeros(2, samples)
    
    # Ref Traj
    for i in range(samples):
        t = i*dt_pol
        r, dr, ddr = des_traj_at(t=t)
        r_list[:,i] = r.flatten()
        dr_list[:,i] = dr.flatten()
        ddr_list[:,i] = ddr.flatten()


    # Helpers
    tmp_q = des_traj_at(t=0.0)[0].clone()
    tmp_dq = des_traj_at(t=0.0)[1].clone()
    robot.setState(q0=tmp_q, dq0=tmp_dq, q=tmp_q, dq=tmp_dq)
    qi = robot.q0.clone()
    qvel_init = robot.dq0.flatten().numpy().copy()
    qpos_init = robot.q0.flatten().numpy().copy()
    
    # -----------------------------
    # 1. PRE-COMPUTE TRAJECTORY inputs for Linearization
    # -----------------------------
    # We need a nominal u_traj to linearize around.
    # Ideally use Inverse Dynamics u = ID(q_ref, dq_ref, ddq_ref).
    # This gives the best feedforward model "around" the trajectory.
    u_traj_ref = torch.zeros(njoint, samples)
    q_traj_ref = torch.zeros(njoint, samples)
    dq_traj_ref = torch.zeros(njoint, samples)
    
    for i in range(samples):
        t_val = i * dt_pol
        r_val, dr_val, ddr_val = des_traj_at(t=t_val)
        
        # Store State
        q_traj_ref[:, i] = r_val.flatten()
        dq_traj_ref[:, i] = dr_val.flatten()
        
        # Compute Inverse Dynamics Model u (uMB + uDyn)
        # We need the robot to compute this.
        # robot.getInvDyn(q, dq, ddq)
        # Careful: robot state might be updated by getInvDyn? No, usually not.
        tau_ref = robot.getInvDyn(r_val, dr_val, ddr_val, damp_fl=True)
        u_traj_ref[:, i] = tau_ref.flatten()

    # -----------------------------
    # 2. CONSTRUCT NONLINEAR LIFTED MODEL
    # -----------------------------
    # Use the full nonlinear dynamics (M, C, G) via finite difference linearization
    G_mat = construct_lifted_model_nonlinear(robot, q_traj_ref, dq_traj_ref, u_traj_ref, dt=dt_pol, samples=samples, dimU=njoint)
    
    # Conservative Gains
    # With accurate G, we can be slightly more aggressive or keep robust settings.
    q_weight = 0.5
    r_weight = 10.0 
    
    Q_mat = q_weight * torch.eye(njoint * samples)
    R_mat = r_weight * torch.eye(njoint * samples)
    
    noilc_ctrl = NOILC(dimU=njoint, samples=samples, G=G_mat, Q=Q_mat, R=R_mat, threshold=1e-4)
    noilc_ctrl.newEp()
    
    if visual:
        mujoco_renderer = MujocoRenderer(model, data, None, 800, 600)
    
    for ep in range(n_ep_reset):
        print(f"Episode {ep}")
        if ep > 0:
            noilc_ctrl.stepILC()
        
        # Reset
        mujoco.mj_resetData(model, data)
        data.qpos = qpos_init
        data.qvel = qvel_init
        mujoco.mj_inverse(model, data)
        data.ctrl[:] = robot.getGravity(q=qi).flatten()
        mujoco.mj_forward(model, data)
        
        dq_old = torch.as_tensor(data.qvel).view(2,1).clone()
        
        # Per Episode Logs
        e_tmp, de_tmp, dde_tmp = [], [], []
        q_tmp, dq_tmp, ddq_tmp = [], [], []
        uILC_tmp, uFB_tmp, uMB_tmp, uRL_tmp_log = [], [], [], []
        
        t = 0.0
        noilc_ctrl.idx = 0 
        
        for i in range(samples):
            r_, dr_, ddr_ = des_traj_at(t=t)
            
            q_curr = torch.zeros(2,1)
            dq_curr = torch.zeros(2,1)
            q_curr[0] = torch.from_numpy(data.sensor("q_hip").data)
            q_curr[1] = torch.from_numpy(data.sensor("q_knee").data)
            dq_curr[0] = torch.from_numpy(data.sensor("dq_hip").data)
            dq_curr[1] = torch.from_numpy(data.sensor("dq_knee").data)
            
            q_curr += noise_q_dev * torch.randn(2,1)
            dq_curr += noise_dq_dev * torch.randn(2,1)
            
            ddq_curr = (dq_curr - dq_old)*f_robot
            dq_old = dq_curr.clone()
            
            e_ = angle_normalize(r_ - q_curr)
            de_ = dr_ - dq_curr
            dde_ = ddr_ - ddq_curr
            
            # Update Memory
            noilc_ctrl.updateMemError(e_=e_, de_=de_, dde_=dde_)
            
            # Control Calculation
            uMB = robot.getGravity(q=q_curr)
            uFB = torch.matmul(torch.diag(torch.tensor([kp, kp])), e_) + \
                  torch.matmul(torch.diag(torch.tensor([kv, kv])), de_)
            
            if ep > 0:
                uILC = noilc_ctrl.getControl()
            else:
                uILC = torch.zeros(2,1)
                noilc_ctrl.idx += 1
            
            # CRITICAL FIX based on test_rilc.py: 
            # The ILC memory must incorporate the Total Feedforward input, 
            # which effectually becomes uILC(new) = uILC(old) + uFB(old) + Learning
            # So we pass uInput = uILC + uFB
            noilc_ctrl.updateMemInput(uILC + uFB)
            
            # Log
            e_tmp.append(e_.flatten().clone())
            de_tmp.append(de_.flatten().clone())
            dde_tmp.append(dde_.flatten().clone())
            q_tmp.append(q_curr.flatten().clone())
            dq_tmp.append(dq_curr.flatten().clone())
            ddq_tmp.append(ddq_curr.flatten().clone())
            uMB_tmp.append(uMB.flatten().clone())
            uFB_tmp.append(uFB.flatten().clone())
            uILC_tmp.append(uILC.flatten().clone())
            uRL_tmp_log.append(torch.zeros(2).clone()) # Dummy for plot consistency
            
            uTot = uMB + uFB + uILC
            
            for _ in range(scaling):
                data.ctrl[:] = uTot.flatten().numpy()
                mujoco.mj_step(model, data, nstep=frame_skip)
                mujoco.mj_rnePostConstraint(model, data)
                t += dt_rob
            
            if visual:
                mujoco_renderer.render("human")
        
        # Append Ep Logs
        e_list.append(e_tmp)
        de_list.append(de_tmp)
        dde_list.append(dde_tmp)
        q_list.append(q_tmp)
        dq_list.append(dq_tmp)
        ddq_list.append(ddq_tmp)
        uMB_list.append(uMB_tmp)
        uFB_list.append(uFB_tmp)
        uILC_list.append(uILC_tmp)
        uRL_list.append(uRL_tmp_log)

    if visual:
        mujoco_renderer.close()

    # ---- PLOTTING (Matched to test_rilc.py) ----
    
    # 1. Console print
    for i in range(len(e_list)):
        rmse_list = torch.sqrt(torch.mean(torch.stack(e_list[i])**2))
        print(f"rilc MSE of episode: {i}", rmse_list)
        
    # 2. Detailed Plot First and Last Episode
    for i in [0, n_ep_reset-1]:
        plt.figure(figsize=(8, 8))
        plt.subplot(2,3,1)
        plt.plot(torch.stack(e_list[i]).T[0,:], label="sim e1")
        plt.plot(torch.stack(e_list[i]).T[1,:], label="sim e2")
        plt.xlabel("Time steps")
        plt.ylabel("Error [$rad$]")
        plt.title(f"Error")
        plt.grid()
        plt.subplot(2,3,2)
        plt.plot(torch.stack(de_list[i]).T[0,:], label="sim de1")
        plt.plot(torch.stack(de_list[i]).T[1,:], label="sim de2")
        plt.xlabel("Time steps")
        plt.ylabel("Dot error [$rad/s$]")
        plt.title(f"Dot Error")
        plt.grid()    
        plt.subplot(2,3,3)
        plt.plot(torch.stack(dde_list[i]).T[0,:], label="sim dde1")
        plt.plot(torch.stack(dde_list[i]).T[1,:], label="sim dde2")
        plt.xlabel("Time steps")
        plt.ylabel("DDot error [$rad/s^2$]")
        plt.title(f"DDot Error  ")
        plt.grid()    
        plt.subplot(2,3,4)
        plt.plot(torch.stack(q_list[i]).T[0,:], label="sim q1")
        plt.plot(torch.stack(q_list[i]).T[1,:], label="sim q2")
        plt.plot(r_list[0,:], label="ref q1")
        plt.plot(r_list[1,:], label="ref q2")
        plt.xlabel("Time steps")
        plt.ylabel("Angle [$rad$]")
        plt.legend()
        plt.title(f"Joints' Angle in episode  {i+1}")
        plt.grid()
        plt.subplot(2,3,5)
        plt.plot(torch.stack(dq_list[i]).T[0,:], label="sim dq1")
        plt.plot(torch.stack(dq_list[i]).T[1,:], label="sim dq2")
        plt.plot(dr_list[0,:], label="ref dq1")
        plt.plot(dr_list[1,:], label="ref dq2")
        plt.xlabel("Time steps")
        plt.ylabel("Dot Angle [$rad/s$]")
        plt.title(f"Joints' Dot Angle")
        plt.grid()
        plt.legend()
        plt.subplot(2,3,6)
        plt.plot(torch.stack(ddq_list[i]).T[0,:], label="sim ddq1")
        plt.plot(torch.stack(ddq_list[i]).T[1,:], label="sim ddq2")
        plt.plot(ddr_list[0,:], label="ref ddq1")
        plt.plot(ddr_list[1,:], label="ref ddq2")
        plt.xlabel("Time steps")
        plt.ylabel("DDot Angle [$rad/s^2$]")
        plt.title(f"Joints' DDot Angle")
        plt.legend()
        plt.grid()
        plt.suptitle(f"ILC in  episode {i+1}")
        plt.tight_layout()
        plt.savefig(f"noilc_detailed_ep_{i}.png")

    # 3. Control Components Plot
    for i in [0, n_ep_reset-1]:
        plt.figure(figsize=(15, 3))
        
        plt.subplot(1, 6, 1)
        uT = torch.stack(uMB_list[i]) + torch.stack(uILC_list[i]) + torch.stack(uFB_list[i])
        plt.plot(uT[:,0])
        plt.plot(uT[:,1])
        plt.title("uTOT")
        plt.grid()
        
        plt.subplot(1, 6, 2)
        plt.plot(torch.stack(uILC_list[i])[:,0]) # RL is 0
        plt.plot(torch.stack(uILC_list[i])[:,1])
        plt.title("uRL+uILC")
        plt.grid()
        
        plt.subplot(1, 6, 3)
        plt.plot(torch.stack(uMB_list[i])[:,0])
        plt.plot(torch.stack(uMB_list[i])[:,1])
        plt.title("uMB")
        plt.grid()
        
        plt.subplot(1, 6, 4)
        plt.plot(torch.stack(uILC_list[i])[:,0])
        plt.plot(torch.stack(uILC_list[i])[:,1])
        plt.title("uILC")
        plt.grid()
        
        plt.subplot(1, 6, 5)
        plt.plot(torch.stack(uFB_list[i])[:,0])
        plt.plot(torch.stack(uFB_list[i])[:,1])
        plt.title("uFB")
        plt.grid()
        
        plt.subplot(1, 6, 6)
        plt.plot(torch.stack(uRL_list[i])[:,0]) # zeros
        plt.plot(torch.stack(uRL_list[i])[:,1])
        plt.title("uRL")
        plt.grid()
        
        plt.suptitle(f"ILC Episode {i+1}")
        plt.tight_layout()
        plt.savefig(f"noilc_controls_ep_{i}.png")
        
    print("All plots saved.")
