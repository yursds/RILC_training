
import torch
from classes.controllers.ilc import ILC_base

class NOILC(ILC_base):
    """
    Norm Optimal Iterative Learning Control (NOILC) implementation.
    
    Minimizes J = e_{j+1}.T Q e_{j+1} + (u_{j+1} - u_j).T R (u_{j+1} - u_j)
    Update law: u_{j+1} = u_j + (G.T Q G + R)^{-1} G.T Q e_j
    
    Processing assumes Time-Major flattening for vectors (t=0..T).
    """
    
    def __init__(
        self,
        dimU: int,
        samples: int,
        G: torch.Tensor,
        Q: torch.Tensor,
        R: torch.Tensor,
        **kwargs
    ):
        """
        :param G: Lifted system matrix (Impulse Response), size (dimU*samples, dimU*samples)
        :param Q: State error weight matrix, size (dimU*samples, dimU*samples)
        :param R: Input change weight matrix, size (dimU*samples, dimU*samples)
        """
        super().__init__(dimU, samples, **kwargs)
        
        self.G = G.type(self.dtype)
        self.Q = Q.type(self.dtype)
        self.R = R.type(self.dtype)
        
        # Precompute Gain K = (G^T Q G + R)^-1 G^T Q
        # This can be expensive for large samples, but done once.
        term1 = self.G.T @ self.Q @ self.G + self.R
        term2 = self.G.T @ self.Q
        
        # Use solve instead of inv for better stability if possible, but inv is explicit
        # K = term1^-1 * term2
        try:
            self.K = torch.linalg.inv(term1) @ term2
        except RuntimeError as e:
            print(f"Warning: Matrix inversion failed in NOILC init: {e}")
            self.K = torch.zeros_like(term1) # Fallback or error

    def stepILC(self) -> None:
        """
        Update control using NOILC law.
        """
        if len(self.mem) == 0:
            raise ValueError("ILC first episode is not initialized")
            
        u_old: torch.Tensor = self.mem[-1]["input"] # (dimU, samples)
        e_old: torch.Tensor = self.mem[-1]["error"] # (dimU, samples)
        
        # Compute RMSE (internal use)
        self.rmse = torch.sqrt(torch.mean(e_old ** 2))
        
        if self.rmse.item() <= self.threshold:
            if not self.done:
                self.done = True
            self.uEp = u_old # Keep same input
            self.newEp()
            return

        # Flatten vectors: (dimU, samples) -> (samples, dimU) -> (samples*dimU, 1)
        # This ensures [u(0), u(1), ...] order which corresponds to Lifted Matrix G standard form
        u_old_flat = u_old.permute(1, 0).reshape(-1, 1).type(self.dtype)
        e_old_flat = e_old.permute(1, 0).reshape(-1, 1).type(self.dtype)
        
        # Calculate Delta u
        delta_u = self.K @ e_old_flat
        
        u_new_flat = u_old_flat + delta_u
        
        # Reshape back: (samples*dimU, 1) -> (samples, dimU) -> (dimU, samples)
        self.best_u = u_new_flat.view(self.samples, self.dimU).permute(1, 0)
        
        self.uEp = self.best_u
        self.newEp()
