"""bfilc_controller.py -- Basis Function ILC (BF-ILC) controller.

Parameterises the feedforward as a linear combination of physically
transferable basis functions:

    u(t) = Psi(t) * theta  ,   Psi in R^{(T*nU) x ntheta}

Each joint gets 4 physics-informed bases -- self acceleration, self damping,
gravity, bias -- avoiding coupling bases that overfit trajectory-specific
kinematics and break zero-shot transfer:

    Joint 1: psi = [q1 , q1 , sin(q1), 1]  (gravity from C1)
    Joint 2: psi = [q2 , q2 , sin(q1+q2), 1]  (gravity from C2)

Learning law (projection-based):

    theta_{j+1} = theta_j + gamma * (PsiTPsi)-1PsiT * L * e_j

Reference:
    - Van de Wijdeven & Bosgra, Int. J. Control, 2010.
    - Bolder & Oomen, IEEE T-CST, 2015.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict


class BFILC:
    """Basis-Function Iterative Learning Control.

    Parameters
    ----------
    dimU : int
        Number of control inputs (joints).
    n_bases : int
        Number of basis functions per joint (default 3: q, q, 1).
    gamma : float
        Learning rate for the projection-based update.
    leak : float
        Leaky learning factor (0 = standard; small non-zero adds robustness).
    threshold : float
        RMSE threshold below which learning stops.
    dtype : torch.dtype
        Tensor data type.
    """

    def __init__(
        self,
        dimU: int,
        n_bases: int = 3,
        gamma: float = 0.5,
        Le: torch.Tensor | float = 1.0,
        Lde: torch.Tensor | float = 0.0,
        leak: float = 0.0,
        threshold: float = 1e-3,
        dtype: torch.dtype = torch.float32,
        cross: bool = False,
        basis_names: list[str] | None = None,
    ):
        self.dimU = dimU
        self.cross = cross

        # Basis function specification
        if basis_names is not None:
            self.basis_names = list(basis_names)
        else:
            # Default: [self-acceleration, damping, gravity, bias]
            self.basis_names = ["ddq", "dq", "sinq", "bias"]

        if cross and dimU > 1 and "ddq_other" not in self.basis_names:
            self.basis_names = self.basis_names + ["ddq_other"]

        self.n_bases = len(self.basis_names)
        self.n_params = dimU * self.n_bases
        self.gamma = gamma
        self.threshold = threshold
        self.dtype = dtype
        self.leak = leak

        if isinstance(Le, (int, float)) or (isinstance(Le, torch.Tensor) and Le.numel() == 1):
            Le = torch.tensor(Le, dtype=dtype).expand(dimU, 1)
        self.Le = Le.type(dtype)
        if isinstance(Lde, (int, float)) or (isinstance(Lde, torch.Tensor) and Lde.numel() == 1):
            Lde = torch.tensor(Lde, dtype=dtype).expand(dimU, 1)
        self.Lde = Lde.type(dtype)

        self.theta = torch.zeros(self.n_params, 1, dtype=dtype)
        self.best_theta = self.theta.clone()

        self.Psi: torch.Tensor | None = None
        self.Psi_pinv: torch.Tensor | None = None
        self.samples = 0

        self.rmse: torch.Tensor | None = None
        self.done = False
        self.uEp: torch.Tensor = torch.zeros(dimU, 1, dtype=dtype)
        self.uk: torch.Tensor = torch.zeros(dimU, 1, dtype=dtype)
        self.idx = 0
        self.episodes = 0
        self.scales: torch.Tensor | None = None

        self.mem: list[TensorDict] = []
        self.__tmplMem = TensorDict(
            {
                "error": torch.Tensor(),
                "dot_error": torch.Tensor(),
                "ddot_error": torch.Tensor(),
                "dddot_error": torch.Tensor(),
                "input": torch.Tensor(),
            },
            batch_size=[],
        )

    def build_basis(self, q_d: torch.Tensor, dq_d: torch.Tensor, ddq_d: torch.Tensor,
                    tau_model: torch.Tensor | None = None,
                    dddq_d: torch.Tensor | None = None,
                    ddddq_d: torch.Tensor | None = None) -> torch.Tensor:
        samples = q_d.shape[1]
        self.samples = samples
        dimU = self.dimU
        n_bases = self.n_bases

        Psi_blocks = []
        for j in range(dimU):
            Psi_j = torch.zeros(samples, n_bases, dtype=self.dtype)
            for k, name in enumerate(self.basis_names):
                if name == "ddq":
                    Psi_j[:, k] = ddq_d[j, :]
                elif name == "tau_model":
                    col = tau_model[j, :] if tau_model is not None else ddq_d[j, :]
                    Psi_j[:, k] = col
                elif name == "dq":
                    Psi_j[:, k] = dq_d[j, :]
                elif name == "sinq":
                    Psi_j[:, k] = torch.sin(q_d[j, :])
                elif name == "bias":
                    Psi_j[:, k] = 1.0
                elif name == "ddq_other":
                    other = (j + 1) % dimU
                    Psi_j[:, k] = ddq_d[other, :]
                elif name == "sinq_other":
                    other = (j + 1) % dimU
                    Psi_j[:, k] = torch.sin(q_d[other, :])
                elif name == "dq_other":
                    other = (j + 1) % dimU
                    Psi_j[:, k] = dq_d[other, :]
                elif name == "q":
                    Psi_j[:, k] = q_d[j, :]
                elif name == "cosq":
                    Psi_j[:, k] = torch.cos(q_d[j, :])
                elif name == "sgn_dq":
                    Psi_j[:, k] = torch.sign(dq_d[j, :])
                elif name == "dq_sq":
                    Psi_j[:, k] = dq_d[j, :] * torch.abs(dq_d[j, :])
                elif name == "dq_prod":
                    Psi_j[:, k] = dq_d[0, :] * dq_d[1, :]
                elif name == "dq_sq_other":
                    other = (j + 1) % dimU
                    Psi_j[:, k] = dq_d[other, :] ** 2
                elif name == "dddq":
                    # dddq_d should be passed or precomputed
                    col = dddq_d[j, :] if dddq_d is not None else torch.zeros(samples)
                    Psi_j[:, k] = col
                elif name == "ddddq":
                    col = ddddq_d[j, :] if ddddq_d is not None else torch.zeros(samples)
                    Psi_j[:, k] = col
                else:
                    raise ValueError(f"Unknown basis function: '{name}'")
            Psi_blocks.append(Psi_j)

        Psi = torch.zeros(samples * dimU, self.n_params, dtype=self.dtype)
        for j in range(dimU):
            rs = j * samples
            re = (j + 1) * samples
            cs = j * n_bases
            ce = (j + 1) * n_bases
            Psi[rs:re, cs:ce] = Psi_blocks[j]

        self.Psi = Psi

        # --- Basis Normalization ---
        # Normalize columns to have unit norm to improve conditioning
        norms = torch.norm(Psi, dim=0, keepdim=True)
        norms[norms < 1e-9] = 1.0
        self.scales = norms.T  # (n_params, 1)
        Psi_norm = Psi / norms

        PsiT_Psi = Psi_norm.T @ Psi_norm
        reg = 0.1 * torch.eye(self.n_params, dtype=self.dtype)  # Lower reg since normalized
        try:
            self.Psi_pinv = torch.linalg.solve(PsiT_Psi + reg, Psi_norm.T)
        except RuntimeError:
            self.Psi_pinv = torch.linalg.lstsq(PsiT_Psi + reg, Psi_norm.T).solution

        return Psi

    def compute_u(self, theta: torch.Tensor | None = None) -> torch.Tensor:
        """Reconstruct feedforward u = Psi * theta.

        Returns
        -------
        u : (dimU, samples) full feedforward signal.
        """
        if theta is None:
            theta = self.theta
        
        # If theta is stored in normalized space, we need to scale it
        # But wait, let's decide: do we store theta in original space or normalized space?
        # If we store it in original space: u = Psi @ theta
        # If we store it in normalized space: u = (Psi/norms) @ theta_norm
        # Let's store it such that theta is "physical", so we scale delta_theta.
        
        u_flat = self.Psi @ theta  # (samples*dimU, 1)
        return u_flat.view(self.samples, self.dimU).T  # (dimU, samples)

    def set_trajectory(self, q_d: torch.Tensor, dq_d: torch.Tensor, ddq_d: torch.Tensor,
                       tau_model: torch.Tensor | None = None,
                       dddq_d: torch.Tensor | None = None,
                       ddddq_d: torch.Tensor | None = None) -> torch.Tensor:
        """Update basis for a *new* trajectory (zero-shot)."""
        self.build_basis(q_d, dq_d, ddq_d, tau_model=tau_model, dddq_d=dddq_d, ddddq_d=ddddq_d)
        self.uEp = self.compute_u()
        return self.uEp

    def _project_update(self, e_j: torch.Tensor, de_j: torch.Tensor | None = None) -> torch.Tensor:
        """Projection-based PD-ILC update with leaky learning.

        theta <- (1 - alpha*gamma) * theta + gamma * (PsiTPsi)-1PsiT * (Lp*e + Ld*e)

        where alpha = leak (0 = standard ILC).
        """
        e_flat = e_j.T.reshape(-1, 1)
        Le_tiled = self.Le.repeat(self.samples, 1).T.reshape(-1, 1)
        combined = Le_tiled * e_flat

        if de_j is not None:
            de_flat = de_j.T.reshape(-1, 1)
            Lde_tiled = self.Lde.repeat(self.samples, 1).T.reshape(-1, 1)
            combined = combined + Lde_tiled * de_flat

        # Combined contains (Lp*e + Ld*e)
        # We project it using the normalized pseudoinverse
        # delta_theta_norm = Psi_norm_pinv @ combined
        # Then we scale it back to original space: delta_theta = delta_theta_norm / scales
        delta_theta_norm = self.gamma * (self.Psi_pinv @ combined)
        delta_theta = delta_theta_norm / self.scales

        if self.leak > 0.0:
            self.theta = (1.0 - self.leak * self.gamma) * self.theta + delta_theta
        else:
            self.theta = self.theta + delta_theta
        return self.theta

    def updateMemError(self, e_: torch.Tensor, de_: torch.Tensor | None = None,
                       dde_: torch.Tensor | None = None, ddde_: torch.Tensor | None = None) -> None:
        if de_ is None:
            de_ = torch.zeros((self.dimU, 1), dtype=self.dtype)
        if dde_ is None:
            dde_ = torch.zeros((self.dimU, 1), dtype=self.dtype)
        if ddde_ is None:
            ddde_ = torch.zeros((self.dimU, 1), dtype=self.dtype)
        for key, val in zip(("error", "dot_error", "ddot_error", "dddot_error"),
                            (e_, de_, dde_, ddde_)):
            tmp = self.mem[-1][key]
            self.mem[-1][key] = torch.cat([tmp.clone(), val.type(self.dtype)], dim=1)

    def updateMemInput(self, u_: torch.Tensor) -> None:
        tmp = self.mem[-1]["input"]
        self.mem[-1]["input"] = torch.cat([tmp.clone(), u_.type(self.dtype)], dim=1)

    def newEp(self) -> None:
        self.mem.append(self.__tmplMem.clone())
        self.episodes += 1
        self.idx = 0

    def stepILC(self) -> None:
        """PD-ILC update: project (Lp*e + Ld*e) onto basis space."""
        if len(self.mem) == 0:
            raise ValueError("BFILC: no episode memory -- call newEp() first.")

        e_old: torch.Tensor = self.mem[-1]["error"]
        de_old: torch.Tensor = self.mem[-1]["dot_error"]
        self.rmse = torch.sqrt(torch.mean(e_old ** 2))

        if self.rmse.item() > self.threshold:
            self._project_update(e_old, de_old if self.Lde.abs().sum().item() > 0 else None)
            self.best_theta = self.theta.clone()
            self.uEp = self.compute_u()
        else:
            if not self.done:
                self.done = True
            self.uEp = self.compute_u()

        self.newEp()

    def getControl(self) -> torch.Tensor:
        """Return u(k) at current timestep."""
        k = self.idx
        if k < self.uEp.shape[1]:
            self.uk = self.uEp[:, k:k + 1]
        self.idx += 1
        return self.uk

    def resetAll(self) -> None:
        self.idx = 0
        self.episodes = 0
        self.mem = []
        self.done = False
        self.rmse = None


    def resetParams(self) -> None:
        """Reset parameters theta to zero (for ablation / comparison)."""
        self.theta = torch.zeros(self.n_params, 1, dtype=self.dtype)
        self.best_theta = self.theta.clone()
        self.done = False
