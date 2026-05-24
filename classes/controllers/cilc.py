"""cilc_controller.py -- Combined ILC (Tsurumoto et al., IFAC 2023)

Implements equations 30 and 32 from:
  Tsurumoto et al., "Task flexible and high performance ILC:
  Preliminary analysis of combining a basis function",
  IFAC-PapersOnLine 56-2 (2023) 1907--1912.

Total feedforward:  u_j = u_j^theta + u_j^{ILC}

  u_j^theta = Psi * theta_j          (basis function component, task-flexible)
  u_j^{ILC}                (frequency-domain component, high-performance)

theta update (eq. 30, with Q_theta=I, SG~=I):
  theta_{j+1} = theta_j + (PsiTPsi)-1PsiT * (e_j + u_j^{ILC})

ILC update (eq. 32, with Q=I, H=1):
  u_{j+1}^{ILC} = u_j^{ILC} + L*e_j + (u_j^theta - u_{j+1}^theta)

Key idea: the correction term (u_j^theta - u_{j+1}^theta) prevents the ILC from
"fighting" the basis function update -- when theta changes, the ILC adjusts
so the total signal changes smoothly.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict
from classes.controllers.bfilc import BFILC


class CILC:
    """C-ILC controller per Tsurumoto et al. (IFAC 2023).

    Decompose feedforward into basis-function (transferable) and
    ILC (residual) components. theta update uses virtual error (eq. 30),
    ILC update includes correction for theta change (eq. 32).

    Parameters
    ----------
    dimU : int
        Number of control inputs (joints).
    basis_names : list[str] | None
        Basis function names per joint (default: 4 physics bases).
    gamma : float
        Learning rate for theta update (default 0.5).
    Le : torch.Tensor | float
        Error learning gain for ILC component.
    Lde : torch.Tensor | float
        Derivative error learning gain for ILC component.
    threshold : float
        RMSE threshold below which learning stops.
    dtype : torch.dtype
        Tensor data type.
    """

    def __init__(
        self,
        dimU: int,
        basis_names: list[str] | None = None,
        gamma: float = 0.5,
        Le: torch.Tensor | float = 1.0,
        Lde: torch.Tensor | float = 0.0,
        threshold: float = 1e-3,
        dtype: torch.dtype = torch.float32,
    ):
        self.dimU = dimU
        self.threshold = threshold
        self.dtype = dtype

        # Basis function engine (handles Psi building and pseudoinverse)
        self.bfilc = BFILC(
            dimU=dimU, basis_names=basis_names, gamma=gamma,
            Le=Le, Lde=Lde, threshold=threshold, dtype=dtype,
        )
        self.Le = self.bfilc.Le
        self.Lde = self.bfilc.Lde

        self.samples = 0

        # ILC component (per-sample, learned)
        self.u_ILC: torch.Tensor | None = None
        # Basis component (Psi*theta)
        self.u_BF: torch.Tensor | None = None

        self.uk = torch.zeros(dimU, 1, dtype=dtype)
        self.uEp: torch.Tensor | None = None
        self.idx = 0
        self.rmse: torch.Tensor | None = None
        self.done = False
        self.mem: list[TensorDict] = []
        self.episodes = 0

        self.__tmplMem = TensorDict({
            "error": torch.Tensor(),
            "dot_error": torch.Tensor(),
            "ddot_error": torch.Tensor(),
            "dddot_error": torch.Tensor(),
            "input_ilc": torch.Tensor(),   # stores ONLY u^{ILC} (eq. 32 needs it)
        }, batch_size=[])

    # -- Properties forwarded to BFILC ------------------------------

    @property
    def theta(self) -> torch.Tensor:
        return self.bfilc.theta

    @theta.setter
    def theta(self, val: torch.Tensor):
        self.bfilc.theta = val
        self.bfilc.best_theta = val.clone()

    @property
    def Psi(self) -> torch.Tensor:
        return self.bfilc.Psi

    @property
    def n_params(self) -> int:
        return self.bfilc.n_params

    def compute_u(self, theta: torch.Tensor | None = None) -> torch.Tensor:
        return self.bfilc.compute_u(theta)

    # -- Trajectory setup -------------------------------------------

    def set_trajectory(
        self, q_d: torch.Tensor, dq_d: torch.Tensor, ddq_d: torch.Tensor,
        tau_model: torch.Tensor | None = None,
        dddq_d: torch.Tensor | None = None,
        ddddq_d: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.bfilc.set_trajectory(q_d, dq_d, ddq_d, tau_model=tau_model,
                                  dddq_d=dddq_d, ddddq_d=ddddq_d)
        self.samples = q_d.shape[1]

        if self.u_ILC is None or self.u_ILC.shape[1] != self.samples:
            self.u_ILC = torch.zeros(self.dimU, self.samples, dtype=self.dtype)

        self.u_BF = self.bfilc.compute_u()
        self.uEp = self.u_ILC + self.u_BF
        return self.uEp

    # -- Memory -----------------------------------------------------

    def updateMemError(
        self, e_: torch.Tensor, de_: torch.Tensor | None = None,
        dde_: torch.Tensor | None = None, ddde_: torch.Tensor | None = None,
    ) -> None:
        if de_ is None:
            de_ = torch.zeros((self.dimU, 1), dtype=self.dtype)
        if dde_ is None:
            dde_ = torch.zeros((self.dimU, 1), dtype=self.dtype)
        if ddde_ is None:
            ddde_ = torch.zeros((self.dimU, 1), dtype=self.dtype)
        for key, val in zip(
            ("error", "dot_error", "ddot_error", "dddot_error"),
            (e_, de_, dde_, ddde_),
        ):
            tmp = self.mem[-1][key]
            self.mem[-1][key] = torch.cat([tmp.clone(), val.type(self.dtype)], dim=1)

    def updateMemInput(self, u_: torch.Tensor) -> None:
        """No-op: u^{ILC} is auto-logged in getControl() for the paper's C-ILC."""
        pass

    # -- Episode management -----------------------------------------

    def newEp(self) -> None:
        self.mem.append(self.__tmplMem.clone())
        self.episodes += 1
        self.idx = 0

    # -- Control output ---------------------------------------------

    def getControl(self) -> torch.Tensor:
        """Return u(k) = u^{ILC}(k) + u^theta(k). Auto-logs u^{ILC} for stepILC."""
        k = self.idx
        if k < self.samples:
            u_ilc_k = self.u_ILC[:, k:k+1]
            u_bf_k  = self.u_BF[:, k:k+1]
            self.uk = u_ilc_k + u_bf_k
            # Auto-log u^{ILC} for eq. 32 update (avoids needing separate store call)
            if len(self.mem) > 0:
                tmp = self.mem[-1]["input_ilc"]
                self.mem[-1]["input_ilc"] = torch.cat([tmp.clone(), u_ilc_k], dim=1)
        self.idx += 1
        return self.uk

    # -- Learning: eq. 30 + eq. 32 ----------------------------------

    def stepILC(self) -> None:
        """Implement Tsurumoto C-ILC update (eq. 30 + eq. 32)."""
        if len(self.mem) == 0:
            raise ValueError("CILC: no episode memory -- call newEp() first.")

        e_old: torch.Tensor = self.mem[-1]["error"]
        de_old: torch.Tensor = self.mem[-1]["dot_error"]
        u_ilc_old: torch.Tensor = self.mem[-1]["input_ilc"]  # u_j^{ILC}
        self.rmse = torch.sqrt(torch.mean(e_old ** 2))

        if self.rmse.item() > self.threshold:
            # -- Step 1: ILC correction L*e -------------------------
            e_flat = e_old.T.reshape(-1, 1)
            Le_tiled = self.Le.repeat(self.samples, 1).T.reshape(-1, 1)
            correction = Le_tiled * e_flat
            if self.Lde.abs().sum().item() > 0:
                de_flat = de_old.T.reshape(-1, 1)
                Lde_tiled = self.Lde.repeat(self.samples, 1).T.reshape(-1, 1)
                correction = correction + Lde_tiled * de_flat

            # -- Step 2: theta update (eq. 30) --------------------------
            #   theta_{j+1} = theta_j + (PsiTPsi)-1PsiT * (e_j + u_j^{ILC})
            #   Virtual error = e_j + u_j^{ILC}  (SG~=I)
            u_ilc_flat = u_ilc_old.T.reshape(-1, 1)
            virtual_error_flat = e_flat + u_ilc_flat
            delta_theta_norm = self.bfilc.Psi_pinv @ virtual_error_flat
            delta_theta = delta_theta_norm / self.bfilc.scales
            theta_new = self.bfilc.theta + delta_theta
            self.bfilc.theta = theta_new
            self.bfilc.best_theta = theta_new.clone()

            # -- Step 3: new basis component ------------------------
            u_bf_new = self.bfilc.compute_u()                 # u_{j+1}^theta
            u_bf_old_flat = self.u_BF.T.reshape(-1, 1)        # u_j^theta (flattened)
            u_bf_new_flat = u_bf_new.T.reshape(-1, 1)          # u_{j+1}^theta

            # -- Step 4: ILC update (eq. 32, Q=I, H=1) -------------
            #   u_{j+1}^{ILC} = u_j^{ILC} + L*e_j + (u_j^theta - u_{j+1}^theta)
            u_ilc_new_flat = u_ilc_flat + correction + (u_bf_old_flat - u_bf_new_flat)

            # -- Step 5: update state -------------------------------
            self.u_ILC = u_ilc_new_flat.view(self.samples, self.dimU).T
            self.u_BF = u_bf_new

            # Total feedforward for next episode
            self.uEp = self.u_ILC + self.u_BF
        else:
            if not self.done:
                self.done = True

        self.newEp()

    # -- Utilities --------------------------------------------------

    def resetAll(self) -> None:
        self.idx = 0
        self.episodes = 0
        self.mem = []
        self.done = False
        self.rmse = None
        self.bfilc.resetAll()

    def resetParams(self) -> None:
        self.bfilc.resetParams()
        self.u_ILC = None
        self.u_BF = None
        self.uEp = None
